import copy
from typing import Dict, Optional, Tuple, Any

import transformers
import tensorflow as tf

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel
from vec2text.models.inversion_from_logits_emb import InversionFromLogitsEmbModel

LOGIT_FILTER_VALUE = -1 * 10**7

class InversionFromMultipleLogitsModel(InversionFromLogitsEmbModel):
    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any],
    ) -> tf.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        input_ids = inputs.get("input_ids")
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
        )
        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=inputs["decoder_input_ids"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

    def get_frozen_embeddings(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        embeddings = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        next_predictions = tf.argmax(embeddings, axis=1, output_type=tf.int32)
        second_input_ids = tf.concat(
            [input_ids, tf.expand_dims(next_predictions, axis=1)], axis=1
        )
        second_attention_mask = tf.ones_like(second_input_ids)
        second_embeddings = self.call_embedding_model(
            input_ids=second_input_ids,
            attention_mask=second_attention_mask,
        )
        all_embeddings = tf.concat([embeddings, second_embeddings], axis=1)
        return all_embeddings

    def embed_and_project(
        self,
        input_ids: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor],
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
        else:
            embeddings = self.call_embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            next_preds = tf.argmax(embeddings, axis=1, output_type=tf.int32)
            second_ids = tf.concat(
                [input_ids, tf.expand_dims(next_preds, 1)], axis=1
            )
            second_mask = tf.ones_like(second_ids)
            second_emb = self.call_embedding_model(
                input_ids=second_ids,
                attention_mask=second_mask,
            )
            embeddings = tf.concat([embeddings, second_emb], axis=1)
        vocab_len = tf.size(self.tokenizer_mapping)
        embeddings = embeddings[:, :vocab_len]
        batch_sz = tf.shape(embeddings)[0]
        new_emb = tf.zeros(
            [batch_sz, self.encoder_decoder.config.vocab_size],
            dtype=embeddings.dtype,
        )
        mapping = tf.broadcast_to(
            tf.expand_dims(self.tokenizer_mapping, 0),
            [batch_sz, vocab_len],
        )
        updates = tf.exp(embeddings)
        batch_idx = tf.repeat(tf.reshape(tf.range(batch_sz), (-1, 1)), vocab_len, axis=1)
        scatter_idx = tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(mapping, [-1])], axis=1)
        new_emb = tf.tensor_scatter_nd_add(new_emb, scatter_idx, tf.reshape(updates, [-1]))
        embeddings = tf.math.log(new_emb + 1e-9)
        embeddings = tf.nan_to_num(embeddings)
        if training := kwargs.get("training", False):
            uni_batch = tf.reduce_mean(embeddings, axis=0, keepdims=True)
            self.unigram.assign(
                (1 - self.unigram_beta) * self.unigram + self.unigram_beta * uni_batch
            )
        embeddings = embeddings - self.unigram
        zeros = tf.zeros([batch_sz, self.num_zeros_to_add], dtype=embeddings.dtype)
        logits = tf.cast(tf.concat([embeddings, zeros], axis=1), self.sequence_weights.dtype)
        num_tokens = self.num_tokens
        logits = tf.reshape(logits, [batch_sz, num_tokens, -1])
        parts = []
        for i in range(0, batch_sz, self.minibatch_size):
            part = tf.einsum("smd,bsm->bsd", self.word_embeddings, logits[i : i + self.minibatch_size])
            parts.append(part)
        embeddings = tf.concat(parts, axis=0)
        embeddings = self.embedding_proj(embeddings)
        attention_mask_out = tf.ones([batch_sz, num_tokens], tf.int32)
        return embeddings, attention_mask_out

        
    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
        decoder_input_ids: Optional[tf.Tensor] = None,
        training=False,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.TFSeq2SeqLMOutput:
        inputs_embeds, attn_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=kwargs.get("frozen_embeddings"),
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            training=training,
        )