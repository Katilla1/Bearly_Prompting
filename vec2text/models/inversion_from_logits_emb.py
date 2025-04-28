import copy
from typing import Dict, Optional, Tuple, Any

import transformers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from vec2text.models.config import InversionConfig
from vec2text.models.inversion_from_logits import InversionFromLogitsModel
from vec2text.tokenize_data import get_tokenizer_mapping


class InversionFromLogitsEmbModel(InversionFromLogitsModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.embedding_proj = keras.Sequential([
            layers.Dense(self.embedder_dim, input_shape=(None, self.encoder_hidden_dim)),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim),
        ])
        word_embeddings = self.encoder_decoder.shared
        inverter_vocab_size = tf.shape(word_embeddings)[0]
        self.num_tokens = 64
        self.num_zeros_to_add = (self.num_tokens - (inverter_vocab_size % self.num_tokens)) % self.num_tokens
        word_embedding_zeros = tf.zeros(
            [self.num_zeros_to_add, tf.shape(word_embeddings)[1]],
            dtype=word_embeddings.dtype,
        )
        padded = tf.concat([word_embeddings, word_embedding_zeros], axis=0)
        word_embeddings = tf.reshape(padded, [self.num_tokens, -1, tf.shape(word_embeddings)[1]])
        self.word_embeddings = tf.Variable(word_embeddings, trainable=False, name="word_embeddings")
        self.minibatch_size = 128
        self.unigram_beta = 0.01
        self.unigram = tf.Variable(
            tf.zeros([1, inverter_vocab_size], dtype=tf.float32),
            trainable=False,
            name="unigram",
        )
        mapping = get_tokenizer_mapping(
            config.embedder_model_name,
            config.model_name_or_path,
            self.encoder_decoder.config.vocab_size,
        )
        self.tokenizer_mapping = tf.constant(mapping.numpy(), dtype=tf.int32)

    def embed_and_project(
        self,
        input_ids: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor],
        frozen_embeddings: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
        elif self.embedder_no_grad:
            embeddings = tf.stop_gradient(self.call_embedding_model(input_ids, attention_mask))
        else:
            embeddings = self.call_embedding_model(input_ids, attention_mask)
        next_predictions = tf.argmax(embeddings, axis=1, output_type=tf.int32)
        vocab_len = tf.size(self.tokenizer_mapping)
        embeddings = embeddings[:, :vocab_len]
        batch_size = tf.shape(embeddings)[0]
        new_emb = tf.zeros([batch_size, self.encoder_decoder.config.vocab_size], dtype=tf.float64)
        updates = tf.exp(tf.cast(embeddings, tf.float64))
        batch_idx = tf.repeat(tf.reshape(tf.range(batch_size), (-1,1)), vocab_len, axis=1)
        scatter_idx = tf.stack([
            tf.reshape(batch_idx, [-1]),
            tf.reshape(self.tokenizer_mapping, [-1])
        ], axis=1)
        new_emb = tf.tensor_scatter_nd_add(new_emb, scatter_idx, tf.reshape(updates, [-1]))
        embeddings = tf.math.log(new_emb + 1e-9)
        embeddings = tf.where(tf.math.is_nan(embeddings), tf.zeros_like(embeddings), embeddings)
        if training:
            uni_batch = tf.reduce_mean(embeddings, axis=0, keepdims=True)
            self.unigram.assign((1-self.unigram_beta)*self.unigram + self.unigram_beta*uni_batch)
        embeddings = embeddings - self.unigram
        zeros = tf.zeros([batch_size, self.num_zeros_to_add], dtype=embeddings.dtype)
        logits = tf.cast(tf.concat([embeddings, zeros], axis=1), self.sequence_weights.dtype)
        logits = tf.reshape(logits, [batch_size, self.num_tokens, -1])
        parts = []
        for i in range(0, batch_size, self.minibatch_size):
            block = logits[i:i+self.minibatch_size]
            parts.append(tf.einsum("smd,bsm->bsd", self.word_embeddings, block))
        embeddings = tf.concat(parts, axis=0)
        embeddings = self.embedding_proj(embeddings)
        attention_mask_out = tf.ones([batch_size, self.num_tokens], tf.int32)
        return embeddings, attention_mask_out, next_predictions
    
    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any],
    ) -> tf.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask, _ = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
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
        
    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
        decoder_input_ids: Optional[tf.Tensor] = None,
        training: bool = False,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.TFSeq2SeqLMOutput:
        inputs_embeds, attn_mask, _ = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=kwargs.get("frozen_embeddings"),
            training=training
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            training=training,
        )
