import copy
from typing import Dict, Optional, Tuple, Any

import transformers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel

LOGIT_FILTER_VALUE = -1 * 10**7

# TODO: Remove conflicting duplicate features: zero-except-top-k and
# emb-top-k.


def zero_embedding_except_topk(
    embeddings: tf.Tensor,
    vocab_size: int,
    k: int,
    default_val: float,
) -> tf.Tensor:
    topk = tf.math.top_k(embeddings[:, :vocab_size], k=k, sorted=False)
    values, indices = topk.values, topk.indices
    new_emb = tf.fill(tf.shape(embeddings), default_val)
    batch_size = tf.shape(embeddings)[0]
    batch_idx = tf.broadcast_to(
        tf.reshape(tf.range(batch_size), [batch_size, 1]), [batch_size, k]
    )
    scatter_idx = tf.stack([batch_idx, indices], axis=2)
    updates = tf.scatter_nd(
        tf.reshape(scatter_idx, [-1, 2]), 
        tf.reshape(values, [-1]), 
        tf.shape(embeddings)
    )
    return new_emb + updates


class InversionFromLogitsModel(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config)
        assert hasattr(self.embedder, "generate")  
        H = self.encoder_decoder.config.hidden_size
        self.encoder_hidden_dim = H
        self.embedder_is_decoder = True
        vocab = self.embedder.config.vocab_size
        self.num_zeros_to_add = H - ((vocab + H) % H)
        self.num_repeat_tokens = (vocab + self.num_zeros_to_add) // H
        bottleneck = self.bottleneck_dim
        self.embedding_transform = tf.keras.Sequential([
            layers.Dense(bottleneck, input_shape=(None, H)),
            layers.Dropout(self.encoder_decoder.config.dropout_rate),
            layers.Activation(activations.gelu),
            layers.Dense(H),
        ])
        self.sequence_weights = tf.Variable(
            tf.random.normal(
                [self.num_repeat_tokens, H, H], dtype=tf.float32
            ),
            trainable=True,
            name="sequence_weights",
        )
        self.unigram_beta = 0.01
        self.unigram = tf.Variable(
            tf.zeros([1, vocab + self.num_zeros_to_add], dtype=tf.float32),
            trainable=False,
            name="unigram",
        )
        self._zero_except_topk = getattr(config, "embedding_zero_except_topk", None)
        self._emb_top_k = None
        self._emb_top_p = None
        self._emb_temp = None
        self._softmax_in_log_space = True

    def call_embedding_model(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        input_strs = self.tokenizer.batch_decode(
            input_ids.numpy().tolist(),
            skip_special_tokens=True,
        )
        emb_inputs = self.embedder_tokenizer(
            input_strs,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        outputs = self.embedder(
            input_ids=emb_inputs["input_ids"],
            attention_mask=emb_inputs["attention_mask"],
        )
        return self._process_embedder_output(
            outputs,
            emb_inputs["attention_mask"],
        )
    
    def embed_and_project(
        self,
        input_ids: Optional[tf.Tensor],
        attention_mask: Optional[tf.Tensor],
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
        elif self.embedder_no_grad:
            embeddings = tf.stop_gradient(self.call_embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ))
        else:
            embeddings = self.call_embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        embeddings = tf.cast(embeddings, self.sequence_weights.dtype)

        if self.trainable and hasattr(self, "unigram"):
            batch_unigram = tf.reduce_mean(embeddings, axis=0, keepdims=True)
            self.unigram.assign(
                (1 - self.unigram_beta) * self.unigram + self.unigram_beta * batch_unigram
            )

        embeddings = embeddings - self.unigram

        if self._zero_except_topk is not None:
            embeddings = zero_embedding_except_topk(
                embeddings,
                vocab_size=self.embedder.config.vocab_size,
                k=self._zero_except_topk,
                default_val=-30.0,
            )

        B = tf.shape(embeddings)[0]
        D = tf.shape(embeddings)[1]
        embeddings = tf.reshape(
            embeddings,
            [B, self.num_repeat_tokens, self.encoder_hidden_dim]
        )

        embeddings = tf.einsum("bsd,sdw->bsw", embeddings, self.sequence_weights)

        embeddings = self.embedding_transform(embeddings)

        attention_mask = tf.ones([tf.shape(embeddings)[0], tf.shape(embeddings)[1]], tf.int32)

        return embeddings, attention_mask

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.Seq2SeqLMOutput,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        logits = outputs.logits
        lengths = tf.reduce_sum(attention_mask, axis=1)
        batch_idx = tf.range(tf.shape(logits)[0])
        time_idx = lengths - 1
        final_logits = tf.gather_nd(
            logits,
            tf.stack([batch_idx, time_idx], axis=1)
        )

        if self._emb_top_k is not None:
            topk = tf.math.top_k(final_logits, k=self._emb_top_k, sorted=False)
            mask = tf.scatter_nd(
                tf.expand_dims(topk.indices, -1),
                tf.ones_like(topk.values),
                tf.shape(final_logits)
            )
            filtered = tf.where(mask>0, final_logits, LOGIT_FILTER_VALUE)
            final_logits = filtered

        if self._emb_top_p is not None:
            probs = tf.nn.softmax(final_logits, axis=1)
            sorted_p, sorted_idx = tf.nn.top_k(probs, k=tf.shape(probs)[1])
            cumsum = tf.cumsum(sorted_p, axis=1)
            cutoff = cumsum >= self._emb_top_p
            scatter_mask = tf.scatter_nd(
                tf.expand_dims(sorted_idx, -1),
                tf.cast(cutoff, final_logits.dtype),
                tf.shape(final_logits)
            )
            final_logits = tf.where(scatter_mask>0, final_logits, LOGIT_FILTER_VALUE)

        if self._emb_temp is not None:
            final_logits = final_logits / self._emb_temp

        if self._softmax_in_log_space:
            embeddings = tf.nn.log_softmax(final_logits, axis=1)
        else:
            embeddings = tf.math.log(tf.nn.softmax(final_logits, axis=1) + 1e-9)

        zeros = tf.zeros([tf.shape(embeddings)[0], self.num_zeros_to_add], dtype=embeddings.dtype)
        return tf.concat([embeddings, zeros], axis=1)

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any],
    ) -> tf.Tensor:
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
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
        training=False,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.TFSeq2SeqLMOutput:
        inputs_embeds, attn = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=kwargs.get("frozen_embeddings"),
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            training=training,
        )
