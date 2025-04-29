from typing import Dict, Optional

import tensorflow as tf
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    load_embedder_and_tokenizer,
    load_tokenizer,
    mean_pool,
)


class InversionModelNonAutoregressive(transformers.TFPreTrainedModel):
    def __init__(
        self,
        config: InversionConfig,
    ):
        super().__init__(config=config)

        encoder = transformers.TFAutoModel.from_pretrained(
            config.model_name_or_path,
        ).encoder
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        self.embedder = embedder
        self.encoder = encoder
        self.embedder_tokenizer = embedder_tokenizer
        self.tokenizer = tokenizer
        self.lm_transform = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_encoder),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5)
        ])
        self.in_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_encoder),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.d_encoder)
        ])

    @property
    def d_encoder(self) -> int:
        return self.encoder.config.d_model

    @property
    def d_embedder(self) -> int:
        return self.embedder.config.d_model

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        input_shape = tf.shape(inputs.get("input_ids", inputs["embedder_input_ids"]))
        
        logits = self.call(**inputs)["logits"]
        
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        summed_log_probs = tf.reduce_sum(log_probs, axis=1)
        top_idxs = tf.math.top_k(summed_log_probs, k=32).indices
        return top_idxs

    def call_embedding_model(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        embeddings = mean_pool(hidden_state, attention_mask)
        return embeddings

    def masked_lm_logits(
        self, inputs_embeds: tf.Tensor, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        projected = self.lm_transform(outputs.last_hidden_state)
        word_embeddings = self.encoder.get_input_embeddings().weights[0]
        logits = tf.matmul(projected, word_embeddings, transpose_b=True)
        return logits

    def masked_lm_loss(
        self,
        logits: tf.Tensor,
        labels: tf.Tensor,
    ) -> tf.Tensor:
        batch_size, seq_length, v = tf.shape(logits)
        logits = tf.reshape(logits, (batch_size * seq_length, v))
        labels = tf.reshape(labels, (batch_size * seq_length,))
        
        mask = tf.not_equal(labels, -100)
        active_logits = tf.boolean_mask(logits, mask)
        active_labels = tf.boolean_mask(labels, mask)
        
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=active_logits, labels=active_labels
            )
        )

    def call(
        self,
        embedder_input_ids: tf.Tensor,
        embedder_attention_mask: tf.Tensor,
        labels: tf.Tensor = None,
        frozen_embeddings: Optional[tf.Tensor] = None,
        training: bool = False,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        batch_size = tf.shape(embedder_input_ids)[0]
        seq_length = tf.shape(embedder_input_ids)[1]
        
        if frozen_embeddings is None:
            embedding = tf.stop_gradient(self.call_embedding_model(
                input_ids=embedder_input_ids, attention_mask=embedder_attention_mask
            ))
        else:
            embedding = frozen_embeddings
            
        tf.debugging.assert_equal(tf.shape(embedding), (batch_size, self.d_embedder))
        embedding = self.in_projection(embedding)
        
        input_ids = tf.ones_like(embedder_input_ids) * self.tokenizer.unk_token_id
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        
        embedding = tf.expand_dims(embedding, axis=1)
        inputs_embeds = tf.concat([embedding, inputs_embeds], axis=1)
        
        attention_mask = tf.ones(tf.shape(inputs_embeds)[:2])
        
        logits = self.masked_lm_logits(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        outputs = {"logits": logits[:, 1:]}
        
        if labels is not None:
            padding = -100 * tf.ones((batch_size, 1), dtype=labels.dtype)
            labels = tf.concat([padding, labels], axis=1)
            loss = self.masked_lm_loss(
                logits=logits,
                labels=labels,
            )
            outputs["loss"] = loss
            
        return outputs