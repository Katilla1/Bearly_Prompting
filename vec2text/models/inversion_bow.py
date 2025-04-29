from typing import Dict, Optional

import tensorflow as tf
import transformers

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    load_embedder_and_tokenizer,
    load_tokenizer,
    mean_pool,
)


class InversionModelBagOfWords(transformers.TFPreTrainedModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        encoder = transformers.TFAutoModel.from_pretrained(
            config.model_name_or_path,
        ).encoder
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
        logits = self.call(**inputs)["logits"]
        max_length = tf.shape(inputs.get("input_ids", inputs["embedder_input_ids"]))[1]
        return tf.math.top_k(logits, k=max_length).indices

    def call_embedding_model(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        return mean_pool(hidden_state, attention_mask)

    def bow_logits(
        self, inputs_embeds: tf.Tensor, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        output_vector = mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self.lm_transform(output_vector)
        word_embeddings = self.encoder.get_input_embeddings().weights[0]
        logits = tf.matmul(projected, word_embeddings, transpose_b=True)
        return logits

    def bow_loss(
        self,
        logits: tf.Tensor,
        labels: tf.Tensor,
    ) -> tf.Tensor:
        vocab_size = tf.shape(self.encoder.get_input_embeddings().weights[0])[0]
        vocab = tf.range(vocab_size, dtype=labels.dtype)
        
        expanded_labels = tf.expand_dims(labels, axis=-1)
        expanded_vocab = tf.expand_dims(tf.expand_dims(vocab, 0), 0)
        
        mask = tf.equal(expanded_labels, expanded_vocab)
        one_hot_labels = tf.cast(tf.reduce_any(mask, axis=1), tf.float32)
        
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                one_hot_labels, logits, from_logits=True
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
            embedding = tf.stop_gradient(
                self.call_embedding_model(
                    input_ids=embedder_input_ids, 
                    attention_mask=embedder_attention_mask
                )
            )
        else:
            embedding = frozen_embeddings
            
        tf.debugging.assert_equal(tf.shape(embedding), (batch_size, self.d_embedder))
        embedding = self.in_projection(embedding)
        
        input_ids = tf.ones_like(embedder_input_ids) * self.tokenizer.unk_token_id
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        
        embedding = tf.expand_dims(embedding, axis=1)
        inputs_embeds = tf.concat([embedding, inputs_embeds], axis=1)
        
        attention_mask = tf.ones(tf.shape(inputs_embeds)[:2])
        
        logits = self.bow_logits(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        outputs = {"logits": logits}
        
        if labels is not None:
            padding = -100 * tf.ones((batch_size, 1), dtype=labels.dtype)
            labels = tf.concat([padding, labels], axis=1)
            loss = self.bow_loss(
                logits=logits,
                labels=labels,
            )
            outputs["loss"] = loss
            
        return outputs