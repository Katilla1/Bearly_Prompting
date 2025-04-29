import copy
from typing import Dict, Optional, Tuple

import tensorflow as tf
import transformers

from vec2text.models.config import InversionConfig


class CorrectorEncoderModel(transformers.TFPreTrainedModel):
    config_class = InversionConfig

    def __init__(
        self,
        config: InversionConfig,
    ):
        super().__init__(config=config)
        if config.embedder_model_api:
            embedder_dim = 1536
        else:
            embedder_dim = 768
        bottleneck_dim = embedder_dim

        num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = config.corrector_ignore_hypothesis_embedding
        self.use_ff_dropout = False

        encoder_decoder = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.encoder_decoder = encoder_decoder
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        
        self.embedding_transform_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(bottleneck_dim),
            tf.keras.layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.encoder_hidden_dim * num_repeat_tokens)
        ])
        
        self.embedding_transform_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(bottleneck_dim),
            tf.keras.layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.encoder_hidden_dim * num_repeat_tokens)
        ])
        
        self.embedding_transform_3 = tf.keras.Sequential([
            tf.keras.layers.Dense(bottleneck_dim),
            tf.keras.layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.encoder_hidden_dim * num_repeat_tokens)
        ])
        
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        self.training_embedding_noise_level = 0
        self.use_ln = True
        if self.use_ln:
            self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def get_encoder_embedding(
        self,
        embedding: tf.Tensor,
        hypothesis_embedding: tf.Tensor,
        hypothesis_input_ids: tf.Tensor,
        hypothesis_attention_mask: tf.Tensor,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(embedding)[0]
        
        tf.debugging.assert_equal(tf.shape(embedding), tf.shape(hypothesis_embedding))
        
        if training and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(embedding)
            )
            hypothesis_embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(hypothesis_embedding)
            )

        if self.ignore_hypothesis_embedding:
            hypothesis_embedding = embedding

        diff_embedding = embedding - hypothesis_embedding

        embedding = self.embedding_transform_1(embedding)
        embedding = tf.reshape(
            embedding, (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        
        diff_embedding = self.embedding_transform_2(diff_embedding)
        diff_embedding = tf.reshape(
            diff_embedding, (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        
        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        hypothesis_embedding = tf.reshape(
            hypothesis_embedding, (batch_size, self.num_repeat_tokens, self.encoder_hidden_dim)
        )
        
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        
        ones = tf.ones((batch_size, 1), dtype=tf.int32)
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        
        inputs_embeds = tf.concat(
            [
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
                sep_token,
                inputs_embeds,
            ],
            axis=1,
        )
        
        if self.use_ln:
            inputs_embeds = self.layernorm(inputs_embeds)
            
        attention_mask = tf.concat(
            [tf.repeat(ones, repeats=[4 + 3 * self.num_repeat_tokens], axis=1), hypothesis_attention_mask],
            axis=1,
        )
        
        return (inputs_embeds, attention_mask)

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, tf.Tensor],
        return_dict_in_generate: bool = False,
    ) -> tf.Tensor:
        if "max_length" not in generation_kwargs:
            generation_kwargs = copy.copy(generation_kwargs)
            generation_kwargs["max_length"] = tf.shape(
                inputs.get("input_ids", inputs["embedder_input_ids"])
            )[1]

        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=inputs["frozen_embeddings"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
            hypothesis_embedding=inputs["hypothesis_embedding"],
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                decoder_input_ids=inputs["decoder_input_ids"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                **generation_kwargs,
            )

    def call(
        self,
        embedding: tf.Tensor,
        hypothesis_embedding: tf.Tensor,
        hypothesis_input_ids: tf.Tensor,
        hypothesis_attention_mask: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
        training: bool = False,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            training=training
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            training=training,
        )