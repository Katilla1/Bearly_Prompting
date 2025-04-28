import copy
from typing import Dict, Optional, Tuple

import transformers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from vec2text.models.config import InversionConfig


class CorrectorEncoderModel(keras.Model):
    """Embeds text and concats with a provided embedding.

    TODO improve comment here.
    """

    config_class = InversionConfig
    encoder_decoder: transformers.PreTrainedModel

    def __init__(self, config: InversionConfig):
        super().__init__()
        self.config = config
        if config.embedder_model_api:
            embedder_dim = 1536
        else:
            embedder_dim = 768
        bottleneck_dim = embedder_dim
        num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = config.corrector_ignore_hypothesis_embedding
        self.use_ff_dropout = False
        # load TF encoder-decoder model
        self.encoder_decoder = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        # three separate embedding transforms
        self.embedding_transform_1 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, embedder_dim)),
            layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim * num_repeat_tokens),
        ])
        self.embedding_transform_2 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, embedder_dim)),
            layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim * num_repeat_tokens),
        ])
        self.embedding_transform_3 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, embedder_dim)),
            layers.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim * num_repeat_tokens),
        ])
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        # TODO argparse; default to 0?
        self.training_embedding_noise_level = 0
        # self.training_embedding_noise_level = 1e-5  # adding for openai...
        self.use_ln = True
        if self.use_ln:
            # layernorm
            self.layernorm = layers.LayerNormalization(epsilon=1e-5)
        # print(f"Corrector encoder noise level {self.training_embedding_noise_level}")

    def get_encoder_embedding(
        self,
        embedding: tf.Tensor,
        hypothesis_embedding: tf.Tensor,
        hypothesis_input_ids: tf.Tensor,
        hypothesis_attention_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(embedding)[0]
        assert embedding.shape[-1] == self.embedder_dim
        assert hypothesis_embedding.shape[-1] == self.embedder_dim

        if self.training and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(embedding), dtype=embedding.dtype
            )
            hypothesis_embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(hypothesis_embedding), dtype=hypothesis_embedding.dtype
            )
        if self.ignore_hypothesis_embedding:
            # For "No Feedback" ablation
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
        #
        ones = tf.ones((batch_size, 1), dtype=tf.int32)
        # TODO: pad_token_id or eos_token_id? Or does it not matter?
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        # inputs_embeds = tf.concat((sep_token, embedding, sep_token, hypothesis_embedding, inputs_embeds), axis=1)
        inputs_embeds = tf.concat(
            (
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
                sep_token,
                inputs_embeds,
            ),
            axis=1,
        )
        if self.use_ln:
            inputs_embeds = self.layernorm(inputs_embeds)
        attention_mask = tf.concat(
            (tf.ones((batch_size, 4 + 3 * self.num_repeat_tokens), dtype=tf.int32), hypothesis_attention_mask),
            axis=1,
        )
        return inputs_embeds, attention_mask

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
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=return_dict_in_generate,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def call(
        self,
        embedding: tf.Tensor,
        hypothesis_embedding: tf.Tensor,
        hypothesis_input_ids: tf.Tensor,
        hypothesis_attention_mask: tf.Tensor,
        labels: Optional[tf.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
