from typing import Tuple, Dict, Optional
import copy
import torch
import transformers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from vec2text.models.config import InversionConfig

from .corrector_encoder import CorrectorEncoderModel


class CorrectorEncoderFromLogitsModel(CorrectorEncoderModel):
    config_class = InversionConfig
    encoder_decoder: transformers.PreTrainedModel

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        config.embedder_dim = 768  # TODO: Pipe this in.
        config.num_zeros_to_add = self.num_zeros_to_add = 512  # TODO: Compute this.
        config.num_repeat_tokens = self.num_repeat_tokens = 42  # TODO: Compute this properly.
        # TODO: Calculate this explicitly from trainer.
        unigram_tensor = torch.load(
            "/home/jxm3/research/retrieval/inversion/llama_unigram.pt",
            map_location="cpu"
        )
        self.unigram = tf.convert_to_tensor(
            unigram_tensor.numpy(), dtype=tf.float32
        )
        self.embedder_dim = config.embedder_dim
        bottleneck_dim = config.embedder_dim
        initializer = tf.keras.initializers.GlorotUniform()
        self.sequence_weights_1 = tf.Variable(
            initializer(shape=(self.num_repeat_tokens, self.embedder_dim, self.embedder_dim)),
            trainable=True,
            name="sequence_weights_1",
        )
        self.sequence_layernorm_1 = layers.LayerNormalization(epsilon=1e-5)
        self.sequence_weights_2 = tf.Variable(
            initializer(shape=(self.num_repeat_tokens, self.embedder_dim, self.embedder_dim)),
            trainable=True,
            name="sequence_weights_2",
        )
        self.sequence_layernorm_2 = layers.LayerNormalization(epsilon=1e-5)
        self.sequence_weights_3 = tf.Variable(
            initializer(shape=(self.num_repeat_tokens, self.embedder_dim, self.embedder_dim)),
            trainable=True,
            name="sequence_weights_3",
        )
        self.sequence_layernorm_3 = layers.LayerNormalization(epsilon=1e-5)
        self.encoder_decoder = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, self.embedder_dim)),
            layers.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim),
        ])
        self.embedding_transform_2 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, self.embedder_dim)),
            layers.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim),
        ])
        self.embedding_transform_3 = keras.Sequential([
            layers.Dense(bottleneck_dim, input_shape=(None, self.embedder_dim)),
            layers.Dropout(self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.encoder_hidden_dim),
        ])

    def get_encoder_embedding(
        self,
        embedding: tf.Tensor,
        hypothesis_embedding: tf.Tensor,
        hypothesis_input_ids: tf.Tensor,
        hypothesis_attention_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(embedding)[0]
        D = tf.shape(embedding)[1]
        if self.training and (self.training_embedding_noise_level > 0):
            embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(embedding), dtype=embedding.dtype
            )
            hypothesis_embedding += self.training_embedding_noise_level * tf.random.normal(
                tf.shape(hypothesis_embedding), dtype=hypothesis_embedding.dtype
            )
        unigram = self.unigram
        embedding = embedding - unigram
        hypothesis_embedding = hypothesis_embedding - unigram
        embedding = embedding[:, :32256]  # (b, 32768) -> (b, 32256)
        hypothesis_embedding = hypothesis_embedding[:, :32256]  # (b, 32768) -> (b, 32256)
        diff_embedding = embedding - hypothesis_embedding
        embedding = tf.cast(embedding, tf.float32)
        embedding = tf.reshape(
            embedding, (batch_size, self.num_repeat_tokens, self.embedder_dim)
        )
        embedding = tf.einsum("bsd,sdw->bsw", embedding, self.sequence_weights_1)
        embedding = tf.cast(embedding, self.sequence_layernorm_1.dtype)
        embedding = self.sequence_layernorm_1(embedding)
        embedding = self.embedding_transform_1(embedding)
        #
        diff_embedding = tf.cast(diff_embedding, tf.float32)
        diff_embedding = tf.reshape(
            diff_embedding, (batch_size, self.num_repeat_tokens, self.embedder_dim)
        )
        diff_embedding = tf.einsum("bsd,sdw->bsw", diff_embedding, self.sequence_weights_2)
        diff_embedding = tf.cast(diff_embedding, self.sequence_layernorm_2.dtype)
        diff_embedding = self.sequence_layernorm_2(diff_embedding)
        diff_embedding = self.embedding_transform_2(diff_embedding)
        #
        hypothesis_embedding = tf.cast(hypothesis_embedding, tf.float32)
        hypothesis_embedding = tf.reshape(
            hypothesis_embedding, (batch_size, self.num_repeat_tokens, self.embedder_dim)
        )
        hypothesis_embedding = tf.einsum(
            "bsd,sdw->bsw",
            hypothesis_embedding,
            self.sequence_weights_3,
        )
        hypothesis_embedding = tf.cast(hypothesis_embedding, self.sequence_layernorm_3.dtype)
        hypothesis_embedding = self.sequence_layernorm_3(hypothesis_embedding)
        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        #
        ones = tf.ones((batch_size, 1), dtype=tf.int32)
        # TODO: pad_token_id or eos_token_id? Or does it not matter?
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)

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
        inputs_embeds = self.layernorm(inputs_embeds)
        attention_mask = tf.concat(
            (tf.repeat(ones, repeats=4 + 3 * self.num_repeat_tokens, axis=1), hypothesis_attention_mask),
            axis=1,
        )

        # if self.training:
        #     import wandb

        #     try:
        #         wandb.log(
        #             {
        #                 "emb_norm/emb": float(tf.reduce_mean(tf.abs(embedding))),
        #                 "emb_norm/hypothesis": float(tf.reduce_mean(tf.abs(hypothesis_embedding))),
        #                 "emb_norm/diff": float(tf.reduce_mean(tf.abs(diff_embedding))),
        #                 "emb_norm/input_length": int(tf.shape(attention_mask)[1]),
        #             }
        #         )
        #     except Exception:
        #         pass
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
            hypothesis_embedding=inputs["hypothesis_embedding"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
        )
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
