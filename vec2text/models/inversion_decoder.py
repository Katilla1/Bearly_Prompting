import copy
import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf
import transformers
from sentence_transformers import SentenceTransformer

from vec2text.models import InversionModel
from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import load_embedder_and_tokenizer, load_tokenizer

logger = logging.getLogger(__name__)

class InversionModelDecoderOnly(InversionModel):
    def __init__(
        self,
        config: InversionConfig,
    ):
        super(InversionModel, self).__init__(config=config)

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        embedder_model_api = config.embedder_model_api

        if "t5" in config.model_name_or_path:
            decoder = transformers.TFT5ForConditionalGeneration.from_pretrained(
                config.model_name_or_path
            )
        else:
            decoder = transformers.TFAutoModelForCausalLM.from_pretrained(
                config.model_name_or_path
            )
        self.embedder = embedder
        self.decoder = decoder

        embedder_no_grad = config.embedder_no_grad
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input

        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            self.embedder_dim = 1536
            bottleneck_dim = 1536
        elif isinstance(self.embedder, SentenceTransformer):
            self.embedder_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder_dim = self.embedder.config.hidden_size

        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim
        self.embedding_transform = tf.keras.layers.Dense(
            self.decoder.config.hidden_size
        )
        
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.noise_level = 0
        self.embeddings_from_layer_n = None

    def embed_and_project(
        self,
        embedder_input_ids: Optional[tf.Tensor],
        embedder_attention_mask: Optional[tf.Tensor],
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(tf.shape(embeddings)) == 2
        elif self.embedder_no_grad:
            embeddings = tf.stop_gradient(self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            ))
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )

        if self.embedding_transform_strategy == "none":
            pass
        elif self.embedding_transform_strategy == "repeat":
            embeddings = self.embedding_transform(embeddings)
            batch_size = tf.shape(embeddings)[0]
            embeddings = tf.reshape(embeddings, (batch_size, 1, -1))
        elif self.embedding_transform_strategy == "nearest_neighbors":
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = tf.ones(
            (tf.shape(embeddings)[0], tf.shape(embeddings)[1])
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs["embedder_input_ids"],
            embedder_attention_mask=inputs["embedder_attention_mask"],
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        if "decoder_input_ids" in inputs:
            return self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                input_ids=inputs["decoder_input_ids"],
                **generation_kwargs,
            )
        else:
            return self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

    def call(
        self,
        embedder_input_ids: tf.Tensor,
        embedder_attention_mask: tf.Tensor,
        input_ids: tf.Tensor = None,
        attention_mask: tf.Tensor = None,
        labels: Optional[tf.Tensor] = None,
        frozen_embeddings: Optional[tf.Tensor] = None,
        training: bool = False,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        if labels is not None:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

        embed_inputs_embeds, embed_attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        input_embeddings_table = self.decoder.get_input_embeddings()
        inputs_embeds = tf.concat(
            [embed_inputs_embeds, input_embeddings_table(input_ids)], axis=1
        )
        attention_mask = tf.concat([embed_attention_mask, attention_mask], axis=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            training=training,
        )