import copy
import logging
from typing import Dict, Optional, Tuple, Any

import transformers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
from sentence_transformers import SentenceTransformer

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
from vec2text.utils import embed_api
from torch import Tensor
import warnings

logger = logging.getLogger(__name__)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionMaskedLogitsModel(keras.Model):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        config.is_decoder = True
        config.add_cross_attention = True
        super().__init__()
        self.config = config
        # self.config.is_decoder = True
        # self.config.add_cross_attention = True
        masked_lm_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        masked_lm_config.is_decoder = True
        masked_lm_config.add_cross_attention = True
        self.masked_lm = transformers.TFRobertaForMaskedLM(masked_lm_config)
        ## overwrite get_extended_attention_mask method
        self.use_logits = True
        bottleneck_dim = 1536
        self.embed_dim = 768
        self.embedding_transform = keras.Sequential([
            layers.Dense(bottleneck_dim,
                        input_shape=(None, self.embed_dim)),
            layers.Dropout(0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.embed_dim),
        ])

    def embed_and_project(
    self,
    frozen_embeddings: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
        embeddings = frozen_embeddings
        # # TODO: what is unigram?
        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        embeddings = tf.concat([
            embeddings,
            tf.zeros(
                (tf.shape(embeddings)[0],
                -tf.shape(embeddings)[1] % self.embed_dim),
                dtype=embeddings.dtype
            )
        ], axis=1)
        num_embeddings = tf.shape(embeddings)[1] // self.embed_dim
        embeddings = tf.reshape(
            embeddings,
            (tf.shape(embeddings)[0], num_embeddings, self.embed_dim)
        )
        embeddings = tf.cast(
            embeddings,
            self.embedding_transform.layers[0].kernel.dtype
        )
        embeddings = self.embedding_transform(embeddings)
        attention_mask = tf.ones(
            (tf.shape(embeddings)[0], tf.shape(embeddings)[1]),
            dtype=tf.int32
        )

        assert embeddings.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.embed_dim,
        )
        return embeddings, attention_mask

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        logits: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.TFMaskedLMOutput:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=logits,
        )
        if self.use_logits is False:
            logit_embeds = None
            logit_attention_mask = None
        return self.masked_lm(
            input_ids=input_ids,
            attention_mask=tf.ones(
                (tf.shape(input_ids)[0], tf.shape(input_ids)[1]),
                dtype=tf.int32
            ),
            labels=labels,
            encoder_hidden_states=logit_embeds,
            encoder_attention_mask=logit_attention_mask,
        )


class InversionMaskedLogitsModelT5(keras.Model):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        super().__init__()
        self.config = config
        t5_config = transformers.T5Config.from_pretrained('t5-base')
        self.masked_lm = transformers.TFT5ForConditionalGeneration(t5_config)
        self.t5_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')

        bottleneck_dim = 1536
        self.embed_dim = 768

        self.embedding_transform = keras.Sequential([
            layers.Dense(bottleneck_dim,
                        input_shape=(None, self.embed_dim)),
            layers.Dropout(0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.embed_dim),
        ])

    def embed_and_project(
        self,
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        embeddings = frozen_embeddings
        # # TODO: what is unigram?
        # if self.training:
        #     # Update unigram.
        #     unigram_batch = embeddings.mean(dim=0, keepdim=True)
        #     self.unigram.data = self.unigram.data * (
        #         1 - self.unigram_beta
        #     ) + unigram_batch * (self.unigram_beta)
        # embeddings -= self.unigram

        embeddings = tf.concat([
        embeddings,
        tf.zeros(
            (tf.shape(embeddings)[0],
             -tf.shape(embeddings)[1] % self.embed_dim),
            dtype=embeddings.dtype
        )
        ], axis=1)
        num_embeddings = tf.shape(embeddings)[1] // self.embed_dim
        embeddings = tf.reshape(
            embeddings,
            (tf.shape(embeddings)[0], num_embeddings, self.embed_dim)
        )
        embeddings = tf.cast(
            embeddings,
            self.embedding_transform.layers[0].kernel.dtype
        )
        embeddings = self.embedding_transform(embeddings)
        attention_mask = tf.ones(
            (tf.shape(embeddings)[0], tf.shape(embeddings)[1]),
            dtype=tf.int32
        )
        return embeddings, attention_mask
    
    def call(
        self,
        labels: Optional[tf.Tensor] = None,
        frozen_embeddings: Optional[tf.Tensor] = None,
        decoder_input_ids: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.T5ForConditionalGenerationOutput:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=frozen_embeddings,
        )
        return self.masked_lm(
            inputs_embeds=logit_embeds,
            attention_mask=logit_attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

    # def generate(
    #     self,
    #     inputs: Dict[str, torch.Tensor],
    #     generation_kwargs: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
    #     inputs_embeds, attention_mask = self.embed_and_project(
    #         input_ids=inputs.get("input_ids"),
    #         attention_mask=inputs.get("attention_mask"),
    #         frozen_embeddings=inputs.get("frozen_embeddings"),
    #     )

    #     return self.encoder_decoder.generate(
    #         # required: input embeddings
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         # optional: input IDs (for starting generation).
    #         # typically not set unless generating prefixes for
    #         # reranking.
    #         decoder_input_ids=inputs["decoder_input_ids"],
    #         # decoder_attention_mask=inputs["decoder_attention_mask"],
    #         **generation_kwargs,
    #     )
    

class InversionMaskedLogitsModelEncoder(keras.Model):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    config_class = InversionConfig
    def __init__(self, config: InversionConfig):
        super().__init__()
        self.config = config
        masked_lm_config = transformers.RobertaConfig.from_pretrained('roberta-base')
        # ← switch to TF Roberta
        self.masked_lm = transformers.TFRobertaForMaskedLM(masked_lm_config)

        bottleneck_dim = 1536
        self.embed_dim = 768

        self.embedding_transform = keras.Sequential([
            layers.Dense(bottleneck_dim,
                        input_shape=(None, self.embed_dim)),
            layers.Dropout(0.0),
            layers.Activation(activations.gelu),
            layers.Dense(self.embed_dim),
        ])

    def embed_and_project(
        self,
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        embeddings = frozen_embeddings
        embeddings = tf.concat([
            embeddings,
            tf.zeros(
                (tf.shape(embeddings)[0],
                -tf.shape(embeddings)[1] % self.embed_dim),
                dtype=embeddings.dtype
            )
        ], axis=1)
        num_embeddings = tf.shape(embeddings)[1] // self.embed_dim
        embeddings = tf.reshape(
            embeddings,
            (tf.shape(embeddings)[0], num_embeddings, self.embed_dim)
        )
        embeddings = tf.cast(
            embeddings,
            self.embedding_transform.layers[0].kernel.dtype
        )
        embeddings = self.embedding_transform(embeddings)
        attention_mask = tf.ones(
            (tf.shape(embeddings)[0], tf.shape(embeddings)[1]),
            dtype=tf.int32
        )
        return embeddings, attention_mask

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        logits: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_tf_outputs.TFMaskedLMOutput:
        logit_embeds, logit_attention_mask = self.embed_and_project(
            frozen_embeddings=logits,
        )
        return self.masked_lm(
            input_ids=input_ids,
            attention_mask=tf.ones(
                (tf.shape(input_ids)[0], tf.shape(input_ids)[1]),
                dtype=tf.int32
            ),
            labels=labels,
        )