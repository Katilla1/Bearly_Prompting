import copy
import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import torch
import transformers
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

logger = logging.getLogger(__name__)


# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionModel(keras.Model):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedder_model_api: Optional[str]

    def __init__(self, config: InversionConfig):
        super().__init__()

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, dtype=config.embedder_dtype
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens

        self.embedder_is_decoder = False

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            # Hard-code OpenAI embedding dim
            self.embedder_dim = 1536
            bottleneck_dim = self.embedder_dim
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim
        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

        self.embedding_transform = tf.keras.Sequential([
            layers.Dense(self.embedder_dim),
            layers.Dropout(self.encoder_decoder.config.dropout_rate),
            layers.Activation(activations.gelu),
            layers.Dense(encoder_hidden_dim * self.num_repeat_tokens),
        ])
        ######################################################
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        # self.freeze(freeze_strategy=config.freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros

        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = 0

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            # in this case, freeze embeddings too
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(
                    outputs, "hidden_states"
                ), "output missing hidden states - did you remember to initialize the model with output_hidden_states=True?"
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                hidden_state = outputs.last_hidden_state
                # Note: why do we need mean pool here?
                embeddings = mean_pool(hidden_state, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        # Llama
        if isinstance(self.embedder, torch.nn.Module):
            np_ids   = input_ids.numpy()
            np_mask  = attention_mask.numpy()
            pt_ids   = torch.from_numpy(np_ids).to(self.embedder.device)
            pt_mask  = torch.from_numpy(np_mask).to(self.embedder.device)
            with torch.no_grad():
                pt_out = self.embedder(input_ids=pt_ids, attention_mask=pt_mask)
            if hasattr(pt_out, "pooler_output") and pt_out.pooler_output is not None:
                pt_emb = pt_out.pooler_output
            else:
                pt_emb = pt_out.last_hidden_state.mean(dim=1)
            emb_np = pt_emb.cpu().numpy()
            embeddings = tf.convert_to_tensor(emb_np, dtype=tf.float32)
        else:
            # BERT, DPR, GPT-2 TF, etc
            if self.embedder_no_grad:
                pass
            if self.embedder_fake_with_zeros:
                batch_size = tf.shape(input_ids)[0]
                embeddings = tf.zeros((batch_size, self.embedder_dim), dtype=tf.float32)
            elif isinstance(self.embedder, SentenceTransformer):
                model_output = self.embedder(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
                embeddings = model_output["sentence_embedding"]
            else:
                model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self._process_embedder_output(model_output, attention_mask)
        if self.noise_level > 0:
            noise = self.noise_level * tf.random.normal(tf.shape(embeddings))
            embeddings = embeddings + noise
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[tf.Tensor],
        embedder_attention_mask: Optional[tf.Tensor],
        frozen_embeddings: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
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
        if self.embedding_transform_strategy == "repeat":
            repeated_embeddings = self.embedding_transform(embeddings)
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = tf.reshape(
                repeated_embeddings,
                [-1, self.num_repeat_tokens,
                 repeated_embeddings.shape[-1] // self.num_repeat_tokens],
            )
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = tf.ones(
            shape=(tf.shape(embeddings)[0], tf.shape(embeddings)[1]),
            dtype=tf.int32,
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def call(
        self,
        embedder_input_ids,
        embedder_attention_mask,
        labels=None,
        training=False,
        **kwargs,
    ):
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=kwargs.get("frozen_embeddings"),
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            training=training,
        )