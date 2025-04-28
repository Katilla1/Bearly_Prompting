import os
from typing import Any, Dict

import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import LlamaForCausalLM
import torch
from sentence_transformers import SentenceTransformer

EMBEDDER_MODEL_NAMES = [
    "bert",
    "contriever",
    "dpr",
    "gtr_base",
    "gtr_base__random_init",
    "medicalai/ClinicalBERT",
    "gtr_large",
    "ance_tele",
    "dpr_st",
    "gtr_base_st",
    "paraphrase-distilroberta",
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def disable_dropout(model: keras.Model):
    count = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dropout):
            layer.rate = 0.0
            count += 1
    print(f"Disabled {count} Dropout layers in model.")


def freeze_params(model: keras.Model):
    for layer in model.layers:
        layer.trainable = False


def mean_pool(hidden_states: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    masked = hidden_states * tf.expand_dims(attention_mask, -1)
    summed = tf.reduce_sum(masked, axis=1)
    lengths = tf.reduce_sum(attention_mask, axis=1, keepdims=True)
    return summed / lengths


def max_pool(hidden_states: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    masked = hidden_states * tf.expand_dims(attention_mask, -1)
    return tf.reduce_max(masked, axis=1)


def stack_pool(hidden_states: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    masked = hidden_states * tf.expand_dims(attention_mask, -1)
    shape = tf.shape(masked)
    B, S, D = shape[0], shape[1], shape[2]
    return tf.reshape(masked, (B, S * D))


def load_embedder_and_tokenizer(name: str, dtype: str, **kwargs):
    model_kwargs = { "output_hidden_states": False }
    if name == "dpr":
        model = transformers.TFDPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
    elif name == "dpr_st":
        model = SentenceTransformer(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        )
        tokenizer = model.tokenizer
    elif name == "contriever":
        model = transformers.TFAutoModel.from_pretrained(
            "facebook/contriever", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "bert":
        model = transformers.TFAutoModel.from_pretrained(
            "bert-base-uncased", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gtr_base":
        model = transformers.TFAutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        model = transformers.TFAutoModel.from_config(config).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name in ("gtr_base_st", "gtr_large"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.TFAutoModel.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder"
        )
    elif name == "paraphrase-distilroberta":
        model = transformers.TFAutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
    elif name == "medicalai/ClinicalBERT":
        model = transformers.TFAutoModel.from_pretrained(
            "medicalai/ClinicalBERT", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif name.startswith("gpt2"):
        from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        model = TFGPT2LMHeadModel.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("meta-llama"):
        # No HuggingFace Llama TF model
        model = LlamaForCausalLM.from_pretrained(
            name,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        model.eval()
    elif name.startswith("sentence-transformers/"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
    else:
        print(f"WARNING: unknown embedder {name}, loading TF AutoModel")
        model = transformers.TFAutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def load_encoder_decoder(model_name: str, lora: bool = False):
    from transformers import TFT5ForConditionalGeneration
    return TFT5ForConditionalGeneration.from_pretrained(model_name)


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        padding="max_length",
        truncation="max_length",
        max_length=max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
