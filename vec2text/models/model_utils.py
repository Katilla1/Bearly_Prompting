import os
from typing import Any, Dict

import tensorflow as tf
import transformers
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


def get_device():
    if tf.config.list_physical_devices('GPU'):
        dev = "GPU"
    else:
        dev = "CPU"
    return dev


device = get_device()


def disable_dropout(model):
    dropout_count = 0
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
            dropout_count += 1
        
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Dropout):
                    sublayer.rate = 0.0
                    dropout_count += 1
    
    print(f"Disabled {dropout_count} dropout modules from model type {type(model)}")


def freeze_params(model):
    total_num_params = 0
    
    for layer in model.layers:
        layer.trainable = False
        total_num_params += sum([tf.keras.backend.count_params(w) for w in layer.weights])


def mean_pool(hidden_states, attention_mask):
    B = tf.shape(hidden_states)[0]
    D = tf.shape(hidden_states)[2]
    
    attention_mask = tf.cast(attention_mask, dtype=hidden_states.dtype)
    attention_mask = tf.expand_dims(attention_mask, axis=-1)
    
    unmasked_outputs = hidden_states * attention_mask
    sum_embeddings = tf.reduce_sum(unmasked_outputs, axis=1)
    sum_mask = tf.reduce_sum(attention_mask, axis=1) + 1e-9
    
    pooled_outputs = sum_embeddings / sum_mask
    
    assert_op = tf.debugging.assert_equal(
        tf.shape(pooled_outputs)[:2],
        [B, D]
    )
    
    with tf.control_dependencies([assert_op]):
        return pooled_outputs


def max_pool(hidden_states, attention_mask):
    B = tf.shape(hidden_states)[0]
    D = tf.shape(hidden_states)[2]
    
    attention_mask = tf.cast(attention_mask, dtype=hidden_states.dtype)
    mask_expanded = tf.expand_dims(attention_mask, axis=-1)
    
    masked_hidden_states = hidden_states * mask_expanded + (1 - mask_expanded) * -1e9
    
    pooled_outputs = tf.reduce_max(masked_hidden_states, axis=1)
    
    assert_op = tf.debugging.assert_equal(
        tf.shape(pooled_outputs)[:2],
        [B, D]
    )
    
    with tf.control_dependencies([assert_op]):
        return pooled_outputs


def stack_pool(hidden_states, attention_mask):
    B = tf.shape(hidden_states)[0]
    S = tf.shape(hidden_states)[1]
    D = tf.shape(hidden_states)[2]
    
    attention_mask = tf.cast(attention_mask, dtype=hidden_states.dtype)
    mask_expanded = tf.expand_dims(attention_mask, axis=-1)
    unmasked_outputs = hidden_states * mask_expanded
    
    pooled_outputs = tf.reshape(unmasked_outputs, [B, S * D])
    
    assert_op = tf.debugging.assert_equal(
        tf.shape(pooled_outputs)[:2],
        [B, S * D]
    )
    
    with tf.control_dependencies([assert_op]):
        return pooled_outputs


def load_embedder_and_tokenizer(name, tf_dtype=None, **kwargs):
    model_kwargs = {
        "output_hidden_states": False,
    }
    
    if name == "dpr":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        try:
            model = transformers.TFDPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base", from_pt=True
            )
        except:
            model = transformers.TFDPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base", from_pt=True
            )
    elif name == "dpr_st":
        st_model = SentenceTransformer(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        )
        tokenizer = st_model.tokenizer
        model = st_model
    elif name == "contriever":
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
        try:
            model = transformers.TFAutoModel.from_pretrained(
                "facebook/contriever", **model_kwargs
            )
        except:
            model = transformers.TFAutoModel.from_pretrained(
                "facebook/contriever", from_pt=True, **model_kwargs
            )
    elif name == "bert":
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        model = transformers.TFAutoModel.from_pretrained(
            "bert-base-uncased", **model_kwargs
        )
    elif name == "gtr_base":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        try:
            model = transformers.TFAutoModel.from_pretrained(
                "sentence-transformers/gtr-t5-base", **model_kwargs
            ).encoder
        except:
            model = transformers.TFAutoModel.from_pretrained(
                "sentence-transformers/gtr-t5-base", from_pt=True, **model_kwargs
            ).encoder
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        model = transformers.TFAutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", from_pt=True, **model_kwargs
        ).encoder
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(
                    tf.keras.initializers.GlorotUniform()(shape=layer.kernel.shape)
                )
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.assign(tf.zeros_like(layer.bias))
                
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_base_st" or name == "gtr_large":
        st_name = "sentence-transformers/gtr-t5-base" if name == "gtr_base_st" else "sentence-transformers/gtr-t5-large"
        st_model = SentenceTransformer(st_name)
        tokenizer = st_model.tokenizer
        model = st_model
    elif name == "ance_tele":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder"
        )
        try:
            model = transformers.TFAutoModel.from_pretrained(
                "OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs
            )
        except:
            model = transformers.TFAutoModel.from_pretrained(
                "OpenMatch/ance-tele_nq_psg-encoder", from_pt=True, **model_kwargs
            )
    elif name == "paraphrase-distilroberta":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
        try:
            model = transformers.TFAutoModel.from_pretrained(
                "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
            )
        except:
            model = transformers.TFAutoModel.from_pretrained(
                "sentence-transformers/paraphrase-distilroberta-base-v1", 
                from_pt=True, 
                **model_kwargs
            )
    elif name == "medicalai/ClinicalBERT":
        tokenizer = transformers.AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        try:
            model = transformers.TFAutoModel.from_pretrained(
                "medicalai/ClinicalBERT", **model_kwargs
            )
        except:
            model = transformers.TFAutoModel.from_pretrained(
                "medicalai/ClinicalBERT", from_pt=True, **model_kwargs
            )
    elif name.startswith("gpt2"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        try:
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name, **model_kwargs
            )
        except:
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name, from_pt=True, **model_kwargs
            )
    elif name.startswith("meta-llama/Llama-2-70b"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        
        try:
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name, from_pt=True, **model_kwargs
            )
        except:
            print("Warning: 70B model may not work well in TensorFlow")
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name, from_pt=True, **model_kwargs
            )
            
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    elif name.startswith("meta-llama/"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        
        if tf_dtype == "float32":
            tf_dtype = tf.float32
        elif tf_dtype == "float16":
            tf_dtype = tf.float16
        elif tf_dtype == "bfloat16":
            tf_dtype = tf.bfloat16
            
        try:
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name,
                **model_kwargs,
                token=os.environ.get("LLAMA_TOKEN"),
                from_pt=True,
                **kwargs,
            )
        except:
            print(f"Error loading LLaMA model {name}")
            model = transformers.TFAutoModelForCausalLM.from_pretrained(
                name, from_pt=True, **model_kwargs
            )
            
        if tf_dtype is not None:
            tf.keras.mixed_precision.set_global_policy(tf_dtype.name)
    elif name.startswith("sentence-transformers/"):
        st_model = SentenceTransformer(name)
        tokenizer = st_model.tokenizer
        model = st_model
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        try:
            model = transformers.TFAutoModel.from_pretrained(
                name, **model_kwargs
            )
        except:
            print(f"No native TF model found for {name}, trying PyTorch conversion")
            model = transformers.TFAutoModel.from_pretrained(
                name, from_pt=True, **model_kwargs
            )
            
    if not isinstance(model, SentenceTransformer):
        try:
            model.compile(optimizer='adam')
        except:
            pass
        
    return model, tokenizer


def load_encoder_decoder(model_name, lora=False):
    model_kwargs = {}
    
    if lora:
        print("Warning: LoRA not directly supported in TensorFlow")
        
    try:
        model = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(
            model_name, **model_kwargs
        )
    except:
        print(f"No native TF model found for {model_name}, trying PyTorch conversion")
        model = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(
            model_name, from_pt=True, **model_kwargs
        )
        
    return model


def load_tokenizer(name, max_length):
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