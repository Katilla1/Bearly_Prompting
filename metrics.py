from typing import Dict, List

import scipy.stats
import torch
import tensorflow as tf

from vec2text.utils import get_embeddings_openai_vanilla


class EmbeddingCosineSimilarity:
    """Computes the cosine similarity between two lists of
    string pairs using OpenAI ada-2 embeddings.
    """

    def __call__(self, s1: List[str], s2: List[str]) -> Dict[str, float]:
        try:
            e1 = tf.tensor(get_embeddings_openai_vanilla(s1), dtype=tf.float32)
            e2 = tf.tensor(get_embeddings_openai_vanilla(s2), dtype=tf.float32)
            torch_sims = torch.nn.functional.cosine_similarity(e1, e2, dim=1)
            print("torch sims: ", torch_sims)
            sims = tf.losses.cosine_distance(tf.nn.l2_normalize(e1, 0), tf.nn.l2_normalize(e2, 0), dim=1)
            print("tf_sims: ", sims)
            return {
                "ada_emb_cos_sim_mean": sims.mean().item(),
                "ada_emb_cos_sim_sem": scipy.stats.sem(sims.numpy()),
            }
        except Exception:
            print(f"Error getting {len(s1)} embeddings from OpenAI. Returning zeros.")
            return {
                "ada_emb_cos_sim_mean": 0.0,
                "ada_emb_cos_sim_sem": 0.0,
            }
