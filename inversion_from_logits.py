from typing import Dict, Optional, Any
import tensorflow as tf

from .inversion_tf import InversionTrainer


def kl_divergence(p: tf.Tensor, q: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Compute KL divergence KL(p || q) = sum p * (log p - log q) over last axis.
    """
    p_clamped = p + eps
    q_clamped = q + eps
    return tf.reduce_sum(p_clamped * (tf.math.log(p_clamped) - tf.math.log(q_clamped)), axis=1)


class InversionFromLogitsTrainer(InversionTrainer):
    """Custom trainer for inverting from logits with optional length-check decoding."""

    generation_method: Optional[str] = None

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        if self.generation_method == "length_check":
            return self.generate_and_check_length(inputs, generation_kwargs)
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def generate_and_check_length(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        # 1) compute frozen embeddings without gradient
        frozen_embeddings = tf.stop_gradient(self.model.call_embedding_model(
            input_ids=inputs["embedder_input_ids"],
            attention_mask=inputs["embedder_attention_mask"],
        ))

        batch_size = tf.shape(inputs["embedder_input_ids"])[0]
        max_len = 64

        closest_generations = None
        closest_distances = None

        # 2) iterate over lengths
        for length in range(1, max_len):
            generation_kwargs["min_length"] = length
            generation_kwargs["max_length"] = length

            # generate sequences of this exact length
            generations = self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

            # decode to text, then re-tokenize for embedder
            gen_strs = self.tokenizer.batch_decode(generations.numpy().tolist(), skip_special_tokens=True)
            emb_inputs = self.embedder_tokenizer(
                gen_strs,
                return_tensors="tf",
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

            new_embeddings = tf.stop_gradient(self.model.call_embedding_model(
                input_ids=emb_inputs["input_ids"],
                attention_mask=emb_inputs["attention_mask"],
            ))

            # 3) compute KL divergence
            new_dist = kl_divergence(frozen_embeddings, new_embeddings)

            # 4) pad to max_len if shorter
            seq_len = tf.shape(generations)[1]
            pad_len = max_len - seq_len
            if pad_len > 0:
                pad_tokens = tf.fill([batch_size, pad_len], self.tokenizer.pad_token_id)
                generations = tf.concat([generations, pad_tokens], axis=1)

            # 5) update closest
            if closest_generations is None:
                closest_generations = generations
                closest_distances = new_dist
            else:
                mask = new_dist < closest_distances
                mask_exp = tf.expand_dims(mask, 1)
                closest_generations = tf.where(mask_exp, generations, closest_generations)
                closest_distances = tf.where(mask, new_dist, closest_distances)

        return closest_generations