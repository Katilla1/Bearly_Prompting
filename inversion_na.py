# trainers/inversion_na_tf.py
import math
from typing import Dict, Any, Optional

import tensorflow as tf
from transformers import TFTrainer

from .base_trainer_tf import BaseTrainer


class InversionTrainerNonAutoregressive(BaseTrainer):
    """Trainer for non-autoregressive inversion, with TensorFlow backend."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Copy tokenizers and embedder call
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        """Delegate to the TF seq2seq generate method."""
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def evaluate(
        self,
        eval_dataset: tf.data.Dataset = None,
        ignore_keys: Optional[Any] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics, then compute perplexity from loss.
        """
        # 1) run standard TFTrainer.evaluate
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        # 2) compute perplexity if possible
        loss_key = f"{metric_key_prefix}_loss"
        try:
            loss_val = metrics[loss_key]
            ppl = math.exp(loss_val)
        except KeyError:
            ppl = -1.0
        except OverflowError:
            ppl = float("inf")
        metrics[f"{metric_key_prefix}_perplexity"] = ppl
        return metrics
