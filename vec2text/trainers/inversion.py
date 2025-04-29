import math
from typing import Dict, Any, Optional

import tensorflow as tf
from transformers import TFTrainer

from .base_trainer_tf import BaseTrainer


class InversionTrainer(BaseTrainer):
    """Trainer for inversion tasks using a TF seq2seq model."""

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
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(
        self,
        model: tf.keras.Model,
        inputs: Dict[str, tf.Tensor]
    ) -> tf.Tensor:

        return super().training_step(model, inputs)

    def evaluate(
        self,
        eval_dataset: tf.data.Dataset = None,
        ignore_keys: Optional[Any] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:

        # 1) run standard TFTrainer.evaluate
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        # 2) compute perplexity
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

    def _remap_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        No state dict remapping needed in TF
        """
        return state_dict
