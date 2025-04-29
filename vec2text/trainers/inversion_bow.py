from typing import Dict, Any
import tensorflow as tf

from .base_trainer_tf import BaseTrainer


class InversionTrainerBagOfWords(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

    def compute_metrics_func(self, eval_preds) -> Dict[str, float]:

        return {}

    def generate(self, inputs: Dict[str, tf.Tensor], generation_kwargs: Dict[str, Any]) -> tf.Tensor:

        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)
