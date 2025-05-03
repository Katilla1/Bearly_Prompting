import tensorflow as tf
import numpy as np
import collections
import copy
import logging
import os
import random
from typing import Callable, Dict, List, Tuple, Union
import evaluate
import nltk
import scipy.stats
import tqdm
from transformers import TFPreTrainedModel
import vec2text

logger = logging.getLogger(__name__)

DEFAULT_INPUT_STRING = "Twas brillig, and the slithy toves, Did gyre and gimble in the wabe, All mimsy were the borogoves, And the mome raths outgrabe."

def sem(L: List[float]) -> float:
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result

def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)

def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total

class TFBaseTrainer:
    def __init__(
        self,
        model: TFPreTrainedModel,
        tokenizer,
        embedder_tokenizer,
        training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = []

        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
        }

        # Setup optimizer and learning rate schedule
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        
    def enable_emb_cos_sim_metric(self) -> None:
        self.additional_metrics.append(vec2text.metrics.EmbeddingCosineSimilarity())

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    def train(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
            
        # Convert dataset to TF dataset
        train_ds = self.train_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator
        )

        # Training loop
        num_update_steps_per_epoch = len(train_ds)
        num_train_steps = num_update_steps_per_epoch * self.args.num_train_epochs

        for epoch in range(int(self.args.num_train_epochs)):
            print(f"\nEpoch {epoch+1}/{int(self.args.num_train_epochs)}")
            progress_bar = tqdm.tqdm(train_ds, total=num_update_steps_per_epoch)
            
            for step, batch in enumerate(progress_bar):
                with tf.GradientTape() as tape:
                    outputs = self.model(batch, training=True)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                progress_bar.set_postfix({"loss": float(loss)})
                
                if step % self.args.eval_steps == 0 and self.eval_dataset is not None:
                    metrics = self.evaluate()
                    print(f"\nEvaluation metrics: {metrics}")

    def evaluate(self):
        if self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")
            
        eval_ds = self.eval_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator
        )
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm.tqdm(eval_ds, desc="Evaluating"):
            outputs = self.model(batch)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            predictions = tf.argmax(logits, axis=-1)
            
            all_preds.extend(predictions.numpy())
            all_labels.extend(batch["labels"].numpy())
            
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)
        
        # Generate text and compute generation metrics
        generation_metrics = self.eval_generation_metrics(eval_ds)
        metrics.update(generation_metrics)
        
        return metrics

    def _compute_metrics(self, preds, labels):
        accuracy = self.metric_accuracy.compute(predictions=preds, references=labels)
        return accuracy

    def generate(self, inputs, generation_kwargs):
        return self.model.generate(inputs, **generation_kwargs)

    def eval_generation_metrics(self, eval_ds):
        # Similar to PyTorch version but adapted for TF
        preds_sample_list = []
        preds_sample_labels_list = []
        all_inputs = {}
        
        for batch in tqdm.tqdm(eval_ds.take(100), desc="Generating"):
            generated_ids = self.generate(batch, self.gen_kwargs)
            preds_sample_list.extend(generated_ids.numpy())
            preds_sample_labels_list.extend(batch["labels"].numpy())
            
            for k, v in batch.items():
                all_inputs.setdefault(k, []).extend(v.numpy())
        
        # Decode predictions and compute metrics
        decoded_preds = self.tokenizer.batch_decode(preds_sample_list, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(preds_sample_labels_list, skip_special_tokens=True)
        
        metrics = self._text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds,
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )
        
        return metrics

    def _text_comparison_metrics(self, predictions_ids, predictions_str, references_ids, references_str):
        # Same implementation as PyTorch version
        if not len(predictions_ids):
            return {}
            
        # Token-level metrics
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        
        for i in range(len(predictions_str)):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1 = 2 * precision * recall / (precision + recall + 1e-20)
            
            f1s.append(f1)
            precision_sum += precision
            recall_sum += recall
            
            num_overlapping_words.append(count_overlapping_ngrams(true_words, pred_words, 1))
            num_overlapping_bigrams.append(count_overlapping_ngrams(true_words, pred_words, 2))
            num_overlapping_trigrams.append(count_overlapping_ngrams(true_words, pred_words, 3))

        # Compute BLEU and ROUGE scores
        bleu_results = np.array([
            self.metric_bleu.compute(predictions=[p], references=[r])["score"]
            for p, r in zip(predictions_str, references_str)
        ])
        
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str,
            references=references_str
        )

        metrics = {
            "token_set_precision": precision_sum / len(predictions_str),
            "token_set_recall": recall_sum / len(predictions_str),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
            "bleu_score": bleu_results.mean(),
            "bleu_score_sem": sem(bleu_results),
            "rouge_score": rouge_result["rouge1"]
        }
        
        return metrics 