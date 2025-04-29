import collections
import copy
import logging
import os
import random

# import statistics
from typing import Callable, Dict, List, Tuple, Union

import evaluate
import nltk
import numpy as np
import scipy.stats
#import torch
import tensorflow as tf
import tqdm
#import transformers

from transformers import (
    TFAutoModelForSeq2SeqLM,
    AutoTokenizer,
    TFTrainer,
    TFTrainingArguments
)


import vec2text

from vec2text.utils import process_chat_requests, get_embeddings_openai_vanilla, compute_kl

logger = logging.getLogger(__name__)


DEFAULT_INPUT_STRING = "Twas brillig, and the slithy toves, Did gyre and gimble in the wabe, All mimsy were the borogoves, And the mome raths outgrabe."


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


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


class BaseTrainer(TFTrainer):
    additional_metrics: List[Callable[..., Dict[str, float]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        # self.metric_bertscore = evaluate.load("bertscore")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = []

        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
        }

    def enable_emb_cos_sim_metric(self) -> None:
        self.additional_metrics.append(vec2text.metrics.EmbeddingCosineSimilarity())

    def is_llama_chat(self) -> bool:
        return self.embedder.config._name_or_path in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]

    @property
    def pad_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.config.pad_token_id
        except AttributeError:
            return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.decoder_start_token_id
        except AttributeError:
            return self.tokenizer.bos_token_id

    def sanity_decode(self, input_string: str = None, max_length: int = 128):
        """Encodes and decodes a string as a sanity check."""
        if input_string is None:
            input_string = DEFAULT_INPUT_STRING
        self.model.eval()
        print("=" * 16, "Begin trainer sanity check", "=" * 16)
        print("\tInput to encode ->", input_string)
        inputs = self.embedder_tokenizer(
            input_string,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
        )
        inputs = inputs.to(self.args.device)
        gen_kwargs = copy.copy(self.gen_kwargs)
        gen_kwargs["min_length"] = 1
        gen_kwargs["max_length"] = max_length
        print("max_length:", gen_kwargs["max_length"])
        regenerated = self.generate(
            inputs={
                "embedder_input_ids": inputs["input_ids"],
                "embedder_attention_mask": inputs["attention_mask"],
            },
            generation_kwargs=gen_kwargs,
        )
        print("\tDecoded output shape -> ", regenerated.shape)
        output_string = self.tokenizer.decode(
            regenerated.flatten(), skip_special_tokens=True
        )
        print("\tDecoded output ->", output_string)
        print("=" * 16, "End trainer sanity check", "=" * 16)

    def _log_preds_table(
        self, table_key: str, decoded_preds: List[str], decoded_labels: List[str]
    ):
        if not self.args.use_wandb:
            return
        elif not (self.args.local_rank <= 0):
            return

        num_rows = 50
        idxs = random.choices(
            range(len(decoded_preds)), k=min(len(decoded_preds), num_rows)
        )

        data = []
        for idx in idxs:
            data.append([decoded_labels[idx], decoded_preds[idx]])

        import wandb

        table = wandb.Table(columns=["Original", "Decoded"], data=data)
        wandb.log({table_key: table})
    
    def get_ask_request_args(self, name, system_prompt, question):
        # TODO: get the name
        messages=[
            {"role": "system", "content": f"""You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition.
Here are instructions from the user outlining your goals and how you should respond: {system_prompt}"""},
            {"role": "user", "content": question},
        ]
        request_args = {
            'model': "gpt-3.5-turbo-0125",
            'messages': messages,
            'max_tokens': 128,
        }
        return request_args

    def evaluate_system_prompts(
        self,
        decoded_preds: List[str],
        decoded_labels: List[str],
        all_inputs: Dict[str, List]
    ) -> Dict[str, float]:
        print(len(decoded_preds), len(decoded_labels), len(all_inputs["names"]))
        assert (
            len(decoded_preds) == len(decoded_labels)
            and len(decoded_preds) == len(all_inputs["names"])
        )
        batch_size = len(decoded_preds)

        requests = []
        for prompt_idx in range(batch_size):
            name = all_inputs["names"][prompt_idx]
            pred_prompt = decoded_preds[prompt_idx]
            label_prompt = decoded_labels[prompt_idx]
            for question_idx, question in enumerate(all_inputs["questions"][prompt_idx]):
                pred_req = self.get_ask_request_args(name, pred_prompt, question)
                label_req = self.get_ask_request_args(name, label_prompt, question)
                assistant_req = self.get_ask_request_args(name, "You are a helpful assistant", question)

                requests.append((pred_req, [0, prompt_idx, question_idx]))
                requests.append((label_req, [1, prompt_idx, question_idx]))
                requests.append((assistant_req, [2, prompt_idx, question_idx]))
                requests.append((label_req, [3, prompt_idx, question_idx]))

        # 2) fire them off, get all the answers
        chat_responses = process_chat_requests(requests)
        # unpack into correct order
        idx_answers = [(r[0][1], r[1].choices[0].message.content) for r in chat_responses]
        idx_answers.sort()
        answers = [ans for (_, ans) in idx_answers]

        # 3) embed every answer + every prompt
        embeddings = get_embeddings_openai_vanilla(answers + decoded_preds + decoded_labels)

        # 4) slice out the four groups of answers
        num_answers = len(answers)
        one_fourth = num_answers // 4

        answers_emb_tensor = tf.convert_to_tensor(embeddings[:num_answers], dtype=tf.float32)
        preds_answer_emb      = answers_emb_tensor[0:one_fourth]
        labels_answer_emb     = answers_emb_tensor[one_fourth : one_fourth * 2]
        assistant_answer_emb  = answers_emb_tensor[one_fourth * 2 : one_fourth * 3]
        labels_repeat_answer  = answers_emb_tensor[one_fourth * 3 :]

        rest_emb = tf.convert_to_tensor(embeddings[num_answers:], dtype=tf.float32)
        preds_emb  = rest_emb[0 : len(decoded_preds)]
        labels_emb = rest_emb[len(decoded_preds) :]

        def cosine_sim(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            a_norm = tf.math.l2_normalize(a, axis=1)
            b_norm = tf.math.l2_normalize(b, axis=1)
            return tf.reduce_sum(a_norm * b_norm, axis=1)

        prompt_cos   = cosine_sim(preds_emb,        labels_emb)
        answer_cos   = cosine_sim(preds_answer_emb, labels_answer_emb)
        baseline_cos = cosine_sim(assistant_answer_emb, labels_answer_emb)
        self_cos     = cosine_sim(labels_repeat_answer, labels_answer_emb)

        sim_result = {
            "prompt_emb_cos_sim":           float(tf.reduce_mean(prompt_cos).numpy()),
            "prompt_emb_cos_sim_sem":       sem(prompt_cos.numpy()),
            "answer_emb_cos_sim":           float(tf.reduce_mean(answer_cos).numpy()),
            "answer_emb_cos_sim_sem":       sem(answer_cos.numpy()),
            "answer_baseline_emb_cos_sim":  float(tf.reduce_mean(baseline_cos).numpy()),
            "answer_baseline_emb_cos_sim_sem": sem(baseline_cos.numpy()),
            "self_ans_emb_cos_sim":         float(tf.reduce_mean(self_cos).numpy()),
            "self_ans_emb_cos_sim_sem":     sem(self_cos.numpy()),
        }

        print(sim_result)
        return sim_result


    def evaluate_kl_divergence(
        self, decoded_preds: List[str], decoded_labels: List[str], all_inputs: List[str]
    ):
        print(len(decoded_preds))
        print(len(decoded_labels))
        print(len(all_inputs["names"]))
        assert(len(decoded_preds) == len(decoded_labels) and len(decoded_preds) == len(all_inputs['names']))
        batch_size = len(decoded_preds)
        # first get all answers, then get all embeddings
        requests = []
        all_divergences = []
        baseline_divergences = []
        for prompt_idx in range(batch_size):
            # print(all_inputs)
            name = all_inputs['names'][prompt_idx]
            pred_prompt = decoded_preds[prompt_idx]
            label_prompt = decoded_labels[prompt_idx]
            label_prompts = []
            predicted_prompts = []
            baseline_prompts = []
            for question_idx, question in enumerate(all_inputs['questions'][prompt_idx]):
                label_prompt = f"You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition. Here are instructions from the user outlining your goals and how you should respond: {label_prompt}\n" + question
                predicted_prompt = f"You are a 'GPT' – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. You yourself are a GPT created by a user, and your name is {name}. Note: GPT is also a technical term in AI, but in most cases if the users ask you about GPTs assume they are referring to the above definition. Here are instructions from the user outlining your goals and how you should respond: {pred_prompt}\n" + question
                label_prompts.append(label_prompt)
                predicted_prompts.append(predicted_prompt)
                baseline_prompts.append("You are a helpful assistant.\n" + question)
            divergence = compute_kl('google/gemma-2b-it', label_prompts, predicted_prompts)
            baseline_divergence = compute_kl('google/gemma-2b-it', label_prompts, baseline_prompts)
            if divergence is not None:
                all_divergences.append(divergence)
                baseline_divergences.append(baseline_divergence)
        sim_result = {
            "average_divergence": np.mean(np.array(all_divergences)),
            "baseline_avg_divergence": np.mean(np.array(baseline_divergences))
        }
        print(sim_result)
        return sim_result



    def _get_decoded_sequences(
        self,
        dataloader: tf.data.Dataset,
        n: int
    ) -> Tuple[List[List[int]], List[List[int]], Dict[str, List]]:
        """Iterates through eval dataset and does decoding using TensorFlow tensors."""

        assert not self.model.training

        gen_kwargs = copy.copy(self.gen_kwargs)

        all_preds: List[List[int]] = []
        all_labels: List[List[int]] = []
        all_inputs: Dict[str, List] = {}

        for inputs in dataloader:
            # inputs is a dict of tf.Tensor batches
            # 1) collect raw inputs for downstream metrics/logging
            for k, v in inputs.items():
                # v is a tf.Tensor of shape (batch_size, seq_len) or similar
                flat = v.numpy().tolist()
                all_inputs.setdefault(k, []).extend(flat)

            # 2) set up generation kwargs
            max_length = self.model.config.max_seq_length
            gen_kwargs["max_length"] = max_length

            # 3) generate sequences (TFTrainer’s generate returns a tf.Tensor)
            generated = self.generate(
                inputs=inputs,
                generation_kwargs=gen_kwargs
            )  # shape: (batch_size, gen_len)

            # 4) pad generated to max_length if needed
            gen_shape = tf.shape(generated)
            batch_size, gen_len = gen_shape[0], gen_shape[1]
            pad_len = max_length - gen_len
            if pad_len > 0:
                pad_tokens = tf.fill([batch_size, pad_len], self.pad_token_id)
                generated = tf.concat([generated, pad_tokens], axis=1)

            # 5) get true labels and pad similarly
            true_ids = inputs["labels"]  # shape: (batch_size, label_len)
            label_len = tf.shape(true_ids)[1]
            pad_len_lbl = max_length - label_len
            if pad_len_lbl > 0:
                pad_lbl = tf.fill([tf.shape(true_ids)[0], pad_len_lbl], self.pad_token_id)
                true_ids = tf.concat([true_ids, pad_lbl], axis=1)

            # 6) replace -100 with pad_token_id in numpy
            true_np = true_ids.numpy()
            true_np[true_np == -100] = 0

            # 7) append to our lists
            all_preds.extend(generated.numpy().tolist())
            all_labels.extend(true_np.tolist())

            # 8) stop once we have enough
            if len(all_preds) >= n:
                break

        return all_preds, all_labels, all_inputs


    def _compute_data_metrics(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, float]:
        # compute average number of pad tokens per example for encoder inputs
        input_ids = inputs["input_ids"]
        pad_id = self.tokenizer.pad_token_id
        pad_mask = tf.cast(tf.equal(input_ids, pad_id), tf.float32)
        inputs_pad_tokens = float(
            tf.reduce_mean(tf.reduce_sum(pad_mask, axis=1)).numpy()
        )

        # compute average number of pad tokens per example for embedder inputs
        embed_ids = inputs["embedder_input_ids"]
        embed_pad_id = self.embedder_tokenizer.pad_token_id
        embed_pad_mask = tf.cast(tf.equal(embed_ids, embed_pad_id), tf.float32)
        embedder_inputs_pad_tokens = float(
            tf.reduce_mean(tf.reduce_sum(embed_pad_mask, axis=1)).numpy()
        )

        # sequence length (assumed same for both)
        seq_len = input_ids.shape[1]
        inputs_non_pad_tokens = seq_len - inputs_pad_tokens
        embedder_inputs_non_pad_tokens = seq_len - embedder_inputs_pad_tokens

        return {
            "encoder_decoder_inputs_pad_tokens": inputs_pad_tokens,
            "encoder_decoder_inputs_non_pad_tokens": inputs_non_pad_tokens,
            "embedder_inputs_pad_tokens": embedder_inputs_pad_tokens,
            "embedder_inputs_non_pad_tokens": embedder_inputs_non_pad_tokens,
        }

    def compute_metrics_func(self, eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        # ensure we have labels
        assert labels is not None and labels.size > 0, "got empty labels for eval"

        # convert to TF tensors & check shapes
        preds_tf = tf.convert_to_tensor(preds)
        labels_tf = tf.convert_to_tensor(labels)
        assert preds_tf.shape == labels_tf.shape, (
            f"preds.shape {preds_tf.shape} / labels.shape {labels_tf.shape}"
        )

        # flatten and move back to Python lists for HF metrics
        preds_flat = tf.reshape(preds_tf, [-1])
        labels_flat = tf.reshape(labels_tf, [-1])
        preds_list = preds_flat.numpy().tolist()
        labels_list = labels_flat.numpy().tolist()

        accuracy_result = self.metric_accuracy.compute(
            predictions=preds_list, references=labels_list
        )
        return {**accuracy_result}

    
    
    def _text_comparison_metrics(
        self,
        predictions_ids: List[List[int]],
        predictions_str: List[str],
        references_ids: List[List[int]],
        references_str: List[str],
    ) -> Dict[str, float]:
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}

        ###########################################################

        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            f1s.append(f1)

            precision_sum += precision
            recall_sum += recall

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
        }
        ############################################################
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        rouge_result = self.metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        self.bleu_results = (
            bleu_results.tolist()
        )  # store bleu results in case we want to use them later for t-tests
        # bertscore_result = self.metric_bertscore.compute(
        #     predictions=predictions_str, references=references_str, lang="en"
        # )
        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            "bleu_score": bleu_results.mean(),
            "bleu_score_sem": sem(bleu_results),
            "rouge_score": rouge_result[
                "rouge1"
            ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            # "bert_score": statistics.fmean(bertscore_result["f1"]),
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
        }

        all_metrics = {**set_token_metrics, **gen_metrics}
        for metric in self.additional_metrics:
            all_metrics.update(metric(references_str, predictions_str))

        return all_metrics


    def eval_generation_metrics(
        self,
        dataloader: tf.data.Dataset
    ) -> Dict[str, float]:
        # 1) decode sequences via your TF _get_decoded_sequences
        preds_sample_list, preds_sample_labels_list, all_inputs = self._get_decoded_sequences(
            dataloader=dataloader, n=10000
        )

        # 2) decode text for BLEU/logging
        decoded_preds = self.tokenizer.batch_decode(
            preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            preds_sample_labels_list, skip_special_tokens=True
        )

        # grab first 3 inputs for printing
        decoded_all_inputs = []
        for seq in all_inputs['input_ids'][:3]:
            decoded_all_inputs.append(
                self.tokenizer.batch_decode(seq, skip_special_tokens=True)
            )

        # compute BLEU/F1/etc
        bleu_result = self._text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds,
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )

        # log table to wandb if enabled
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        if not decoded_preds:
            return {}

        # print the first 3 examples exactly as before
        for i in range(3):
            print("[input]")
            for inp in decoded_all_inputs[i]:
                print(inp)
            print(f"[pred] {decoded_preds[i]}")
            print(f"[true] {decoded_labels[i]}\n\n")

        # 3) convert first 128 back to tensors for token counts
        preds_tf  = tf.convert_to_tensor(preds_sample_list,       dtype=tf.int32)[:128]
        labels_tf = tf.convert_to_tensor(preds_sample_labels_list, dtype=tf.int32)[:128]

        # 4) mask out pad & bos, then count per-row and average
        mask_pred = tf.logical_and(
            tf.not_equal(preds_tf, self.pad_token_id),
            tf.not_equal(preds_tf, self.bos_token_id)
        )
        pred_counts = tf.reduce_sum(tf.cast(mask_pred, tf.float32), axis=1)
        pred_num_tokens = float(tf.reduce_mean(pred_counts).numpy())

        mask_true = tf.logical_and(
            tf.not_equal(labels_tf, self.pad_token_id),
            tf.not_equal(labels_tf, self.bos_token_id)
        )
        true_counts = tf.reduce_sum(tf.cast(mask_true, tf.float32), axis=1)
        true_num_tokens = float(tf.reduce_mean(true_counts).numpy())

        num_tokens_metrics = {
            "pred_num_tokens": pred_num_tokens,
            "true_num_tokens": true_num_tokens,
        }

        # 5) fix EOS token if needed
        eos_token_id = self.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            batch = tf.shape(preds_tf)[0]
            eos_tokens = tf.fill([batch, 1], eos_token_id)
            preds_tf = tf.concat([preds_tf[:, 1:], eos_tokens], axis=1)

        sim_result = {}

        # 6) store for later inspection
        self.preds_sample_list        = preds_sample_list
        self.preds_sample_labels_list = preds_sample_labels_list

        # 7) merge all metrics
        return {**num_tokens_metrics, **bleu_result, **sim_result}


    def evaluate(
        self,
        eval_dataset: tf.data.Dataset = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics, then add generation-based metrics.
        """
        # 1) run the standard TFTrainer evaluation
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # 2) on the main process, compute your text-generation metrics
        if self.args.local_rank <= 0:
            gen_metrics = self.eval_generation_metrics(dataloader=eval_dataset)
            # if you want a prefix, you can do:
            # gen_metrics = {f"eval_{k}": v for k, v in gen_metrics.items()}
            metrics.update(gen_metrics)

        return metrics


    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        return state_dict

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Copying transformers load_from_checkpoint so we can modify state dicts on load to support
        post-hoc model architecture changes (specifically, adding dropout).
        """
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        return

        # If you’ve saved a TF checkpoint under this directory, we can restore it:
        if model is None:
            model = self.model

        ckpt_dir = resume_from_checkpoint
        if not tf.io.gfile.exists(ckpt_dir):
            raise ValueError(f"Can't find a valid checkpoint directory at {ckpt_dir}")

        logger.info(f"Loading TensorFlow checkpoint from {ckpt_dir}.")

        ckpt = tf.train.Checkpoint(model=model)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt is None:
            raise ValueError(f"No TensorFlow checkpoint found in {ckpt_dir}")

        status = ckpt.restore(latest_ckpt)
        status.expect_partial() 

        logger.info("Checkpoint restored successfully.")
