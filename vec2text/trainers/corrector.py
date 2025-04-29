import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
#import torch
#import torch.nn as nn
import tensorflow as tf
import transformers

from vec2text.models import CorrectorEncoderModel
from vec2text.models.model_utils import freeze_params
from vec2text.run_args import TrainingArguments
from vec2text.utils import dataset_map_multi_worker

from .base import BaseTrainer
from .inversion import InversionTrainer

logger = logging.getLogger(__name__)


class Corrector(BaseTrainer):
    """Trains an encoder model to generate embeddings that recursively correct of an
    InversionTrainer.
    """

    train_dataset: datasets.Dataset
    eval_dataset: Dict[str, datasets.Dataset]
    # TODO: don't assume that the encoder has to have the same tokenizer as the encoder_decoder
    # or embedder model.

    _hypothesis_cache: Dict[str, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]

    # If set, only take hypothesis if it improves our distance to ground-truth.
    return_best_hypothesis: bool = False

    # Initialize from this hypothesis, if set
    initial_hypothesis_str: Optional[str] = None

    def __init__(
        self,
        model: CorrectorEncoderModel,
        inversion_trainer: InversionTrainer,
        args: Optional[TrainingArguments],
        **kwargs,
    ):
        # Freeze other model params
        freeze_params(inversion_trainer.model)
        # We're training this corrector model to correct outputs from
        # a model trained & loaded via the inversion trainer.
        self.inversion_trainer = inversion_trainer
        self.inversion_trainer.model.use_frozen_embeddings_as_input = True
        super().__init__(
            model=model,
            args=args,
            train_dataset=self.inversion_trainer.train_dataset,
            eval_dataset=self.inversion_trainer.eval_dataset,
            **kwargs,
        )
        self.tokenizer = self.inversion_trainer.model.tokenizer
        self.embedder_tokenizer = self.inversion_trainer.model.embedder_tokenizer
        self.call_embedding_model = self.inversion_trainer.model.call_embedding_model

        self.initial_hypothesis_str = None

        # Number of steps of self-correction
        self.num_gen_recursive_steps = 1
        self.sequence_beam_width = 1

        # If set, return closest (in embedding space) hypothesis we see during generation
        self.return_best_hypothesis = False

        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def evaluate(
        self,
        eval_dataset: tf.data.Dataset = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics, then add multi-round generation metrics
        for msmarco/nq splits.
        """
        # 1) run the standard TFTrainer.evaluate
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        # 2) on the chief process, for specific splits, run multi-round generation
        if getattr(self.args, "process_index", 0) == 0 and metric_key_prefix in {"eval_msmarco", "eval_nq"}:
            n_rounds = 5
            self.num_gen_recursive_steps = n_rounds
            gen_metrics = self.eval_generation_metrics(dataloader=eval_dataset)
            # prefix each generated metric
            prefixed = {
                f"{metric_key_prefix}_{n_rounds}round_{k}": v
                for k, v in gen_metrics.items()
            }
            metrics.update(prefixed)
            self.num_gen_recursive_steps = 1

        return metrics


    def _precompute_hypothesis_and_embedding(
        self,
        ds_inputs: Dict[str, Any],
        collator=None,
    ) -> Dict[str, Any]:
        """
        Given dataset inputs, generate a hypothesis and embedding, caching them in ds_inputs.
        """
        # 1) Pad inputs (TF tensors)
        padded = collator.tokenizer.pad(
            {k: v for k, v in ds_inputs.items() if k != "labels"},
            padding=collator.padding,
            max_length=collator.max_length,
            pad_to_multiple_of=collator.pad_to_multiple_of,
            return_tensors="tf",
        )

        # 2) Generate hypothesis & embeddings
        fe, hi, hm, he = self._get_hypothesis_uncached(inputs=padded)

        # 3) Cache back into ds_inputs as numpy arrays for the HF dataset
        ds_inputs["frozen_embeddings"] = fe.numpy()
        ds_inputs["hypothesis_embedding"] = he.numpy()

        # 4) Truncate padding so we can batch by length later
        ds_inputs["hypothesis_input_ids"] = []
        ds_inputs["hypothesis_attention_mask"] = []
        for ids, mask in zip(hi.numpy(), hm.numpy()):
            length = int(mask.sum())
            ds_inputs["hypothesis_input_ids"].append(ids[: length + 1])
            ds_inputs["hypothesis_attention_mask"].append(mask[: length + 1])

        # 5) (Optional) debug print
        print("input_ids[0]:", self.tokenizer.decode(ds_inputs["input_ids"][0]))
        print(
            "hypothesis_input_ids[0]:",
            self.tokenizer.decode(ds_inputs["hypothesis_input_ids"][0]),
        )

        return ds_inputs


    def _preprocess_dataset_hypotheses(
        self,
        dataset: datasets.Dataset,
        filter_correct_examples: bool = False
    ) -> Tuple[datasets.Dataset, str]:
        """
        Precompute or load cached hypotheses and embeddings for a HF dataset,
        storing them under VEC2TEXT_CACHE.
        """
        cache_dir = os.environ["VEC2TEXT_CACHE"]
        assert os.path.exists(cache_dir), f"Cache dir {cache_dir} not found"
        cache_path = os.path.join(cache_dir, f"{dataset._fingerprint}_hypotheses.cache")

        if not os.path.exists(cache_path):
            print(f"[{dataset.builder_name}] Saving hypotheses to {cache_path}")
            # 1) Map over the dataset to generate frozen_embeddings & hypotheses
            dataset = dataset_map_multi_worker(
                dataset=dataset,
                map_fn=functools.partial(
                    self._precompute_hypothesis_and_embedding,
                    collator=self.data_collator,
                ),
                batched=True,
                batch_size=self.args.train_batch_size * 2,
                desc="Precomputing hypotheses for data",
            )

            # 2) Optionally filter out examples where hypothesis == frozen embedding
            if filter_correct_examples:
                old_len = len(dataset)

                def embedding_is_not_correct(ex):
                    fe = tf.convert_to_tensor(ex["frozen_embeddings"], dtype=tf.float32)
                    he = tf.convert_to_tensor(ex["hypothesis_embedding"], dtype=tf.float32)
                    # keep examples where not all elements are close
                    correct_mask = tf.reduce_all(tf.math.is_close(fe, he, atol=1e-6), axis=1)
                    return (~correct_mask).numpy().tolist()

                dataset = dataset.filter(
                    embedding_is_not_correct,
                    batched=True,
                    batch_size=1024,
                )
                print(f"filtered {old_len} → {len(dataset)} examples")

            # 3) Save to disk for future runs
            dataset.save_to_disk(cache_path)
        else:
            logging.info("Loading hypotheses from %s", cache_path)
            print(f"[{dataset.builder_name}] Loading hypotheses from {cache_path}")
            dataset = datasets.load_from_disk(cache_path)

        # 4) Make sure HF dataset returns NumPy arrays (so TF collators can consume)
        dataset.set_format("numpy")
        return dataset, cache_path

    def precompute_hypotheses(self) -> None:
        """Generates and embeds hypotheses using `self.inversion_trainer`.

        Returns path to precomputed-and-saved train dataset, which is sometimes
        useful for outside processes.
        """
        logger.info("Precomputing frozen embedding & hypotheses before training")

        self.train_dataset, train_cache_path = self._preprocess_dataset_hypotheses(
            dataset=self.train_dataset, filter_correct_examples=True
        )
        for k, v in self.eval_dataset.items():
            self.eval_dataset[k], _ = self._preprocess_dataset_hypotheses(
                dataset=v, filter_correct_examples=False
            )

    def _inner_training_loop(self, *args, **kwargs):
        # Don't let tokenizers run in parallel mode.
        # os.environ["TOKENIZERS_PARALLELISM"] = "False"

        self.model.eval()
        self.model.to(self.args.device)
        self.inversion_trainer.model.to(next(self.model.parameters()).device)
        self.precompute_hypotheses()
        self.model.train()
        self.inversion_trainer.model.cpu()
        return super()._inner_training_loop(*args, **kwargs)

    def generate(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any],
        num_recursive_steps: Optional[int] = None,
        sequence_beam_width: Optional[int] = None,
    ) -> tf.Tensor:
        # Ensure hypothesis inputs present
        if 'frozen_embeddings' in inputs:
            fe = inputs['frozen_embeddings']
            hi = inputs['hypothesis_input_ids']
            hm = inputs['hypothesis_attention_mask']
            he = inputs['hypothesis_embedding']
        else:
            fe, hi, hm, he = self._get_hypothesis_uncached(inputs)
        inputs['frozen_embeddings'] = fe
        inputs['hypothesis_input_ids'] = hi
        inputs['hypothesis_attention_mask'] = hm
        inputs['hypothesis_embedding'] = he
        steps = num_recursive_steps or self.num_gen_recursive_steps
        beam_w = sequence_beam_width or self.sequence_beam_width
        steps_done = 0
        total_best = None
        while steps > 0:
            gen_ids, he, best_scores = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=steps,
                num_recursive_steps_so_far=steps_done,
                sequence_beam_width=beam_w,
            )
            inputs['hypothesis_input_ids'] = gen_ids
            inputs['hypothesis_attention_mask'] = tf.cast(
                tf.not_equal(gen_ids, self.pad_token_id), tf.int32
            )
            inputs['hypothesis_embedding'] = he
            steps -= 1
            steps_done += 1
            if best_scores is not None and total_best is not None:
                if tf.reduce_all(tf.math.is_close(best_scores, total_best, atol=1e-3)):
                    print(
                        "scores stopped increasing! stopping early after",
                        steps_done,
                        "steps",
                    )
                    break
            total_best = best_scores
        return gen_ids


    def _generate_with_beam(
        self,
        inputs: Dict[str, tf.Tensor],
        generation_kwargs: Dict[str, Any],
        num_recursive_steps: int,
        num_recursive_steps_so_far: int,
        sequence_beam_width: int,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        assert num_recursive_steps >= 1
        frozen_embeddings = inputs["frozen_embeddings"]

        # Set up beam size in kwargs
        if not generation_kwargs.get("do_sample", False):
            num_return_sequences = max(
                sequence_beam_width, generation_kwargs.get("num_beams", 1)
            )
            generation_kwargs["num_beams"] = num_return_sequences
            generation_kwargs["num_return_sequences"] = num_return_sequences

        # Step 0: use initial hypothesis if provided
        if num_recursive_steps_so_far == 0 and self.initial_hypothesis_str is not None:
            logger.info(f"Using initial hypothesis: {self.initial_hypothesis_str}")
            batch_size = tf.shape(frozen_embeddings)[0]
            hyp_ids = self.embedder_tokenizer(
                [self.initial_hypothesis_str],
                return_tensors="tf",
                max_length=inputs["hypothesis_input_ids"].shape[1],
                truncation=True,
                padding="max_length",
            )["input_ids"]
            gen_text_ids = tf.tile(hyp_ids, [batch_size, 1])
            bos_id = self.model.encoder_decoder.config.decoder_start_token_id
            bos_tokens = tf.fill([batch_size, 1], bos_id)
            gen_text_ids = tf.concat([bos_tokens, gen_text_ids[:, :-1]], axis=1)
        else:
            # Generate via HF TF generate
            outputs = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                return_dict_in_generate=True,
            )
            gen_text_ids = outputs.sequences

            # Compute sequence‐level scores
            if hasattr(outputs, "beam_indices"):
                transition_scores = self.model.encoder_decoder.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    outputs.beam_indices,
                    normalize_logits=True,
                )
            else:
                transition_scores = self.model.encoder_decoder.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True,
                )
            lp = self.model.encoder_decoder.generation_config.length_penalty
            neg_mask = tf.less(transition_scores, 0)
            output_length = tf.reduce_sum(tf.cast(neg_mask, tf.float32), axis=1)
            gen_text_scores = (
                tf.reduce_sum(transition_scores, axis=1) /
                tf.pow(output_length, lp)
            )

        # Re‐embed to rerank
        hypothesis_embedding = self.embed_generated_hypothesis(input_ids=gen_text_ids)

        # Determine real batch size
        if num_recursive_steps_so_far == 0:
            batch_size = tf.shape(frozen_embeddings)[0]
        else:
            batch_size = tf.shape(frozen_embeddings)[0] // sequence_beam_width

        best_scores = None

        # If beam search expanded the batch
        total_seqs = tf.shape(gen_text_ids)[0]
        if total_seqs > batch_size:
            beam_width = total_seqs // batch_size

            # Reshape to [batch, beam, ...]
            hyp_emb = tf.reshape(hypothesis_embedding, [batch_size, beam_width, -1])
            frozen = tf.expand_dims(frozen_embeddings, 1)  # [batch,1,dim]
            distances = tf.reduce_sum(
                tf.math.l2_normalize(hyp_emb, axis=2) *
                tf.math.l2_normalize(frozen, axis=2),
                axis=2
            )

            if self.return_best_hypothesis:
                scores = distances
            else:
                scores = tf.reshape(gen_text_scores, [batch_size, beam_width])

            # Pick best index per batch
            best_idx = tf.argmax(scores, axis=1, output_type=tf.int32)

            # Gather best hypotheses
            row_idx = tf.range(batch_size, dtype=tf.int32)
            hypothesis_embedding = tf.gather_nd(
                hyp_emb, tf.stack([row_idx, best_idx], axis=1)
            )
            ids_reshaped = tf.reshape(gen_text_ids, [batch_size, beam_width, -1])
            gen_text_ids = tf.gather_nd(
                ids_reshaped, tf.stack([row_idx, best_idx], axis=1)
            )

            # Flatten back
            gen_text_ids = tf.reshape(gen_text_ids, [batch_size * sequence_beam_width, -1])
            hypothesis_embedding = tf.reshape(
                hypothesis_embedding, [batch_size * sequence_beam_width, -1]
            )

            best_scores = tf.reduce_max(scores, axis=1)

        # Sanity check dims
        assert hypothesis_embedding.shape[-1] == frozen_embeddings.shape[-1]

        return gen_text_ids, hypothesis_embedding, best_scores


    def get_frozen_embeddings(
            self,
            embedder_input_ids: tf.Tensor,
            embedder_attention_mask: tf.Tensor,
        ) -> tf.Tensor:
        """
        Compute and return frozen embeddings from the inversion trainer without tracking gradients.
        """
        # Forward through the frozen embedder model
        frozen_embeddings = self.inversion_trainer.call_embedding_model(
            input_ids=embedder_input_ids,
            attention_mask=embedder_attention_mask,
        )
        # Detach to avoid gradient computation
        return tf.stop_gradient(frozen_embeddings)

    def embed_generated_hypothesis(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Embeds a generated hypothesis by decoding to strings, re-tokenizing with the embedder tokenizer,
        and retrieving frozen embeddings.
        """
        ids_list = input_ids.numpy().tolist()
        inputs_str = self.tokenizer.batch_decode(ids_list, skip_special_tokens=True)
        emb_inputs = self.embedder_tokenizer(
            inputs_str,
            max_length=self.model.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )
        return self.get_frozen_embeddings(
            embedder_input_ids=emb_inputs["input_ids"],
            embedder_attention_mask=emb_inputs["attention_mask"],
        )


    def _get_hypothesis_uncached(
        self, inputs: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Generate hypothesis inputs & embeddings when not cached.
        Returns:
            frozen_embeddings: tf.Tensor
            hypothesis_input_ids: tf.Tensor
            hypothesis_attention_mask: tf.Tensor (bool)
            hypothesis_embedding: tf.Tensor
        """
        # 1) Obtain or compute frozen embeddings
        if "frozen_embeddings" in inputs:
            frozen_embeddings = inputs["frozen_embeddings"]
        else:
            assert "embedder_input_ids" in inputs, f"Cannot generate hypothesis with inputs {list(inputs.keys())}"
            frozen_embeddings = self.embed_generated_hypothesis(input_ids=inputs["input_ids"])

        # 2) Prepare generation kwargs
        generation_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            "max_length": self.model.config.max_seq_length,
        }

        # 3) Generate raw hypothesis token IDs
        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={"frozen_embeddings": frozen_embeddings},
            generation_kwargs=generation_kwargs,
        )

        # 4) Build attention mask (True for real tokens, False for padding)
        pad_id = self.model.encoder_decoder.config.pad_token_id
        hypothesis_attention_mask = tf.not_equal(hypothesis_input_ids, pad_id)

        # 5) Embed the generated hypothesis
        hypothesis_embedding = self.embed_generated_hypothesis(input_ids=hypothesis_input_ids)

        return (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_embedding,
        )


    def compute_loss(
        self,
        model: CorrectorEncoderModel,
        inputs: Dict[str, tf.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[tf.Tensor, Dict[str, tf.Tensor]], tf.Tensor]:
        """
        Compute loss for the corrector model, generating hypotheses if needed.
        """
        # Ensure hypothesis embeddings exist or generate them
        try:
            fe = inputs["frozen_embeddings"]
            hi = inputs["hypothesis_input_ids"]
            hm = inputs["hypothesis_attention_mask"]
            he = inputs["hypothesis_embedding"]
        except KeyError:
            fe, hi, hm, he = self._get_hypothesis_uncached(inputs)

        labels = inputs["labels"]
        outputs = model(
            embedding=fe,
            hypothesis_embedding=he,
            hypothesis_input_ids=hi,
            hypothesis_attention_mask=hm,
            labels=labels,
        )
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
            self,
            model: transformers.TFModel,
            inputs: Dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`. Called during TFTrainer.evaluate().
            """
            # In TF everything is eager by default; no .to(device) needed
            loss = self.compute_loss(model=model, inputs=inputs)
            # We don’t return logits or labels here
            return loss, None, None


    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        # TF checkpoint loading doesn’t need PyTorch-style key renames
            return state_dict
