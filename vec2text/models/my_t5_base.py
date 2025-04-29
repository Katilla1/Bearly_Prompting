from typing import List, Optional, Tuple, Union, Dict, Any
import inspect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class T5SparseEncoderTF(TFT5ForConditionalGeneration):
    def my_encoder(self, input_ids, output_attentions=True):
        mode = self.mode
        ss = tf.shape(input_ids)
        if mode == 'single_sentence':
            input_ids = input_ids[:, -1, :]
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions,
                return_dict=True
            )
            hidden_states = encoder_outputs.last_hidden_state
        elif mode == 'full_attention':
            input_ids = tf.reshape(input_ids, (ss[0], ss[1] * ss[2]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions,
                return_dict=True
            )
            hidden_states = encoder_outputs.last_hidden_state
        else:
            input_ids = tf.reshape(input_ids, (-1, ss[2]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions,
                return_dict=True
            )
            if mode == 'last_token':
                hidden_states = encoder_outputs.last_hidden_state
                hidden_states = tf.reshape(hidden_states, (ss[0], ss[1], ss[2], -1))
                hidden_states = hidden_states[:, :, -1, :]
            elif mode == 'average_pooling':
                hidden_states = encoder_outputs.last_hidden_state
                hidden_states = tf.reshape(hidden_states, (ss[0], ss[1], ss[2], -1))
                hidden_states = tf.reduce_mean(hidden_states, axis=1)
            elif mode == 'just_sparse':
                hidden_states = encoder_outputs.last_hidden_state
                hidden_states = tf.reshape(hidden_states, (ss[0], ss[1] * ss[2], -1))
        attentions = None
        if output_attentions:
            attentions = encoder_outputs.attentions
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=attentions
        )

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        decoder_input_ids: Optional[tf.Tensor] = None,
        decoder_attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        decoder_head_mask: Optional[tf.Tensor] = None,
        cross_attn_head_mask: Optional[tf.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[Tuple[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        decoder_inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[tf.Tensor], Seq2SeqLMOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")
git b
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="tf"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="tf").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.my_encoder(input_ids, output_attentions=output_attentions)
        hidden_states = encoder_outputs.last_hidden_state
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fn = SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            token_losses = loss_fn(labels, lm_logits)
            loss = tf.reduce_mean(token_losses)
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + (encoder_outputs,)
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: tf.Tensor,
        model_kwargs: Dict[str, Any],
        model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # # as the inputs.
        # if hasattr(self, "hf_device_map"):
        #     if hasattr(encoder, "_hf_hook"):
        #         encoder._hf_hook.io_same_device = True
        #     else:
        #         add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))
        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(self.my_encoder).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }
        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        hidden_states = self.my_encoder(inputs_tensor)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
        return model_kwargs