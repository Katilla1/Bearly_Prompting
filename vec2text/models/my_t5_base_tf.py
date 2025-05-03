import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, TFT5Model
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Tokenizer
import inspect
from vec2text.models.inversion_from_logits_emb import InversionFromLogitsEmbModel


class TFT5SparseEncoder(TFT5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'just_sparse'  # default mode

    def my_encoder(self, input_ids, output_attentions=True):
        mode = self.mode
        ss = tf.shape(input_ids)
        
        if mode == 'single_sentence':
            input_ids = input_ids[:, -1, :]
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions
            )
            hidden_states = encoder_outputs[0]
        elif mode == 'full_attention':
            input_ids = tf.reshape(input_ids, (ss[0], ss[1] * ss[2]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions
            )
            hidden_states = encoder_outputs[0]
        else:
            input_ids = tf.reshape(input_ids, (-1, ss[-1]))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=tf.ones_like(input_ids),
                output_attentions=output_attentions
            )
            if mode == 'last_token':
                hidden_states = tf.reshape(encoder_outputs[0], (ss[0], ss[1], ss[2], -1))
                hidden_states = hidden_states[:, :, -1, :]
            elif mode == 'average_pooling':
                hidden_states = tf.reshape(encoder_outputs[0], (ss[0], ss[1], ss[2], -1))
                hidden_states = tf.reduce_mean(hidden_states, axis=1)
            elif mode == 'just_sparse':
                hidden_states = tf.reshape(encoder_outputs[0], (ss[0], ss[1] * ss[2], -1))

        attentions = None
        if output_attentions:
            attentions = encoder_outputs['attentions']

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': None,
            'attentions': attentions
        }

    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        decoder_input_ids: Optional[tf.Tensor] = None,
        decoder_attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        decoder_head_mask: Optional[tf.Tensor] = None,
        cross_attn_head_mask: Optional[tf.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        decoder_inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.my_encoder(input_ids)
        hidden_states = encoder_outputs['last_hidden_state']

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
            training=training,
        )

        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            loss = loss_fn(labels, logits)
            loss = tf.reduce_mean(loss)

        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': decoder_outputs.past_key_values,
            'decoder_hidden_states': decoder_outputs.hidden_states,
            'decoder_attentions': decoder_outputs.attentions,
            'cross_attentions': decoder_outputs.cross_attentions,
            'encoder_last_hidden_state': encoder_outputs['last_hidden_state'],
            'encoder_hidden_states': encoder_outputs['hidden_states'],
            'encoder_attentions': encoder_outputs['attentions'],
        }

    def generate(self, inputs, generation_config=None, **kwargs):
        if isinstance(inputs, dict):
            inputs = inputs['input_ids']
        
        if generation_config is None:
            generation_config = self.generation_config
            
        return super().generate(
            inputs,
            generation_config=generation_config,
            **kwargs
        ) 