"""Qwen2.5-VL forward path for ordered FrozenLake auxiliary-image targets."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from torch.nn import CrossEntropyLoss
from transformers.utils import is_torchdynamo_compiling

from src.constants import IGNORE_INDEX
from src.train.monkey_patch_forward_lvr import (
    Qwen2_5_VLCausalLMOutputWithPast,
    set_lvr_loss_fct,
)


def replace_qwen2_5_with_frozenlake_forward() -> None:
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = (
        qwen2_5_frozenlake_lvr_forward
    )


def qwen2_5_frozenlake_lvr_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    lvr_tokens: Optional[torch.Tensor] = None,
    lvr_tokens_thw: Optional[torch.LongTensor] = None,
    lvr_mode_switch: Optional[torch.Tensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if lvr_mode_switch is not None and torch.any(lvr_mode_switch):
        inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]

    if (
        (lvr_mode_switch is None or not torch.any(lvr_mode_switch))
        and pixel_values is None
        and pixel_values_videos is None
    ):
        dummy_pixel = torch.zeros(784, 1176, device=self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]], device=self.model.visual.device)
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        dummy_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        inputs_embeds += dummy_embeds.mean() * 0

    selected_lvr_embeds = None
    batch_indices = None
    seq_positions = None
    if pixel_values is not None:
        image_embeds = torch.cat(self.model.get_image_features(pixel_values, image_grid_thw), dim=0)
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: tokens={n_image_tokens}, "
                f"features={image_embeds.shape[0]}"
            )
        image_mask = input_ids == self.config.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds)

        if lvr_tokens is not None:
            if lvr_tokens_thw is None:
                raise ValueError("lvr_tokens_thw is required with auxiliary-image targets")
            lvr_mask = input_ids == self.config.lvr_id
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
            selected_lvr_embeds = torch.cat(
                self.model.get_image_features(lvr_tokens, lvr_tokens_thw), dim=0
            )
            if selected_lvr_embeds.shape[0] != seq_positions.numel():
                raise ValueError(
                    "Latent and target feature counts differ: "
                    f"tokens={seq_positions.numel()}, features={selected_lvr_embeds.shape[0]}"
                )
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        prefill_compiled = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_eager = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if prefill_compiled or prefill_eager or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    loss_ce = None
    loss_lvr = None
    loss_mode_switch = None

    if labels is not None:
        if selected_lvr_embeds is None or batch_indices is None:
            raise ValueError("Training labels require auxiliary-image targets")
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_labels = shift_labels.masked_fill(
            shift_labels == self.config.lvr_id, IGNORE_INDEX
        ).to(shift_logits.device)
        loss_ce = CrossEntropyLoss()(shift_logits, shift_labels)

        predicted_latents = hidden_states[batch_indices, seq_positions - 1].float()
        target_latents = selected_lvr_embeds.to(predicted_latents.device).float()
        loss_lvr = set_lvr_loss_fct(self.config.loss_lvr_fct)(
            predicted_latents, target_latents
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state=outputs.last_hidden_state[:, -1, :],
    )
