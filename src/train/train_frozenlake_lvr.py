"""Train Qwen2.5-VL LVR on FrozenLake successor-state trajectories."""

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional

# DeepSpeed launches this file by path, which otherwise exposes ``src/train``
# rather than the repository root on Python's module search path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoProcessor, HfArgumentParser

from src.frozenlake_lvr_dataset import make_frozenlake_lvr_data_module
from src.model.qwen_lvr_model import QwenWithLVR
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.monkey_patch_forward_frozenlake import replace_qwen2_5_with_frozenlake_forward
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)
from src.trainer import QwenLVRSFTTrainer


@dataclass
class FrozenLakeDataArguments(DataArguments):
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Optional FrozenLake validation JSONL."}
    )


def _set_requires_grad(parameters, value: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = value


def _find_lora_target_modules(model) -> list[str]:
    """Adapt the language path, including token input/output projections."""
    targets = []
    for name, module in model.named_modules():
        if "visual" in name:
            continue
        if isinstance(module, (nn.Linear, nn.Embedding)):
            targets.append(name)
    if not targets:
        raise ValueError("No language modules were found for LoRA")
    return targets


def _load_non_lora_trainables(model, checkpoint: str) -> None:
    state_path = Path(checkpoint) / "non_lora_state_dict.bin"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"LoRA checkpoint is missing its latent trainables: {state_path}"
        )
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected non-LoRA checkpoint keys: {unexpected}")
    loaded = set(state_dict)
    if not any(name.endswith("lvr_latent_end_emb") for name in loaded):
        raise ValueError("LoRA checkpoint does not contain lvr_latent_end_emb")


def train() -> None:
    parser = HfArgumentParser((ModelArguments, FrozenLakeDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "Qwen2.5" not in model_args.model_id:
        raise ValueError("FrozenLake LVR currently supports Qwen2.5-VL models")
    if not model_args.latent_end_token:
        raise ValueError("Variable-length FrozenLake trajectories require --latent_end_token True")
    if model_args.lvr_head:
        raise ValueError("The FrozenLake trajectory baseline requires --lvr_head False")
    if training_args.enable_data_packing:
        raise ValueError("FrozenLake auxiliary-image batches do not support data packing")
    if training_args.online_checkpoint:
        raise ValueError("Use a local output directory for the FrozenLake entry point")
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("LoRA requires --freeze_llm True so the base LLM stays frozen")
    if training_args.vision_lora:
        raise ValueError("FrozenLake LoRA currently keeps the vision tower frozen")

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16
        if training_args.bf16
        else torch.float32
    )
    # A PEFT checkpoint stores adapters rather than a second copy of Qwen, so
    # resume always reconstructs the base model from model_id first.
    model_path = (
        model_args.model_id
        if training_args.lora_enable
        else training_args.checkpoint_name or model_args.model_id
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.latent_end_token = True
    config.lvr_head = False
    config.lvr_head_type = model_args.lvr_head_type
    config.loss_lvr_fct = training_args.loss_lvr_fct
    config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

    replace_qwen2_5_with_frozenlake_forward()
    model, loading_info = QwenWithLVR.from_pretrained(
        model_path,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
        ),
        output_loading_info=True,
    )
    # The base Qwen checkpoint has no learned latent-end parameter.  Initialize
    # it explicitly after meta-device checkpoint loading; preserve it on resume.
    if "lvr_latent_end_emb" in loading_info["missing_keys"]:
        model.reset_lvr_latent_end_emb()
    if not model.lvr_latent_end_is_finite():
        raise FloatingPointError("lvr_latent_end_emb is non-finite immediately after loading")
    replace_qwen_2_5_vl_patch_emb()
    model.config.use_cache = False

    # Match the original Stage-1 policy: optimize the language model while the
    # visual encoder and merger provide stable input/target embeddings.
    _set_requires_grad(model.model.parameters(), not training_args.freeze_llm)
    _set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    _set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)
    _set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)
    model.lvr_latent_end_emb.requires_grad = True
    model.visual.to(dtype=compute_dtype, device=training_args.device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # On resume, load the tokenizer from the checkpoint so the LVR token ids
    # cannot silently drift from the resized embedding table.
    processor_path = training_args.checkpoint_name or model_path
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    for token in (
        "<|lvr_start|>",
        "<|lvr|>",
        "<|lvr_latent_end|>",
        "<|lvr_end|>",
    ):
        processor.tokenizer.add_tokens(token, special_tokens=True)
    model.config.lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    model.config.lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids(
        "<|lvr_latent_end|>"
    )
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.lora_enable:
        if training_args.checkpoint_name:
            _load_non_lora_trainables(model, training_args.checkpoint_name)
            model = PeftModel.from_pretrained(
                model,
                training_args.checkpoint_name,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=_find_lora_target_modules(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
            )
            model = get_peft_model(model, peft_config)

        # PEFT freezes every non-adapter parameter. The learned stopping target
        # is deliberately the one small non-LoRA exception.
        base_model = model.get_base_model()
        base_model.lvr_latent_end_emb.requires_grad = True
        model.print_trainable_parameters()

    data_module = make_frozenlake_lvr_data_module(processor, data_args)
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )
    # The upstream trainer initializes these only for OCI runs but reads
    # temp_folder during every checkpoint save.
    trainer.temp_folder = None
    trainer.oci_handler = None
    trainer.train(resume_from_checkpoint=training_args.checkpoint_name)
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        lora_state = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        saved_non_lora_state = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        non_lora_state = {
            name.rsplit(".", 1)[-1]: value
            for name, value in saved_non_lora_state.items()
            if name.endswith("lvr_latent_end_emb")
        }
        if set(non_lora_state) != {"lvr_latent_end_emb"}:
            raise ValueError("Could not collect the trainable latent-end vector")
        if trainer.is_world_process_zero():
            model.get_base_model().config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=lora_state)
            torch.save(
                non_lora_state,
                Path(training_args.output_dir) / "non_lora_state_dict.bin",
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        # Preserve the four added LVR tokens with the final standalone model.
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
