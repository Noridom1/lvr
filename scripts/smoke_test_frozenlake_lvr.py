#!/usr/bin/env python3
"""Run one FrozenLake LVR batch through Qwen2.5-VL without updating weights.

Run this on the GPU training machine before starting DeepSpeed.  The command
checks the processor, latent/image cardinality, custom forward path, and the
paper-aligned CE plus MSE losses on one real trajectory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

# Support both ``python -m scripts.smoke_test_frozenlake_lvr`` and direct
# ``python scripts/smoke_test_frozenlake_lvr.py`` execution from the repo.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from transformers import AutoConfig, AutoProcessor

from src.frozenlake_lvr_dataset import FrozenLakeLVRDataset, FrozenLakeLVRCollator
from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_frozenlake import replace_qwen2_5_with_frozenlake_forward
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb


LVR_TOKENS = (
    "<|lvr_start|>",
    "<|lvr|>",
    "<|lvr_end|>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data-path", default="data/frozenlake/train.jsonl")
    parser.add_argument("--image-folder", default="training_samples/frozenlake")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--min-visual-tokens", type=int, default=64)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke check requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured smoke check requires BF16 support")

    min_pixels = args.min_visual_tokens * 28 * 28
    max_pixels = args.max_visual_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=min_pixels, max_pixels=max_pixels
    )
    for token in LVR_TOKENS:
        processor.tokenizer.add_tokens(token, special_tokens=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.latent_end_token = False
    config.lvr_head = False
    config.lvr_head_type = "simple"
    config.loss_lvr_fct = "mse"
    config.frozenlake_objective = "paper_aligned_fixed_steps_v2"
    replace_qwen2_5_with_frozenlake_forward()
    model = QwenWithLVR.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention,
    )
    model = model.cuda()
    replace_qwen_2_5_vl_patch_emb()
    if hasattr(model, "lvr_latent_end_emb"):
        raise AssertionError("Paper-aligned FrozenLake unexpectedly initialized latent-end state")

    model.config.lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")
    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_end|>")
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))

    data_args = SimpleNamespace(
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
        image_resized_width=None,
        image_resized_height=None,
    )
    dataset = FrozenLakeLVRDataset(
        args.data_path, args.image_folder, processor, data_args
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}]")
    example = dataset[args.sample_index]
    batch = FrozenLakeLVRCollator(processor.tokenizer.pad_token_id)([example])
    batch = {key: value.cuda() for key, value in batch.items()}

    latent_positions = int((batch["input_ids"] == model.config.lvr_id).sum().item())
    target_features = int(
        (
            batch["lvr_tokens_thw"].long().prod(dim=-1)
            // processor.image_processor.merge_size**2
        ).sum().item()
    )
    if latent_positions != target_features:
        raise AssertionError(
            f"latent positions ({latent_positions}) != target features ({target_features})"
        )

    model.eval()
    with torch.no_grad():
        outputs = model(**batch, return_dict=True)
    losses = {
        "loss_ce": float(outputs.loss_ce),
        "loss_lvr": float(outputs.loss_lvr),
    }
    if outputs.loss_mode_switch is not None:
        raise AssertionError("Paper-aligned FrozenLake produced a mode-switch loss")
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError(f"Non-finite loss detected: {losses}")

    print(
        {
            "status": "ok",
            "sample_index": args.sample_index,
            "sequence_tokens": int(batch["attention_mask"].sum().item()),
            "latent_positions": latent_positions,
            "target_images": int(batch["lvr_tokens_thw"].shape[0]),
            "loss_mode_switch": None,
            **losses,
        }
    )


if __name__ == "__main__":
    main()
