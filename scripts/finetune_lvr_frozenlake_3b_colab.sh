#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Colab normally exposes one accelerator. Accumulating 16 micro-batches keeps
# the same effective batch size as the default 2-GPU x 8 configuration.
export NUM_GPUS="${NUM_GPUS:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

# Full-parameter AdamW is killed by the 83 GiB Colab host when its CPU optimizer
# states are materialized. Adapt the language path with LoRA. ZeRO-2 keeps the
# small optimizer on the A100 and avoids CPUAdam's first-step RAM spike. This is
# a resource-constrained variant; the paper's canonical setup updates the LLM.
export LORA_ENABLE="${LORA_ENABLE:-True}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export FREEZE_LLM="${FREEZE_LLM:-True}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

# Keep one resumable checkpoint and write it less often to protect Colab disk.
export EVAL_STEPS="${EVAL_STEPS:-250}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# The Colab notebook intentionally removes FlashAttention because binary wheels
# are frequently incompatible with Colab's current PyTorch ABI.
export DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"

bash "$REPO_ROOT/scripts/finetune_lvr_frozenlake_3b.sh"
