#!/bin/bash
set -euo pipefail

# Colab normally exposes one accelerator. Accumulating 16 micro-batches keeps
# the same effective batch size as the default 2-GPU x 8 configuration.
export NUM_GPUS="${NUM_GPUS:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

# A full-parameter AdamW checkpoint includes large optimizer states. Keep one
# resumable checkpoint and write it less often to protect Colab local storage.
export EVAL_STEPS="${EVAL_STEPS:-250}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

bash scripts/finetune_lvr_frozenlake_3b.sh
