#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_PATH="${DATA_PATH:-data/frozenlake/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/frozenlake/validation.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-training_samples/frozenlake}"
OUTPUT_DIR="${OUTPUT_DIR:-frozenlake_checkpoints/qwen2.5-vl-3b}"
RUN_NAME="${RUN_NAME:-FrozenLake-LVR-Qwen2.5-VL-3B}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"

NUM_GPUS="${NUM_GPUS:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:--1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-False}"

# A 256x256 state is about 81 merged visual tokens. The dataset averages 3.2
# successor states, or approximately 259 supervised latent tokens per sample.
MIN_VISUAL_TOKENS="${MIN_VISUAL_TOKENS:-64}"
MAX_VISUAL_TOKENS="${MAX_VISUAL_TOKENS:-256}"

EXTRA_ARGS=()
if [[ -n "$CHECKPOINT_NAME" ]]; then
    EXTRA_ARGS+=(--checkpoint_name "$CHECKPOINT_NAME")
fi

deepspeed --num_gpus "$NUM_GPUS" src/train/train_frozenlake_lvr.py \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --coconut True \
    --latent_end_token True \
    --lvr_head False \
    --mode_switch_loss True \
    --loss_lvr_fct cosine \
    --loss_lvr_lambda 0.1 \
    --loss_mode_switch_fct mse \
    --loss_mode_switch_lambda 0.1 \
    --deepspeed scripts/zero3_offload.json \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 "$DISABLE_FLASH_ATTN2" \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --enable_data_packing False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --image_min_pixels $((MIN_VISUAL_TOKENS * 28 * 28)) \
    --image_max_pixels $((MAX_VISUAL_TOKENS * 28 * 28)) \
    --max_seq_length 4096 \
    --eval_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps 1 \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --report_to none \
    "${EXTRA_ARGS[@]}"
