#!/usr/bin/env python3
"""Evaluate a FrozenLake LVR checkpoint on action-only navigation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

# Make the repository's top-level ``src`` package importable when this file is
# launched directly as ``python evaluation/evaluate_frozenlake_lvr.py``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_frozenlake import replace_qwen2_5_with_frozenlake_forward


ACTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}
DELTAS = {
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "UP": (-1, 0),
    "DOWN": (1, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/frozenlake/test.jsonl")
    parser.add_argument("--image-folder", default="training_samples/frozenlake")
    parser.add_argument("--output-dir", default="frozenlake_evaluation")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Evaluate only this zero-based record index from the selected JSONL split.",
    )
    parser.add_argument("--max-lvr-steps", type=int, default=2048)
    parser.add_argument("--lvr-end-threshold", type=float, default=0.02)
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract_actions(text: str) -> tuple[list[str], bool]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    answer_text = match.group(1) if match else text
    raw_tokens = answer_text.upper().strip().split()
    actions = [token for token in raw_tokens if token in ACTIONS]
    valid_format = bool(match) and len(actions) == len(raw_tokens) and bool(actions)
    return actions, valid_format


def simulate(layout: list[list[str]], actions: list[str]) -> tuple[bool, str]:
    size = len(layout)
    start = next(
        (row * size + col for row in range(size) for col in range(size) if layout[row][col] == "S"),
        None,
    )
    goal = next(
        (row * size + col for row in range(size) for col in range(size) if layout[row][col] == "G"),
        None,
    )
    if start is None or goal is None:
        return False, "invalid_layout"
    row, col = divmod(start, size)
    for action in actions:
        if action not in DELTAS:
            return False, "invalid_action"
        dr, dc = DELTAS[action]
        row, col = row + dr, col + dc
        if not (0 <= row < size and 0 <= col < size):
            return False, "left_board"
        if layout[row][col] == "H":
            return False, "hit_obstacle"
    return row * size + col == goal, "goal" if row * size + col == goal else "not_at_goal"


def prepare_inputs(processor, record: dict[str, Any], image_folder: Path):
    image_path = Path(record["initial_image"])
    if not image_path.is_absolute():
        image_path = image_folder / image_path
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": record["instruction"]},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages)
    return processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
    )


def load_checkpoint(checkpoint: str, attention: str):
    checkpoint_path = Path(checkpoint)
    adapter_path = checkpoint_path / "adapter_config.json"
    if not adapter_path.is_file():
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        model = QwenWithLVR.from_pretrained(
            checkpoint,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=attention,
            device_map="auto",
        ).eval()
        return model, AutoProcessor.from_pretrained(checkpoint)

    from peft import PeftConfig, PeftModel

    adapter_config = PeftConfig.from_pretrained(checkpoint)
    task_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    base_config = AutoConfig.from_pretrained(
        adapter_config.base_model_name_or_path, trust_remote_code=True
    )
    for name in (
        "latent_end_token",
        "lvr_head",
        "lvr_head_type",
        "loss_lvr_fct",
        "loss_mode_switch_fct",
        "lvr_id",
        "lvr_latent_end_id",
        "lvr_start_id",
        "lvr_end_id",
    ):
        if hasattr(task_config, name):
            setattr(base_config, name, getattr(task_config, name))

    model, loading_info = QwenWithLVR.from_pretrained(
        adapter_config.base_model_name_or_path,
        config=base_config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=attention,
        device_map="auto",
        output_loading_info=True,
    )
    if "lvr_latent_end_emb" in loading_info["missing_keys"]:
        model.reset_lvr_latent_end_emb()

    processor = AutoProcessor.from_pretrained(checkpoint)
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    latent_path = checkpoint_path / "non_lora_state_dict.bin"
    if not latent_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint is missing {latent_path.name}")
    latent_state = torch.load(latent_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(latent_state, strict=False)
    if unexpected or "lvr_latent_end_emb" not in latent_state:
        raise ValueError("Invalid FrozenLake non-LoRA trainables file")

    # Merge only for inference. This returns the original QwenWithLVR class, so
    # its custom latent decoding method remains directly available.
    model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload().eval()
    return model, processor


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FrozenLake evaluation requires CUDA")
    records = load_jsonl(args.data_path)
    if args.sample_index is not None:
        if args.max_samples is not None:
            raise ValueError("--sample-index and --max-samples cannot be used together")
        if not 0 <= args.sample_index < len(records):
            raise IndexError(
                f"sample index {args.sample_index} is outside [0, {len(records) - 1}]"
            )
        records = [records[args.sample_index]]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    replace_qwen2_5_with_frozenlake_forward()
    model, processor = load_checkpoint(args.checkpoint, args.attention)
    if not getattr(model.config, "latent_end_token", False):
        raise ValueError("Checkpoint was not trained with latent_end_token")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "predictions.jsonl"
    image_folder = Path(args.image_folder)
    counts = {
        "total": 0,
        "exact_match": 0,
        "valid_format": 0,
        "goal_success": 0,
        "shortest_path_success": 0,
    }

    with result_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in tqdm(records, desc="FrozenLake evaluation"):
            inputs = prepare_inputs(processor, record, image_folder).to(model.device)
            prompt_length = inputs.input_ids.shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_lvr_steps + args.max_action_tokens,
                    decoding_strategy="latent",
                    criterion="mse",
                    lvr_end_threshold=args.lvr_end_threshold,
                    lvr_steps=[args.max_lvr_steps],
                )
            decoded = processor.decode(
                generated[0, prompt_length:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            predicted, valid_format = extract_actions(decoded)
            expected = record["actions"]
            goal_success, terminal_reason = simulate(record["source"]["layout"], predicted)
            exact_match = predicted == expected
            shortest_success = goal_success and len(predicted) == len(expected)
            counts["total"] += 1
            counts["exact_match"] += int(exact_match)
            counts["valid_format"] += int(valid_format)
            counts["goal_success"] += int(goal_success)
            counts["shortest_path_success"] += int(shortest_success)
            output.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "expected_actions": expected,
                        "predicted_actions": predicted,
                        "raw_output": decoded,
                        "valid_format": valid_format,
                        "exact_match": exact_match,
                        "goal_success": goal_success,
                        "shortest_path_success": shortest_success,
                        "terminal_reason": terminal_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = counts["total"]
    summary = {
        **counts,
        "exact_match_accuracy": counts["exact_match"] / total if total else 0.0,
        "valid_format_rate": counts["valid_format"] / total if total else 0.0,
        "goal_success_rate": counts["goal_success"] / total if total else 0.0,
        "shortest_path_success_rate": counts["shortest_path_success"] / total if total else 0.0,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "lvr_end_threshold": args.lvr_end_threshold,
        "max_lvr_steps": args.max_lvr_steps,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
