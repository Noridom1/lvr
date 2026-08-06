#!/usr/bin/env python3
"""Evaluate paper-aligned FrozenLake LVR with fixed latent budgets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

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
PAPER_ALIGNED_OBJECTIVE = "paper_aligned_fixed_steps_v2"
DEFAULT_BUDGET_SWEEP = (4, 8, 16, 32, 64, 128, 256, 512)


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
    parser.add_argument("--lvr-steps", type=int, default=16)
    parser.add_argument(
        "--sweep-lvr-steps",
        nargs="+",
        type=int,
        help=(
            "Validation-only fixed-budget sweep. Recommended values: "
            + " ".join(str(value) for value in DEFAULT_BUDGET_SWEEP)
        ),
    )
    parser.add_argument("--max-action-tokens", type=int, default=64)
    parser.add_argument(
        "--attention",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
    )
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


def summarize_latent_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "latent_started": trace.get("latent_started", False),
        "latent_start_generated_index": trace.get("latent_start_generated_index"),
        "latent_steps": trace.get("latent_steps", 0),
        "latent_exit_reason": trace.get("latent_exit_reason", "missing_diagnostics"),
        "action_start_generated_index": trace.get("action_start_generated_index"),
        "action_token_count": trace.get("action_token_count", 0),
        "generated_token_count": trace.get("generated_token_count", 0),
        "transition_token_id": trace.get("transition_token_id"),
    }


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
        dr, dc = DELTAS[action]
        row, col = row + dr, col + dc
        if not (0 <= row < size and 0 <= col < size):
            return False, "left_board"
        if layout[row][col] == "H":
            return False, "hit_obstacle"
    success = row * size + col == goal
    return success, "goal" if success else "not_at_goal"


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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    return processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
    )


def validate_checkpoint_config(config, checkpoint: str) -> None:
    if getattr(config, "latent_end_token", False):
        raise ValueError(
            f"{checkpoint} uses the deprecated FrozenLake latent-end objective; "
            "retrain from the base model"
        )
    if getattr(config, "frozenlake_objective", None) != PAPER_ALIGNED_OBJECTIVE:
        raise ValueError(
            f"{checkpoint} is not marked as a paper-aligned FrozenLake checkpoint; "
            "refusing to mix incompatible objectives"
        )


def load_checkpoint(checkpoint: str, attention: str):
    checkpoint_path = Path(checkpoint)
    if not (checkpoint_path / "adapter_config.json").is_file():
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        validate_checkpoint_config(config, checkpoint)
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
    validate_checkpoint_config(task_config, checkpoint)
    base_config = AutoConfig.from_pretrained(
        adapter_config.base_model_name_or_path, trust_remote_code=True
    )
    for name in (
        "latent_end_token",
        "lvr_head",
        "lvr_head_type",
        "loss_lvr_fct",
        "lvr_id",
        "lvr_start_id",
        "lvr_end_id",
        "frozenlake_objective",
    ):
        if hasattr(task_config, name):
            setattr(base_config, name, getattr(task_config, name))

    model = QwenWithLVR.from_pretrained(
        adapter_config.base_model_name_or_path,
        config=base_config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=attention,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    if model.config.vocab_size < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload().eval()
    return model, processor


def evaluate_budget(
    model,
    processor,
    records: list[dict[str, Any]],
    image_folder: Path,
    output_dir: Path,
    lvr_steps: int,
    max_action_tokens: int,
    checkpoint: str,
    data_path: str,
) -> dict[str, Any]:
    if lvr_steps <= 0:
        raise ValueError("--lvr-steps values must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "predictions.jsonl"
    counts = {
        "total": 0,
        "exact_match": 0,
        "valid_format": 0,
        "goal_success": 0,
        "shortest_path_success": 0,
        "latent_started": 0,
        "latent_fixed_budget_exit": 0,
        "latent_not_started": 0,
    }
    latent_step_total = 0

    with result_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in tqdm(records, desc=f"FrozenLake LVR steps={lvr_steps}"):
            inputs = prepare_inputs(processor, record, image_folder).to(model.device)
            prompt_length = inputs.input_ids.shape[1]
            diagnostics: dict[str, Any] = {}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=lvr_steps + max_action_tokens + 2,
                    decoding_strategy="steps",
                    lvr_steps=[lvr_steps],
                    lvr_diagnostics=diagnostics,
                )
            sequence = generated.sequences[0] if hasattr(generated, "sequences") else generated[0]
            trace = diagnostics.get("sequences", [{}])[0]
            latent = summarize_latent_trace(trace)
            action_start = latent["action_start_generated_index"]
            action_ids = (
                sequence[prompt_length + action_start :]
                if action_start is not None
                else sequence[0:0]
            )
            action_output = processor.decode(
                action_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if latent["latent_exit_reason"] == "fixed_budget":
                predicted, valid_format = extract_actions(action_output)
            else:
                predicted, valid_format = [], False

            expected = record["actions"]
            goal_success, terminal_reason = simulate(record["source"]["layout"], predicted)
            exact_match = predicted == expected
            shortest_success = goal_success and len(predicted) == len(expected)
            counts["total"] += 1
            counts["exact_match"] += int(exact_match)
            counts["valid_format"] += int(valid_format)
            counts["goal_success"] += int(goal_success)
            counts["shortest_path_success"] += int(shortest_success)
            counts["latent_started"] += int(latent["latent_started"])
            counts["latent_fixed_budget_exit"] += int(
                latent["latent_exit_reason"] == "fixed_budget"
            )
            counts["latent_not_started"] += int(not latent["latent_started"])
            latent_step_total += latent["latent_steps"]

            transition_token = None
            if latent["transition_token_id"] is not None:
                transition_token = processor.decode(
                    [latent["transition_token_id"]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            row = {
                "id": record["id"],
                "expected_actions": expected,
                "predicted_actions": predicted,
                "action_output": action_output,
                **latent,
                "transition_token": transition_token,
                "transition_is_lvr_end": (
                    latent["transition_token_id"] == getattr(model.config, "lvr_end_id", None)
                ),
                "lvr_steps_budget": lvr_steps,
                "valid_format": valid_format,
                "exact_match": exact_match,
                "goal_success": goal_success,
                "shortest_path_success": shortest_success,
                "terminal_reason": terminal_reason,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            tqdm.write(
                f"{record['id']}: exit={latent['latent_exit_reason']} "
                f"steps={latent['latent_steps']} actions={' '.join(predicted) or '<none>'}"
            )

    total = counts["total"]
    summary = {
        **counts,
        "exact_match_accuracy": counts["exact_match"] / total if total else 0.0,
        "valid_format_rate": counts["valid_format"] / total if total else 0.0,
        "goal_success_rate": counts["goal_success"] / total if total else 0.0,
        "shortest_path_success_rate": (
            counts["shortest_path_success"] / total if total else 0.0
        ),
        "lvr_activation_rate": counts["latent_started"] / total if total else 0.0,
        "fixed_budget_exit_rate": (
            counts["latent_fixed_budget_exit"] / total if total else 0.0
        ),
        "average_latent_steps": latent_step_total / total if total else 0.0,
        "checkpoint": checkpoint,
        "data_path": data_path,
        "lvr_steps": lvr_steps,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    return summary


def budget_selection_key(summary: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        summary["goal_success_rate"],
        summary["shortest_path_success_rate"],
        summary["exact_match_accuracy"],
        -summary["lvr_steps"],
    )


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
    elif args.max_samples is not None:
        records = records[: args.max_samples]

    replace_qwen2_5_with_frozenlake_forward()
    model, processor = load_checkpoint(args.checkpoint, args.attention)
    output_dir = Path(args.output_dir)
    image_folder = Path(args.image_folder)

    if args.sweep_lvr_steps:
        budgets = sorted(set(args.sweep_lvr_steps))
        if any(value <= 0 for value in budgets):
            raise ValueError("Every --sweep-lvr-steps value must be positive")
        summaries = [
            evaluate_budget(
                model,
                processor,
                records,
                image_folder,
                output_dir / f"steps-{budget}",
                budget,
                args.max_action_tokens,
                args.checkpoint,
                args.data_path,
            )
            for budget in budgets
        ]
        selected = max(summaries, key=budget_selection_key)
        sweep = {
            "selection_order": [
                "goal_success_rate",
                "shortest_path_success_rate",
                "exact_match_accuracy",
                "smaller_lvr_steps",
            ],
            "selected_lvr_steps": selected["lvr_steps"],
            "results": summaries,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "budget_sweep.json").open("w", encoding="utf-8") as output:
            json.dump(sweep, output, indent=2)
            output.write("\n")
        print(json.dumps(sweep, indent=2))
    else:
        summary = evaluate_budget(
            model,
            processor,
            records,
            image_folder,
            output_dir,
            args.lvr_steps,
            args.max_action_tokens,
            args.checkpoint,
            args.data_path,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
