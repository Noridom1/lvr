"""FrozenLake trajectory batches for latent visual reasoning SFT.

The user sees only the initial frame. Ordered successor frames are processed as
separate visual targets, while the assistant text contains only navigation
actions inside the repository's answer control tags.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
    LVR_END_TOKEN,
    LVR_START_TOKEN,
    LVR_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from src.dataset.data_utils import get_image_info, pad_sequence


def load_json_records(data_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL manifest."""
    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def visual_token_count(image_grid_thw: torch.Tensor, merge_size: int) -> int:
    """Count Qwen visual features after spatial merging."""
    if image_grid_thw.ndim != 2 or image_grid_thw.size(-1) != 3:
        raise ValueError(f"Expected image_grid_thw [N, 3], got {tuple(image_grid_thw.shape)}")
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    patch_counts = image_grid_thw.to(torch.long).prod(dim=-1)
    divisor = merge_size**2
    if torch.any(patch_counts % divisor != 0):
        raise ValueError("Image grid is not divisible by the spatial merge size")
    return int((patch_counts // divisor).sum().item())


def build_lvr_response(answer: str, latent_tokens: int) -> str:
    if latent_tokens <= 0:
        raise ValueError("A trajectory requires at least one latent visual token")
    latent = LVR_START_TOKEN + (LVR_TOKEN * latent_tokens) + LVR_END_TOKEN
    return f"{latent}\n<answer>{answer}</answer>"


def build_frozenlake_prompt_text(instruction: str) -> str:
    """Build the exact system/user/assistant prefix used by FrozenLake SFT."""
    user_content = (
        f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}\n{instruction}"
    )
    user_text = (
        f"{DEFAULT_IM_START_TOKEN}user\n{user_content}{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_IM_START_TOKEN}assistant\n"
    )
    if not SYSTEM_MESSAGE:
        return user_text
    system_text = (
        f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
    )
    return system_text + user_text


def validate_frozenlake_record(record: dict[str, Any]) -> None:
    required = {"id", "task", "initial_image", "aux_images", "final_image", "actions", "answer", "instruction"}
    missing = required - record.keys()
    sample_id = record.get("id", "<unknown>")
    if missing:
        raise ValueError(f"{sample_id}: missing fields {sorted(missing)}")
    if record["task"] != "frozenlake":
        raise ValueError(f"{sample_id}: task is not frozenlake")
    if not record["aux_images"]:
        raise ValueError(f"{sample_id}: aux_images is empty")
    if len(record["aux_images"]) != len(record["actions"]):
        raise ValueError(f"{sample_id}: actions and aux_images are not aligned")
    if record["answer"] != " ".join(record["actions"]):
        raise ValueError(f"{sample_id}: answer does not match actions")
    if record["final_image"] != record["aux_images"][-1]:
        raise ValueError(f"{sample_id}: final_image is not the last auxiliary image")
    if any(action not in {"LEFT", "RIGHT", "UP", "DOWN"} for action in record["actions"]):
        raise ValueError(f"{sample_id}: unsupported action")


class FrozenLakeLVRDataset(Dataset):
    def __init__(self, data_path: str, image_folder: str, processor, data_args):
        if not image_folder:
            raise ValueError("FrozenLake training requires --image_folder")
        self.records = load_json_records(data_path)
        self.image_folder = Path(image_folder)
        self.processor = processor
        self.image_min_pixels = data_args.image_min_pixels
        self.image_max_pixels = data_args.image_max_pixels
        self.image_resized_width = data_args.image_resized_width
        self.image_resized_height = data_args.image_resized_height

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, relative_path: str):
        path = Path(relative_path)
        if not path.is_absolute():
            path = self.image_folder / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return get_image_info(
            str(path),
            self.image_min_pixels,
            self.image_max_pixels,
            self.image_resized_width,
            self.image_resized_height,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        validate_frozenlake_record(record)
        initial_image = self._load_image(record["initial_image"])
        aux_images = [self._load_image(path) for path in record["aux_images"]]

        # These placeholders only let the Qwen processor batch all target images;
        # the resulting token ids are discarded and never enter the user prompt.
        target_placeholders = (VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN) * len(aux_images)
        target_inputs = self.processor(
            text=[target_placeholders],
            images=aux_images,
            videos=None,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )
        lvr_tokens = target_inputs["pixel_values"]
        lvr_tokens_thw = target_inputs["image_grid_thw"]
        merge_size = int(self.processor.image_processor.merge_size)
        latent_count = visual_token_count(lvr_tokens_thw, merge_size)

        prompt_text = build_frozenlake_prompt_text(record["instruction"])
        prompt = self.processor(
            text=[prompt_text],
            images=[initial_image],
            videos=None,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )
        response_text = f"{build_lvr_response(record['answer'], latent_count)}{DEFAULT_IM_END_TOKEN}\n"
        response_ids = self.processor.tokenizer(
            response_text, add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"].squeeze(0)

        prompt_ids = prompt["input_ids"].squeeze(0)
        input_parts = [prompt_ids, response_ids]
        label_parts = [torch.full_like(prompt_ids, IGNORE_INDEX), response_ids]
        input_ids = torch.cat(input_parts).to(torch.long)
        labels = torch.cat(label_parts).to(torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "pixel_values": prompt["pixel_values"],
            "image_grid_thw": prompt["image_grid_thw"],
            "lvr_tokens": lvr_tokens,
            "lvr_tokens_thw": lvr_tokens_thw,
        }


class FrozenLakeLVRCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        input_ids = pad_sequence(
            [item["input_ids"] for item in examples], "right", self.pad_token_id
        )
        labels = pad_sequence(
            [item["labels"] for item in examples], "right", IGNORE_INDEX
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.pad_token_id,
            "labels": labels,
            "pixel_values": torch.cat([item["pixel_values"] for item in examples]),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in examples]),
            "lvr_tokens": torch.cat([item["lvr_tokens"] for item in examples]),
            "lvr_tokens_thw": torch.cat([item["lvr_tokens_thw"] for item in examples]),
        }


def make_frozenlake_lvr_data_module(processor, data_args):
    train_dataset = FrozenLakeLVRDataset(
        data_args.data_path, data_args.image_folder, processor, data_args
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = FrozenLakeLVRDataset(
            data_args.eval_data_path, data_args.image_folder, processor, data_args
        )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": FrozenLakeLVRCollator(processor.tokenizer.pad_token_id),
    }
