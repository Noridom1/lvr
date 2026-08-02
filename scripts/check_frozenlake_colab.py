#!/usr/bin/env python3
"""Fail-fast environment and storage checks for the Colab FrozenLake run."""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
import torch
import transformers


EXPECTED_TRANSFORMERS = "4.54.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser.parse_args()


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        errors.append(
            f"transformers must be {EXPECTED_TRANSFORMERS}; found {transformers.__version__}"
        )
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLModel,
        )

        if not hasattr(Qwen2_5_VLModel, "get_image_features"):
            errors.append("Qwen2_5_VLModel.get_image_features is unavailable")
    except Exception as exc:
        errors.append(
            "Qwen2.5-VL could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    if package_version("deepspeed") is None:
        errors.append("deepspeed is not installed")
    if package_version("qwen-vl-utils") is None:
        errors.append("qwen-vl-utils is not installed")
    if args.attention == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            errors.append(
                "flash-attn cannot be imported "
                f"({type(exc).__name__}: {exc}); rerun with --attention sdpa"
            )

    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        gpu_name = None
        gpu_memory_gib = 0.0
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if not torch.cuda.is_bf16_supported():
            errors.append("the assigned GPU does not support BF16")
        if gpu_memory_gib < 39:
            warnings.append(f"only {gpu_memory_gib:.1f} GiB GPU memory is visible")

    memory = psutil.virtual_memory()
    ram_total_gib = memory.total / 1024**3
    ram_free_gib = memory.available / 1024**3
    disk_free_gib = shutil.disk_usage(workspace).free / 1024**3
    if ram_total_gib < 75:
        errors.append(f"full CPU-offload training expects about 75+ GiB RAM; found {ram_total_gib:.1f}")
    if ram_free_gib < 70:
        warnings.append(f"only {ram_free_gib:.1f} GiB RAM is currently free")
    if disk_free_gib < 100:
        warnings.append(f"only {disk_free_gib:.1f} GiB local disk is free")

    required_paths = (
        "data/frozenlake/train.jsonl",
        "data/frozenlake/validation.jsonl",
        "data/frozenlake/test.jsonl",
        "training_samples/frozenlake/00000000/frame_000.png",
        "scripts/zero3_offload.json",
    )
    missing_paths = [path for path in required_paths if not (workspace / path).is_file()]
    if missing_paths:
        errors.append(f"missing required paths: {missing_paths}")

    result = {
        "status": "error" if errors else "ok",
        "gpu": gpu_name,
        "gpu_memory_gib": round(gpu_memory_gib, 1),
        "ram_total_gib": round(ram_total_gib, 1),
        "ram_free_gib": round(ram_free_gib, 1),
        "disk_free_gib": round(disk_free_gib, 1),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "deepspeed": package_version("deepspeed"),
        "qwen_vl_utils": package_version("qwen-vl-utils"),
        "attention": args.attention,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
