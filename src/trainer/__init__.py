"""Trainer exports loaded on demand.

Keeping these imports lazy prevents unrelated SFT/GRPO dependencies and
machine-specific paths from being imported by the FrozenLake LVR entry point.
"""

from typing import Any


__all__ = ["QwenSFTTrainer", "QwenLVRSFTTrainer", "QwenGRPOTrainer"]


def __getattr__(name: str) -> Any:
    if name == "QwenSFTTrainer":
        from .sft_trainer import QwenSFTTrainer

        return QwenSFTTrainer
    if name == "QwenLVRSFTTrainer":
        from .lvr_trainer import QwenLVRSFTTrainer

        return QwenLVRSFTTrainer
    if name == "QwenGRPOTrainer":
        from .grpo_trainer import QwenGRPOTrainer

        return QwenGRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
