"""Dataset factory exports loaded only when requested."""

from typing import Any


__all__ = [
    "make_dpo_data_module",
    "make_supervised_data_module",
    "make_grpo_data_module",
    "make_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr_fixedToken",
]


def __getattr__(name: str) -> Any:
    if name == "make_dpo_data_module":
        from .dpo_dataset import make_dpo_data_module

        return make_dpo_data_module
    if name == "make_supervised_data_module":
        from .sft_dataset import make_supervised_data_module

        return make_supervised_data_module
    if name == "make_grpo_data_module":
        from .grpo_dataset import make_grpo_data_module

        return make_grpo_data_module
    if name == "make_supervised_data_module_lvr":
        from .lvr_sft_dataset import make_supervised_data_module_lvr

        return make_supervised_data_module_lvr
    if name == "make_packed_supervised_data_module_lvr":
        from .lvr_sft_dataset_packed import make_packed_supervised_data_module_lvr

        return make_packed_supervised_data_module_lvr
    if name == "make_packed_supervised_data_module_lvr_fixedToken":
        from .lvr_sft_dataset_packed_fixedToken import (
            make_packed_supervised_data_module_lvr_fixedToken,
        )

        return make_packed_supervised_data_module_lvr_fixedToken
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
