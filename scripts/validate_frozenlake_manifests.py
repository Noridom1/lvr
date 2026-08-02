#!/usr/bin/env python3
"""Validate formatted FrozenLake manifests without importing Torch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
ACTIONS = {"LEFT", "RIGHT", "UP", "DOWN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/frozenlake"))
    parser.add_argument(
        "--image-folder", type=Path, default=Path("training_samples/frozenlake")
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def expected_action(source: int, destination: int, level: int) -> str:
    source_row, source_col = divmod(source, level)
    destination_row, destination_col = divmod(destination, level)
    delta = destination_row - source_row, destination_col - source_col
    mapping = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }
    require(delta in mapping, f"non-adjacent state transition {source}->{destination}")
    return mapping[delta]


def validate_record(record: dict[str, Any], split: str, image_folder: Path) -> None:
    sample_id = record.get("id", "<unknown>")
    prefix = f"{sample_id}: "
    required = {
        "schema_version",
        "id",
        "task",
        "level",
        "initial_image",
        "aux_images",
        "final_image",
        "instruction",
        "actions",
        "answer",
        "trajectory",
        "source",
        "split",
    }
    require(not (required - record.keys()), prefix + "missing required field")
    require(record["task"] == "frozenlake", prefix + "wrong task")
    require(record["split"] == split, prefix + "wrong split field")

    actions = record["actions"]
    aux_images = record["aux_images"]
    require(actions and set(actions) <= ACTIONS, prefix + "invalid action sequence")
    require(len(aux_images) == len(actions), prefix + "actions/aux_images mismatch")
    require(record["answer"] == " ".join(actions), prefix + "answer mismatch")
    require(record["final_image"] == aux_images[-1], prefix + "final image mismatch")

    expected_names = [f"frame_{index:03d}.png" for index in range(len(actions) + 1)]
    paths = [record["initial_image"], *aux_images]
    require(
        [Path(path).name for path in paths] == expected_names,
        prefix + "non-contiguous frame mapping",
    )
    for relative_path in paths:
        require(not Path(relative_path).is_absolute(), prefix + "image path must be relative")
        require((image_folder / relative_path).is_file(), prefix + f"missing {relative_path}")

    level = record["level"]
    layout = record["source"]["layout"]
    require(isinstance(level, int) and level > 1, prefix + "invalid level")
    require(
        isinstance(layout, list)
        and len(layout) == level
        and all(isinstance(row, list) and len(row) == level for row in layout),
        prefix + "invalid layout",
    )
    states = record["trajectory"]["states"]
    require(len(states) == len(actions) + 1, prefix + "state count mismatch")
    require(
        record["trajectory"]["transition_count"] == len(actions),
        prefix + "transition count mismatch",
    )
    derived = [
        expected_action(source, destination, level)
        for source, destination in zip(states, states[1:])
    ]
    require(derived == actions, prefix + "actions do not follow trajectory states")
    require(layout[states[0] // level][states[0] % level] == "S", prefix + "bad start")
    require(layout[states[-1] // level][states[-1] % level] == "G", prefix + "bad goal")
    require(
        all(layout[state // level][state % level] != "H" for state in states),
        prefix + "trajectory enters a hole",
    )

    trace_path = record["source"]["trace"]
    require(not Path(trace_path).is_absolute(), prefix + "trace path must be relative")
    require((image_folder / trace_path).is_file(), prefix + f"missing {trace_path}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    image_folder = args.image_folder.resolve()
    info_path = data_dir / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)

    by_split: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    layout_owners: dict[tuple[int, str], str] = {}
    transition_counts: list[int] = []
    referenced_images: set[str] = set()
    for split in SPLITS:
        records = load_jsonl(data_dir / f"{split}.jsonl")
        by_split[split] = records
        require(len(records) == info["split_counts"][split], f"{split}: count mismatch")
        for record in records:
            validate_record(record, split, image_folder)
            sample_id = record["id"]
            require(sample_id not in all_ids, f"duplicate id {sample_id}")
            all_ids.add(sample_id)
            require(record["instruction"] == info["instruction"], f"{sample_id}: prompt drift")
            transition_counts.append(len(record["actions"]))
            referenced_images.update([record["initial_image"], *record["aux_images"]])

            flat_layout = "".join(
                cell for row in record["source"]["layout"] for cell in row
            )
            layout_key = record["level"], flat_layout
            previous_split = layout_owners.setdefault(layout_key, split)
            require(previous_split == split, f"layout leaked across {previous_split}/{split}")

    all_records = load_jsonl(data_dir / "all.jsonl")
    union_by_id = {
        record["id"]: record for split in SPLITS for record in by_split[split]
    }
    require(len(all_records) == info["sample_count"], "all.jsonl count mismatch")
    require(
        {record["id"]: record for record in all_records} == union_by_id,
        "all.jsonl does not equal the split union",
    )
    require(len(all_ids) == info["sample_count"], "unique sample count mismatch")
    require(sum(transition_counts) == info["transition_count"]["total"], "transition total mismatch")

    level_counts = Counter(record["level"] for record in all_records)
    require(
        {str(level): count for level, count in sorted(level_counts.items())}
        == info["level_counts"],
        "level counts mismatch",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(all_ids),
                "split_counts": {split: len(by_split[split]) for split in SPLITS},
                "transitions": sum(transition_counts),
                "referenced_images": len(referenced_images),
                "layout_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
