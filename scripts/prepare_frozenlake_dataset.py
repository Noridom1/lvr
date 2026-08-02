#!/usr/bin/env python3
"""Convert reconstructed FrozenLake traces into LVR trajectory manifests.

The converter does not copy images. Paths in the generated JSONL files are
relative to ``--input-root`` so the dataset can be moved without rewriting the
manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "frozenlake_lvr_v1"
DEFAULT_INSTRUCTION = (
    "You are playing a deterministic FrozenLake navigation puzzle. In the image, "
    "locate the character, the treasure (goal), and the holes (obstacles). Move the "
    "character one grid cell at a time using only LEFT, RIGHT, UP, or DOWN. Touching "
    "a hole or obstacle loses the game, and moves may not leave the board. Find a "
    "shortest safe route to the treasure. Return only the space-separated action "
    "sequence."
)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ConvertedSample:
    record: dict[str, Any]
    layout_key: tuple[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("training_samples/frozenlake"),
        help="Directory containing numbered trace directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frozenlake"),
        help="Directory in which JSONL manifests are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Fixed user instruction placed after the input image token.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def action_between(source: int, destination: int, level: int) -> str:
    """Return the deterministic cardinal action for two adjacent cell ids."""
    source_row, source_col = divmod(source, level)
    destination_row, destination_col = divmod(destination, level)
    delta = (destination_row - source_row, destination_col - source_col)
    actions = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }
    _require(delta in actions, f"non-adjacent transition {source} -> {destination}")
    return actions[delta]


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def convert_trace(trace_path: Path, input_root: Path, instruction: str) -> ConvertedSample:
    with trace_path.open("r", encoding="utf-8") as stream:
        trace = json.load(stream)

    sample_dir = trace_path.parent
    sample_id = sample_dir.name
    prefix = f"{sample_id}: "
    _require(trace.get("task") == "frozenlake", prefix + "task is not frozenlake")
    _require(sample_id.isdigit(), prefix + "sample directory is not numeric")
    _require(int(sample_id) == trace.get("trace_index"), prefix + "trace_index mismatch")

    level = trace.get("meta", {}).get("level")
    layout = trace.get("meta", {}).get("layout")
    start = trace.get("meta", {}).get("start_pos")
    target = trace.get("meta", {}).get("target_pos")
    input_states = trace.get("input_states")
    transition_count = trace.get("transition_count")
    frame_count = trace.get("frame_count")

    _require(isinstance(level, int) and level > 1, prefix + "invalid level")
    _require(isinstance(layout, list) and len(layout) == level, prefix + "invalid layout")
    _require(
        all(isinstance(row, list) and len(row) == level for row in layout),
        prefix + "layout is not square",
    )
    _require(isinstance(input_states, list) and input_states, prefix + "input_states is empty")
    _require(
        isinstance(transition_count, int) and transition_count > 0,
        prefix + "invalid transition_count",
    )
    _require(frame_count == transition_count + 1, prefix + "frame/transition count mismatch")
    _require(len(input_states) == transition_count, prefix + "state/transition count mismatch")

    states = [*input_states, target]
    _require(states[0] == start, prefix + "first state is not start_pos")
    _require(layout[start // level][start % level] == "S", prefix + "start_pos is not S")
    _require(layout[target // level][target % level] == "G", prefix + "target_pos is not G")
    _require(
        all(layout[state // level][state % level] != "H" for state in states),
        prefix + "path enters a hole",
    )

    actions = [action_between(a, b, level) for a, b in zip(states, states[1:])]
    _require(len(actions) == transition_count, prefix + "derived action count mismatch")

    frame_paths = [sample_dir / f"frame_{index:03d}.png" for index in range(frame_count)]
    for frame_path in frame_paths:
        _require(frame_path.is_file(), prefix + f"missing {frame_path.name}")
    actual_frames = sorted(sample_dir.glob("frame_*.png"))
    _require(actual_frames == frame_paths, prefix + "frame filenames are not exact and contiguous")

    initial_image = _relative_posix(frame_paths[0], input_root)
    # Every action is aligned to the state image that results from that action.
    # This includes the terminal goal image and guarantees one target per action.
    aux_images = [_relative_posix(path, input_root) for path in frame_paths[1:]]
    final_image = aux_images[-1]
    answer = " ".join(actions)
    trace_relpath = _relative_posix(trace_path, input_root)

    flat_layout = "".join(cell for row in layout for cell in row)
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": f"frozenlake-{sample_id}",
        "task": "frozenlake",
        "level": level,
        "image": [initial_image],
        "initial_image": initial_image,
        "aux_images": aux_images,
        "final_image": final_image,
        "instruction": instruction,
        "actions": actions,
        "answer": answer,
        "trajectory": {
            "states": states,
            "transition_count": transition_count,
        },
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{instruction}",
            },
            {
                "from": "gpt",
                "value": f"<lvr>\n<answer>{answer}</answer>",
            },
        ],
        "source": {
            "trace": trace_relpath,
            "trace_index": trace["trace_index"],
            "layout": layout,
        },
    }
    return ConvertedSample(record=record, layout_key=(level, flat_layout))


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def assign_splits(
    samples: list[ConvertedSample], ratios: dict[str, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Split samples by full board layout, approximately stratified by level."""
    by_level: dict[int, list[ConvertedSample]] = defaultdict(list)
    for sample in samples:
        by_level[sample.record["level"]].append(sample)

    output: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for level, level_samples in sorted(by_level.items()):
        groups: dict[tuple[int, str], list[ConvertedSample]] = defaultdict(list)
        for sample in level_samples:
            groups[sample.layout_key].append(sample)

        randomizer = random.Random(seed + level)
        shuffled_groups = list(groups.values())
        randomizer.shuffle(shuffled_groups)
        shuffled_groups.sort(key=len, reverse=True)

        targets = _target_counts(len(level_samples), ratios)
        current = Counter()
        for group in shuffled_groups:
            # Assign the whole layout group to the split with the largest
            # remaining sample deficit. Stable tuple ordering breaks ties.
            split = max(
                SPLIT_NAMES,
                key=lambda name: (targets[name] - current[name], -SPLIT_NAMES.index(name)),
            )
            for sample in group:
                record = dict(sample.record)
                record["split"] = split
                output[split].append(record)
            current[split] += len(group)

    for split in SPLIT_NAMES:
        output[split].sort(key=lambda record: record["id"])
    return output


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    image_root_hint = args.input_root.as_posix()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    _require(input_root.is_dir(), f"input root does not exist: {input_root}")

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    _require(all(ratio >= 0 for ratio in ratios.values()), "split ratios must be non-negative")
    _require(math.isclose(sum(ratios.values()), 1.0), "split ratios must sum to 1")

    trace_paths = sorted(input_root.glob("[0-9]*/trace.json"))
    _require(trace_paths, f"no trace.json files found below {input_root}")
    samples = [convert_trace(path, input_root, args.instruction) for path in trace_paths]
    splits = assign_splits(samples, ratios, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    all_records = sorted(
        (record for records in splits.values() for record in records),
        key=lambda record: record["id"],
    )
    write_jsonl(output_dir / "all.jsonl", all_records)

    layout_splits: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for split, records in splits.items():
        for record in records:
            layout = "".join(cell for row in record["source"]["layout"] for cell in row)
            layout_splits[split].add(f'{record["level"]}:{layout}')
    overlap = {
        f"{left}_{right}": len(layout_splits[left] & layout_splits[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }

    info = {
        "schema_version": SCHEMA_VERSION,
        "image_root": image_root_hint,
        "path_semantics": "All image and source trace paths are relative to image_root.",
        "instruction": args.instruction,
        "action_vocabulary": ["LEFT", "RIGHT", "UP", "DOWN"],
        "action_separator": " ",
        "latent_target_semantics": (
            "aux_images[i] is the successor-state image produced by actions[i]; "
            "the last aux image is also final_image."
        ),
        "seed": args.seed,
        "requested_split_ratios": ratios,
        "sample_count": len(all_records),
        "split_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "level_counts": dict(
            sorted(Counter(record["level"] for record in all_records).items())
        ),
        "transition_count": {
            "total": sum(len(record["actions"]) for record in all_records),
            "minimum": min(len(record["actions"]) for record in all_records),
            "maximum": max(len(record["actions"]) for record in all_records),
            "mean": sum(len(record["actions"]) for record in all_records) / len(all_records),
        },
        "layout_overlap_between_splits": overlap,
    }
    with (output_dir / "dataset_info.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(info, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return info


def main() -> None:
    args = parse_args()
    info = prepare_dataset(args)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
