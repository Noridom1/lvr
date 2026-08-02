import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_frozenlake_dataset import (
    DEFAULT_INSTRUCTION,
    action_between,
    prepare_dataset,
)


class PrepareFrozenLakeDatasetTest(unittest.TestCase):
    def test_action_between(self):
        self.assertEqual(action_between(4, 1, 3), "UP")
        self.assertEqual(action_between(4, 7, 3), "DOWN")
        self.assertEqual(action_between(4, 3, 3), "LEFT")
        self.assertEqual(action_between(4, 5, 3), "RIGHT")
        with self.assertRaisesRegex(ValueError, "non-adjacent"):
            action_between(2, 3, 3)

    def test_prepare_dataset_aligns_actions_and_future_images(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_root = root / "raw"
            sample_dir = input_root / "00000000"
            output_dir = root / "formatted"
            sample_dir.mkdir(parents=True)
            trace = {
                "task": "frozenlake",
                "trace_index": 0,
                "transition_count": 2,
                "frame_count": 3,
                "input_states": [7, 8],
                "meta": {
                    "start_pos": 7,
                    "target_pos": 5,
                    "layout": [
                        ["F", "H", "F"],
                        ["H", "F", "G"],
                        ["F", "S", "F"],
                    ],
                    "level": 3,
                },
            }
            (sample_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
            for index in range(3):
                (sample_dir / f"frame_{index:03d}.png").write_bytes(b"png")

            args = argparse.Namespace(
                input_root=input_root,
                output_dir=output_dir,
                seed=42,
                train_ratio=1.0,
                validation_ratio=0.0,
                test_ratio=0.0,
                instruction=DEFAULT_INSTRUCTION,
            )
            info = prepare_dataset(args)

            record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["initial_image"], "00000000/frame_000.png")
            self.assertEqual(
                record["aux_images"],
                ["00000000/frame_001.png", "00000000/frame_002.png"],
            )
            self.assertEqual(record["final_image"], "00000000/frame_002.png")
            self.assertEqual(record["trajectory"]["states"], [7, 8, 5])
            self.assertEqual(record["actions"], ["RIGHT", "UP"])
            self.assertEqual(record["answer"], "RIGHT UP")
            self.assertEqual(
                record["conversations"][1]["value"],
                "<lvr>\n<answer>RIGHT UP</answer>",
            )
            self.assertEqual(info["sample_count"], 1)
            self.assertEqual(info["layout_overlap_between_splits"]["train_validation"], 0)


if __name__ == "__main__":
    unittest.main()
