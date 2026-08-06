import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FrozenLakePaperAlignedTest(unittest.TestCase):
    def test_response_uses_standard_lvr_boundary_without_latent_end(self):
        source = (REPOSITORY_ROOT / "src/frozenlake_lvr_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "LVR_START_TOKEN + (LVR_TOKEN * latent_tokens) + LVR_END_TOKEN",
            source,
        )
        self.assertNotIn("LVR_LATENT_END_TOKEN", source)

    def test_fixed_decoder_forces_placeholder_and_end_tokens(self):
        source = (REPOSITORY_ROOT / "src/model/qwen_lvr_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def update_fixed_lvr_state", source)
        self.assertIn("remaining_steps = remaining_steps - previous_mode.long()", source)
        self.assertIn("budget_finished = previous_mode & (remaining_steps <= 0)", source)
        self.assertIn("torch.full_like(next_tokens, self.config.lvr_id)", source)
        self.assertIn("torch.full_like(next_tokens, self.config.lvr_end_id)", source)
        self.assertIn('trace["latent_exit_reason"] = "fixed_budget"', source)

    def test_budget_selection_is_deterministic(self):
        source = (
            REPOSITORY_ROOT / "evaluation/evaluate_frozenlake_lvr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("DEFAULT_BUDGET_SWEEP = (4, 8, 16, 32, 64, 128, 256, 512)", source)
        self.assertIn("summary[\"goal_success_rate\"]", source)
        self.assertIn("summary[\"shortest_path_success_rate\"]", source)
        self.assertIn("summary[\"exact_match_accuracy\"]", source)
        self.assertIn("-summary[\"lvr_steps\"]", source)


if __name__ == "__main__":
    unittest.main()
