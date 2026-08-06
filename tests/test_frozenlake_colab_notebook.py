import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "frozenlake_lvr_colab.ipynb"


def cell_source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


class FrozenLakeColabNotebookTest(unittest.TestCase):
    def test_notebook_code_compiles_and_uses_task_requirements(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for index, source in enumerate(code_cells, start=1):
            compile(source, f"{NOTEBOOK_PATH.name}:code-cell-{index}", "exec")

        combined_source = "\n".join(code_cells)
        self.assertIn("requirements-frozenlake-colab.txt", combined_source)
        self.assertIn("torch==2.6.0", combined_source)
        self.assertIn("Expected torch 2.6.0", combined_source)
        self.assertNotIn("PREFER_FLASH_ATTENTION", combined_source)
        self.assertIn('"pip", "uninstall", "-y", "flash-attn"', combined_source)
        self.assertIn('ATTENTION = "sdpa"', combined_source)
        self.assertIn('DISABLE_FLASH_ATTN2 = "True"', combined_source)
        self.assertIn("from scripts.colab_subprocess import run_streaming", combined_source)
        self.assertIn("run_streaming(", combined_source)
        self.assertNotIn("import flash_attn", combined_source)
        self.assertIn("FrozenLake preflight passed", combined_source)
        self.assertNotIn("capture_output=True", combined_source)
        self.assertIn('"scripts.smoke_test_frozenlake_lvr"', combined_source)
        self.assertIn('"evaluation.evaluate_frozenlake_lvr"', combined_source)
        self.assertIn("--sample-index", combined_source)
        self.assertIn("TEST_SAMPLE_INDEX", combined_source)
        self.assertIn("TRAIN_SAMPLE_INDEX", combined_source)
        self.assertNotIn("--save-distance-trace", combined_source)
        self.assertIn("--lvr-steps", combined_source)
        self.assertIn("--sweep-lvr-steps", combined_source)
        self.assertIn("--teacher-forced-diagnostic", combined_source)
        self.assertIn('prediction["raw_generation"]', combined_source)
        self.assertIn('prediction["teacher_forced"]', combined_source)
        self.assertIn("latent_exit_reason", combined_source)
        self.assertIn("qwen2.5-vl-3b-lora", combined_source)
        self.assertIn('drive.mount("/content/drive")', combined_source)
        self.assertNotIn('"scripts/smoke_test_frozenlake_lvr.py"', combined_source)
        self.assertNotIn('"evaluation/evaluate_frozenlake_lvr.py"', combined_source)
        self.assertNotIn(
            'str(REPO_DIR / "requirements.txt")',
            combined_source,
        )

    def test_dependency_files_use_colab_compatible_av_wheel_release(self) -> None:
        general = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
        colab = (REPOSITORY_ROOT / "requirements-frozenlake-colab.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("av==14.3.0", general)
        self.assertNotIn("av==14.4.0", general)
        self.assertIn("av==14.2.0", general)
        self.assertIn("av==14.2.0", colab)

    def test_colab_launcher_defaults_to_sdpa(self) -> None:
        launcher = (
            REPOSITORY_ROOT / "scripts" / "finetune_lvr_frozenlake_3b_colab.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'export DISABLE_FLASH_ATTN2="${DISABLE_FLASH_ATTN2:-True}"',
            launcher,
        )
        self.assertIn('export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"', launcher)

    def test_colab_launcher_uses_memory_safe_lora(self) -> None:
        launcher = (
            REPOSITORY_ROOT / "scripts" / "finetune_lvr_frozenlake_3b_colab.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export LORA_ENABLE="${LORA_ENABLE:-True}"', launcher)
        self.assertIn('export FREEZE_LLM="${FREEZE_LLM:-True}"', launcher)
        self.assertIn(
            'export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"',
            launcher,
        )

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        combined_source = "\n".join(
            cell_source(cell) for cell in notebook["cells"]
        )
        self.assertIn('"LORA_ENABLE": "True"', combined_source)
        self.assertIn('"DEEPSPEED_CONFIG": "scripts/zero2.json"', combined_source)

    def test_frozenlake_lora_uses_standard_adapter_checkpoint(self) -> None:
        train_source = (
            REPOSITORY_ROOT / "src/train/train_frozenlake_lvr.py"
        ).read_text(encoding="utf-8")
        eval_source = (
            REPOSITORY_ROOT / "evaluation/evaluate_frozenlake_lvr.py"
        ).read_text(encoding="utf-8")
        trainer_source = (
            REPOSITORY_ROOT / "src/trainer/lvr_trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("get_peft_model", train_source)
        self.assertNotIn("non_lora_state_dict.bin", train_source)
        self.assertNotIn("non_lora_state_dict.bin", eval_source)
        self.assertIn("non_lora_state_dict.bin", trainer_source)
        self.assertIn("PeftModel.from_pretrained", eval_source)
        self.assertIn("merge_and_unload", eval_source)
        self.assertIn("args.sample_index", eval_source)

    def test_evaluator_separates_latent_placeholders_from_action_output(self) -> None:
        eval_source = (
            REPOSITORY_ROOT / "evaluation/evaluate_frozenlake_lvr.py"
        ).read_text(encoding="utf-8")
        model_source = (REPOSITORY_ROOT / "src/model/qwen_lvr_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("lvr_diagnostics", model_source)
        self.assertIn('"action_start_generated_index"', model_source)
        self.assertIn('"fixed_budget"', model_source)
        self.assertIn('latent["latent_exit_reason"] == "fixed_budget"', eval_source)
        self.assertIn('"latent_fixed_budget_exit"', eval_source)
        self.assertIn('"action_output": action_output', eval_source)
        self.assertIn('"raw_generation": raw_output', eval_source)
        self.assertIn('"teacher_forced": teacher_forced', eval_source)
        self.assertIn("run_teacher_forced_diagnostic", eval_source)
        self.assertNotIn("--save-distance-trace", eval_source)

    def test_direct_entrypoints_add_repository_root_to_python_path(self) -> None:
        expected_roots = {
            "scripts/smoke_test_frozenlake_lvr.py": "parents[1]",
            "evaluation/evaluate_frozenlake_lvr.py": "parents[1]",
            "src/train/train_frozenlake_lvr.py": "parents[2]",
        }
        for relative_path, expected_parent in expected_roots.items():
            source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(f"Path(__file__).resolve().{expected_parent}", source)
            self.assertIn("sys.path.insert(0, str(REPOSITORY_ROOT))", source)

    def test_generic_latent_end_ablation_is_preserved_but_frozenlake_avoids_it(self) -> None:
        model_source = (REPOSITORY_ROOT / "src/model/qwen_lvr_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def reset_lvr_latent_end_emb", model_source)
        self.assertIn("dtype=torch.float32", model_source)

        for relative_path in (
            "scripts/smoke_test_frozenlake_lvr.py",
            "src/train/train_frozenlake_lvr.py",
            "evaluation/evaluate_frozenlake_lvr.py",
        ):
            source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("model.reset_lvr_latent_end_emb()", source)

        self.assertIn("deepspeed.zero.GatheredParameters", model_source)
        self.assertIn("def lvr_latent_end_is_finite", model_source)

    def test_streaming_subprocess_helper_reports_live_command_failures(self) -> None:
        helper_source = (REPOSITORY_ROOT / "scripts/colab_subprocess.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("stderr=subprocess.STDOUT", helper_source)
        self.assertIn("flush=True", helper_source)
        self.assertIn("Last {len(recent_output)} output lines", helper_source)


if __name__ == "__main__":
    unittest.main()
