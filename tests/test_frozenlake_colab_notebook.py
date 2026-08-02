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

    def test_missing_latent_end_parameter_is_initialized_after_loading(self) -> None:
        model_source = (REPOSITORY_ROOT / "src/model/qwen_lvr_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def reset_lvr_latent_end_emb", model_source)
        self.assertIn("dtype=torch.float32", model_source)

        for relative_path in (
            "scripts/smoke_test_frozenlake_lvr.py",
            "src/train/train_frozenlake_lvr.py",
        ):
            source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("output_loading_info=True", source)
            self.assertIn('if "lvr_latent_end_emb" in loading_info["missing_keys"]', source)
            self.assertIn("model.reset_lvr_latent_end_emb()", source)

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
