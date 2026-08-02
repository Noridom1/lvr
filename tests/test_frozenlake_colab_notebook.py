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
        self.assertNotIn(
            'str(REPO_DIR / "requirements.txt")',
            combined_source,
        )

    def test_dependency_files_do_not_reference_missing_av_release(self) -> None:
        general = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
        colab = (REPOSITORY_ROOT / "requirements-frozenlake-colab.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("av==14.3.0", general)
        self.assertIn("av==14.4.0", general)
        self.assertIn("av==14.4.0", colab)


if __name__ == "__main__":
    unittest.main()
