import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_have_valid_multicell_python():
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebook_paths

    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        cell_ids = [cell.get("id") for cell in cells]

        assert len(code_cells) >= 2, f"{path.name} should keep demo steps in separate cells"
        assert any(cell["cell_type"] == "markdown" for cell in cells), path.name
        assert all(cell_ids), f"{path.name} contains cells without ids"
        assert len(cell_ids) == len(set(cell_ids)), f"{path.name} contains duplicate cell ids"

        for cell_number, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            source = source if isinstance(source, str) else "".join(source)
            if source.lstrip().startswith("%"):
                continue
            compile(source, f"{path.name}:cell{cell_number}", "exec")
