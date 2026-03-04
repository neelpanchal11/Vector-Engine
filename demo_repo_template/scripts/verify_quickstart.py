from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    print(f"$ {' '.join(cmd)}")
    print(out.strip())
    print("-" * 40)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"

    _run([sys.executable, str(examples / "semantic_demo.py")])
    _run([sys.executable, str(examples / "item_similarity_demo.py")])
    _run([sys.executable, str(examples / "benchmark_demo.py")])
    print("quickstart_verify: ok")


if __name__ == "__main__":
    main()
