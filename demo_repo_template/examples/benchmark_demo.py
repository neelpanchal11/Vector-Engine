from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main() -> None:
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "benchmarks", "compare_bruteforce_vs_faiss.py"),
        "--mode",
        "exact",
        "--loops",
        "2",
        "--warmup",
        "1",
    ]
    out = subprocess.check_output(cmd, text=True)
    print(out)


if __name__ == "__main__":
    main()
