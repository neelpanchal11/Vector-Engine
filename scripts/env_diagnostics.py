from __future__ import annotations

import json
import os
import platform
import subprocess
import sys


def _run_subprocess(code: str) -> dict[str, object]:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "segfault_suspected": proc.returncode == -11 or proc.returncode == 139,
    }


def _numpy_info_subprocess() -> dict[str, object]:
    return _run_subprocess(
        (
            "import numpy as np; "
            "import sys; "
            "print(np.__version__); "
            "np.__config__.show()"
        )
    )


def _numpy_smoke_subprocess() -> dict[str, object]:
    return _run_subprocess(
        (
            "import numpy as np; "
            "x=np.random.default_rng(7).normal(size=(64,64)).astype('float32'); "
            "y=x@x.T; "
            "z=np.linalg.svd(y[:8,:8], full_matrices=False); "
            "print('ok', y.shape[0], len(z))"
        )
    )


def collect_env_report() -> dict[str, object]:
    numpy_info = _numpy_info_subprocess()
    smoke = _numpy_smoke_subprocess()
    is_macos_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    suggestions: list[str] = []
    if is_macos_arm:
        suggestions.append(
            "Use a clean virtualenv with python3.12 and constraints-macos-arm64-py312.txt before pytest."
        )
    if numpy_info.get("segfault_suspected") or smoke.get("segfault_suspected"):
        suggestions.extend(
            [
                "Recreate venv from scratch and reinstall constrained dependencies.",
                "Use a Python 3.12 patch release (3.12.6+) instead of 3.12.0 if available.",
                "Ensure python and pip point to the same .venv312 interpreter.",
                "Retry with: python -m pytest -q after running this script again.",
            ]
        )

    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "venv_active": sys.prefix != sys.base_prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
        },
        "numpy": {
            "info_probe": numpy_info,
            "smoke": smoke,
        },
        "suggestions": suggestions,
    }


def main() -> None:
    payload = collect_env_report()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
