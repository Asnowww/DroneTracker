"""Run every offline test. Requires no AirSim, no GPU, no trained weights.

    python tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "tests/test_predictor.py",
    "tests/test_controller.py",
    "tests/test_labeling.py",
    "tests/test_prediction_benchmark.py",
]


def main() -> None:
    compile_check = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"], cwd=str(ROOT)
    )
    if compile_check.returncode != 0:
        sys.exit("compileall failed")
    print("[PASS] compileall")

    failures: list[str] = []
    for module in MODULES:
        print(f"\n--- {module} ---")
        result = subprocess.run([sys.executable, module], cwd=str(ROOT))
        if result.returncode != 0:
            failures.append(module)

    print()
    if failures:
        for module in failures:
            print(f"[FAIL] {module}")
        sys.exit(1)
    for module in MODULES:
        print(f"[PASS] {module}")
    print("\nall_offline_tests_passed=true")


if __name__ == "__main__":
    main()
