"""Minimal smoke test for validating a fresh MemoryBench installation.

This script is intentionally lightweight:
- No API keys required.
- No model calls or dataset downloads.
- No experiment execution.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _check_python_version(errors: list[str]) -> None:
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10+ is required, but current version is {sys.version.split()[0]}"
        )


def _check_imports(errors: list[str]) -> None:
    modules = [
        "datasets",
        "dotenv",
        "streamlit",
        "tqdm",
    ]
    for module_name in modules:
        try:
            __import__(module_name)
        except Exception as exc:
            errors.append(f"Failed to import '{module_name}': {exc}")


def _check_core_entry(errors: list[str]) -> None:
    try:
        import memorybench  # type: ignore

        required_symbols = ["load_memory_bench", "evaluate", "summary_results"]
        for symbol in required_symbols:
            if not hasattr(memorybench, symbol):
                errors.append(f"memorybench is missing required symbol: {symbol}")
    except Exception as exc:
        errors.append(f"Failed to import 'memorybench': {exc}")


def _check_files(errors: list[str]) -> None:
    required_paths = [
        ROOT / "configs" / "datasets" / "each.json",
        ROOT / "configs" / "datasets" / "domain.json",
        ROOT / "configs" / "datasets" / "task.json",
        ROOT / "frontend" / "streamlit_app.py",
        ROOT / "frontend" / "requirements.txt",
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required file: {path.relative_to(ROOT)}")


def _check_frontend_module_load(errors: list[str]) -> None:
    frontend_app = ROOT / "frontend" / "streamlit_app.py"
    try:
        spec = importlib.util.spec_from_file_location("memorybench_frontend_app", frontend_app)
        if spec is None or spec.loader is None:
            errors.append("Failed to load module spec for frontend/streamlit_app.py")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        errors.append(f"Failed to import frontend/streamlit_app.py: {exc}")


def _run_python_module_check(module_args: list[str], errors: list[str], timeout_sec: int = 120) -> None:
    cmd = [sys.executable, "-m", *module_args]
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        errors.append(f"Command timed out: {' '.join(cmd)}")
        return
    except Exception as exc:
        errors.append(f"Failed to run command {' '.join(cmd)}: {exc}")
        return

    if ret.returncode != 0:
        output = (ret.stdout or "") + (ret.stderr or "")
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        tail = " | ".join(lines[-8:]) if lines else "No output captured"
        errors.append(f"Command failed ({ret.returncode}): {' '.join(cmd)} | {tail}")


def _check_entrypoint_execution(errors: list[str]) -> None:
    # Verify frontend can be started from current Python environment.
    _run_python_module_check(["streamlit", "--version"], errors)

    # Verify experiment entrypoints are executable without API config.
    _run_python_module_check(["src.off-policy", "--help"], errors)
    _run_python_module_check(["src.on-policy", "--help"], errors)


def main() -> int:
    errors: list[str] = []
    _check_python_version(errors)
    _check_imports(errors)
    _check_files(errors)
    _check_core_entry(errors)
    _check_frontend_module_load(errors)
    _check_entrypoint_execution(errors)

    if errors:
        print("MemoryBench smoke test: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print("MemoryBench smoke test: PASSED")
    print("Environment is ready (no API configuration required for this check).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
