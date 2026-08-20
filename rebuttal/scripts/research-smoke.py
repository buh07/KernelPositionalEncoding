#!/usr/bin/python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class _Args(argparse.Namespace):
    target: str | None = None
    fingerprint: bool = False


def _repo_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in (candidate.parent, *candidate.parents):
        if (parent / "rebuttal" / "configs" / "base.toml").is_file():
            return parent
    raise RuntimeError("Could not resolve repository root from research smoke entrypoint.")


def _fingerprint_line(repo_root: Path) -> str:
    script_path = repo_root / "rebuttal" / "scripts" / "research-smoke.py"
    digest = hashlib.sha256()
    digest.update(b"research-smoke-v1\0")
    digest.update(script_path.read_bytes())
    return f"research-smoke-v1 sha256:{digest.hexdigest()}"


def _resolve_python() -> str:
    override = os.environ.get("SI_REBUTTAL_SMOKE_PYTHON", "").strip()
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            raise RuntimeError("SI_REBUTTAL_SMOKE_PYTHON must be an absolute executable path.")
        resolved = shutil.which(str(candidate))
        if resolved is None:
            raise RuntimeError(f"SI_REBUTTAL_SMOKE_PYTHON is unavailable: {candidate}")
        if not os.path.isabs(resolved):
            raise RuntimeError(f"Resolved SI_REBUTTAL_SMOKE_PYTHON is not absolute: {resolved}")
        return resolved

    for name in ("python3", "python"):
        resolved = shutil.which(name)
        if resolved is None:
            continue
        if not os.path.isabs(resolved):
            raise RuntimeError(f"Resolved {name} from PATH is not absolute: {resolved}")
        return resolved
    raise RuntimeError(
        "Could not resolve a Python executable from SI_REBUTTAL_SMOKE_PYTHON or PATH."
    )


def _normalize_target(target: str, repo_root: Path) -> str:
    candidate = Path(target)
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Smoke target must resolve inside the repository: {target}") from exc
    if relative != "rebuttal":
        raise RuntimeError(f"Smoke target must be exactly 'rebuttal', found {target!r}.")
    return relative


def _temp_env(repo_root: Path, temp_root: Path) -> dict[str, str]:
    rebuttal_root = repo_root / "rebuttal"
    mutable_root = temp_root / "mutable"
    runs_root = mutable_root / "runs"
    data_root = mutable_root / "data"
    materialized_root = data_root / "materialized"
    logs_root = mutable_root / "logs"
    cache_root = mutable_root / "cache"
    models_root = temp_root / "models"
    pycache_root = temp_root / "pycache"
    subprocess_tmp_root = temp_root / "tmp"

    _ = runs_root.mkdir(parents=True, exist_ok=True)
    _ = materialized_root.mkdir(parents=True, exist_ok=True)
    _ = logs_root.mkdir(parents=True, exist_ok=True)
    _ = cache_root.mkdir(parents=True, exist_ok=True)
    _ = models_root.mkdir(parents=True, exist_ok=True)
    _ = pycache_root.mkdir(parents=True, exist_ok=True)
    _ = subprocess_tmp_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SI_REBUTTAL_RUNS_ROOT"] = runs_root.relative_to(rebuttal_root).as_posix()
    env["SI_REBUTTAL_DATA_ROOT"] = data_root.relative_to(rebuttal_root).as_posix()
    env["SI_REBUTTAL_LOGS_ROOT"] = logs_root.relative_to(rebuttal_root).as_posix()
    env["SI_REBUTTAL_CACHE_ROOT"] = cache_root.relative_to(rebuttal_root).as_posix()
    env["SI_REBUTTAL_MODELS_ROOT"] = str(models_root)
    env["TMPDIR"] = str(subprocess_tmp_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(pycache_root)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""

    for name in (
        "LLAMA_3_1_8B",
        "MISTRAL_7B_V0_1",
        "OLMO_2_7B",
    ):
        model_dir = models_root / name.lower()
        tokenizer_dir = models_root / f"{name.lower()}_tokenizer"
        _ = model_dir.mkdir(parents=True, exist_ok=True)
        _ = tokenizer_dir.mkdir(parents=True, exist_ok=True)
        env[f"SI_REBUTTAL_MODEL_{name}"] = str(model_dir)
        env[f"SI_REBUTTAL_TOKENIZER_{name}"] = str(tokenizer_dir)

    pythonpath_entries = [str((repo_root / "rebuttal" / "src").resolve())]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _run_command(
    command: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
    )


def _emit_failure(result: subprocess.CompletedProcess[bytes], *, label: str) -> int:
    if result.stdout:
        _ = sys.stderr.buffer.write(result.stdout)
        if not result.stdout.endswith(b"\n"):
            _ = sys.stderr.buffer.write(b"\n")
    if result.stderr:
        _ = sys.stderr.buffer.write(result.stderr)
        if not result.stderr.endswith(b"\n"):
            _ = sys.stderr.buffer.write(b"\n")
    if not result.stdout and not result.stderr:
        _ = sys.stderr.write(f"{label} failed with exit code {result.returncode}\n")
    return result.returncode or 1


def _emit_bytes(label: str, payload: bytes) -> None:
    _ = sys.stderr.write(f"{label}:\n")
    _ = sys.stderr.buffer.write(payload)
    if payload and not payload.endswith(b"\n"):
        _ = sys.stderr.buffer.write(b"\n")


def _run_smoke(target: str) -> int:
    repo_root = _repo_root()
    _ = _normalize_target(target, repo_root)
    python_executable = _resolve_python()
    with tempfile.TemporaryDirectory(
        prefix="research-smoke-", dir=repo_root / "rebuttal"
    ) as raw_temp:
        temp_root = Path(raw_temp).resolve()
        env = _temp_env(repo_root, temp_root)
        validate_command = [
            python_executable,
            "-m",
            "si_rebuttal",
            "validate-config",
            "--config",
            "rebuttal/configs/base.toml",
            "--sweep",
            "rebuttal/configs/sweep.toml",
        ]
        first = _run_command(validate_command, cwd=repo_root, env=env)
        if first.returncode != 0:
            return _emit_failure(first, label="validate-config")
        second = _run_command(validate_command, cwd=repo_root, env=env)
        if second.returncode != 0:
            return _emit_failure(second, label="validate-config")
        if first.stdout != second.stdout:
            _ = sys.stderr.write("validate-config stdout differed between repeated runs\n")
            _emit_bytes("first stdout", first.stdout)
            _emit_bytes("second stdout", second.stdout)
            return 1
        toy_run_root = temp_root / "mutable" / "runs" / "toy-smoke"
        _ = toy_run_root.mkdir(parents=True, exist_ok=True)
        smoke = _run_command(
            [
                python_executable,
                "-m",
                "si_rebuttal",
                "toy-smoke",
                "--config",
                "rebuttal/configs/base.toml",
                "--sweep",
                "rebuttal/configs/sweep.toml",
                "--run-root",
                str(toy_run_root),
            ],
            cwd=repo_root,
            env=env,
        )
        if smoke.returncode != 0:
            return _emit_failure(smoke, label="toy-smoke")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-smoke")
    _ = parser.add_argument("target", nargs="?")
    _ = parser.add_argument("--fingerprint", action="store_true")
    args = parser.parse_args(argv, namespace=_Args())

    try:
        repo_root = _repo_root()
        if args.fingerprint:
            if args.target is not None:
                _ = parser.error("--fingerprint does not accept a target")
            _ = sys.stdout.write(_fingerprint_line(repo_root) + "\n")
            return 0
        if args.target is None:
            _ = parser.error("target is required unless --fingerprint is set")
        return _run_smoke(args.target)
    except RuntimeError as exc:
        _ = sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
