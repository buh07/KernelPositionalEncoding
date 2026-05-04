#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import ASSETS_ROOT, SCHEMAS_ROOT, ensure_dir, file_sha256, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import CORE_SCHEMA_FILES  # noqa: E402


def _module_exists(name: str, python_bin: str) -> bool:
    proc = subprocess.run(
        [
            str(python_bin),
            "-c",
            "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
            str(name),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    return int(proc.returncode) == 0


def _install_fasttext(python_bin: str) -> None:
    subprocess.run([python_bin, "-m", "pip", "install", "fasttext"], check=True, cwd=str(ROOT))


def _download_fasttext_model(asset_dir: Path) -> Path:
    import fasttext.util  # type: ignore

    ensure_dir(asset_dir)
    # This downloads to cwd as cc.en.300.bin
    cwd_before = os.getcwd()
    try:
        os.chdir(str(asset_dir))
        fasttext.util.download_model("en", if_exists="ignore")
    finally:
        os.chdir(cwd_before)

    model_path = asset_dir / "cc.en.300.bin"
    if not model_path.exists():
        raise RuntimeError("fastText download finished without cc.en.300.bin")
    return model_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="reinforce_exp2 strict preflight checks + dependency/assets prep",
        allow_abbrev=False,
    )
    p.add_argument("--python-bin", default=str(ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--output-root", default=str(ROOT / "results" / "reinforce_exp2" / "preflight"))
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.output_root))
    python_bin = str(args.python_bin)
    asset_dir = ensure_dir(Path(ASSETS_ROOT) / "fasttext")
    model_path = asset_dir / "cc.en.300.bin"
    pin_path = asset_dir / "cc.en.300.sha256"

    steps: list[dict[str, Any]] = []

    # Validate schema files exist before pipeline starts.
    missing_schemas = []
    for _, schema_name in CORE_SCHEMA_FILES.items():
        sp = SCHEMAS_ROOT / schema_name
        if not sp.exists():
            missing_schemas.append(str(sp))
    if missing_schemas:
        raise RuntimeError(f"Missing reinforce_exp2 schemas: {missing_schemas}")
    steps.append({"step": "schema_presence", "ok": True, "n_schemas": len(CORE_SCHEMA_FILES)})

    fasttext_pre = _module_exists("fasttext", python_bin)
    if not fasttext_pre:
        _install_fasttext(python_bin)
    fasttext_post = _module_exists("fasttext", python_bin)
    if not fasttext_post:
        raise RuntimeError("fasttext install failed")
    steps.append({"step": "fasttext_install", "ok": True, "installed": (not fasttext_pre)})

    if not model_path.exists():
        dl_path = _download_fasttext_model(asset_dir)
        if dl_path.resolve() != model_path.resolve():
            shutil.copy2(dl_path, model_path)
    if not model_path.exists():
        raise RuntimeError(f"Missing fastText model after download: {model_path}")

    model_sha = file_sha256(model_path)

    checksum_pinned = False
    if pin_path.exists():
        expected = pin_path.read_text(encoding="utf-8").strip().lower()
        if expected and expected != str(model_sha).lower():
            raise RuntimeError(
                f"Pinned checksum mismatch for {model_path}: expected={expected} actual={model_sha}. "
                "Delete asset and rerun preflight if intentional refresh is required."
            )
        checksum_pinned = True
    else:
        pin_path.write_text(str(model_sha) + "\n", encoding="utf-8")
        checksum_pinned = True

    steps.append(
        {
            "step": "fasttext_model",
            "ok": True,
            "path": str(model_path),
            "sha256": model_sha,
            "sha256_pin_path": str(pin_path),
            "checksum_pinned": bool(checksum_pinned),
            "size_bytes": model_path.stat().st_size,
        }
    )

    payload = {
        "timestamp": timestamp_now(),
        "experiment_id": "preflight",
        "strict_todo_ready": True,
        "steps": steps,
    }
    write_json(out_dir / "summary.json", payload)
    write_json(
        out_dir / "fasttext_asset_manifest.json",
        {
            "path": str(model_path),
            "sha256": model_sha,
            "sha256_pin_path": str(pin_path),
            "checksum_pinned": bool(checksum_pinned),
            "timestamp": timestamp_now(),
        },
    )
    print(f"[preflight] fasttext model ready: {model_path}")
    print(f"[preflight] sha256={model_sha}")


if __name__ == "__main__":
    main()
