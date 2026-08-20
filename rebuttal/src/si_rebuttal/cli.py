from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from . import runner as _runner
from .operations import OperationsError

JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
RunnerJsonCommand: TypeAlias = Callable[..., JsonObject]


class _DictBackedResult(Protocol):
    __dict__: dict[str, object]


DataclassResultCommand: TypeAlias = Callable[..., _DictBackedResult]

RunnerError = _runner.RunnerError
admit_lanes = cast(DataclassResultCommand, _runner.admit_lanes)
benchmark_model = cast(DataclassResultCommand, _runner.benchmark_model)
finalize_batch = cast(RunnerJsonCommand, _runner.finalize_batch)
refinalize_batch_v2 = cast(RunnerJsonCommand, _runner.refinalize_batch_v2)
launch = cast(RunnerJsonCommand, _runner.launch)
materialize_data = cast(RunnerJsonCommand, _runner.materialize_data)
run_model = cast(RunnerJsonCommand, _runner.run_model)
status = cast(RunnerJsonCommand, _runner.status)
toy_smoke = cast(DataclassResultCommand, _runner.toy_smoke)
validate_config = cast(RunnerJsonCommand, _runner.validate_config)
validate_model = cast(RunnerJsonCommand, _runner.validate_model)


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _common_paths(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--config", default="rebuttal/configs/base.toml")
    _ = parser.add_argument("--sweep", default="rebuttal/configs/sweep.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m si_rebuttal")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config_parser = subparsers.add_parser("validate-config")
    _common_paths(validate_config_parser)

    materialize_parser = subparsers.add_parser("materialize-data")
    _common_paths(materialize_parser)
    _ = materialize_parser.add_argument("--run-root", required=True)

    toy_parser = subparsers.add_parser("toy-smoke")
    _common_paths(toy_parser)
    _ = toy_parser.add_argument("--run-root", required=True)

    validate_model_parser = subparsers.add_parser("validate-model")
    _common_paths(validate_model_parser)
    _ = validate_model_parser.add_argument("--model", required=True)
    _ = validate_model_parser.add_argument("--run-root", required=True)

    benchmark_parser = subparsers.add_parser("benchmark-model")
    _common_paths(benchmark_parser)
    _ = benchmark_parser.add_argument("--model", required=True)
    _ = benchmark_parser.add_argument("--run-root", required=True)

    lane_parser = subparsers.add_parser("admit-lanes")
    _common_paths(lane_parser)
    _ = lane_parser.add_argument("--run-root", required=True)

    run_model_parser = subparsers.add_parser("run-model")
    _common_paths(run_model_parser)
    _ = run_model_parser.add_argument("--run-root", required=True)
    _ = run_model_parser.add_argument("--models", required=True)
    mode = run_model_parser.add_mutually_exclusive_group(required=True)
    _ = mode.add_argument("--dry-run", action="store_true")
    _ = mode.add_argument("--execute", action="store_true")
    _ = run_model_parser.add_argument("--batch-id")

    finalize_parser = subparsers.add_parser("finalize-batch")
    _common_paths(finalize_parser)
    _ = finalize_parser.add_argument("--run-root", required=True)

    refinalize_parser = subparsers.add_parser("refinalize-batch-v2")
    _common_paths(refinalize_parser)
    _ = refinalize_parser.add_argument("--run-root", required=True)
    _ = refinalize_parser.add_argument("--source-summary-v1", required=True)

    launch_parser = subparsers.add_parser("launch")
    _common_paths(launch_parser)
    _ = launch_parser.add_argument("--run-root", required=True)
    _ = launch_parser.add_argument("--batch-id", required=True)
    launch_mode = launch_parser.add_mutually_exclusive_group(required=True)
    _ = launch_mode.add_argument("--dry-run", action="store_true")
    _ = launch_mode.add_argument("--execute", action="store_true")

    status_parser = subparsers.add_parser("status")
    _ = status_parser.add_argument("--run-root", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload: object
        if args.command == "validate-config":
            payload = validate_config(args.config, args.sweep)
        elif args.command == "materialize-data":
            payload = materialize_data(args.config, args.sweep, run_root=Path(args.run_root))
        elif args.command == "toy-smoke":
            payload = dict(vars(toy_smoke(args.config, args.sweep, run_root=Path(args.run_root))))
        elif args.command == "validate-model":
            payload = validate_model(
                args.config, args.sweep, model_name=args.model, run_root=Path(args.run_root)
            )
        elif args.command == "benchmark-model":
            payload = dict(
                vars(
                    benchmark_model(
                        args.config, args.sweep, model_name=args.model, run_root=Path(args.run_root)
                    )
                )
            )
        elif args.command == "admit-lanes":
            payload = dict(vars(admit_lanes(args.config, args.sweep, run_root=Path(args.run_root))))
        elif args.command == "run-model":
            payload = run_model(
                args.config,
                args.sweep,
                run_root=Path(args.run_root),
                model_names=tuple(token for token in args.models.split(",") if token),
                execute=bool(args.execute),
            )
        elif args.command == "finalize-batch":
            payload = finalize_batch(args.config, args.sweep, run_root=Path(args.run_root))
        elif args.command == "refinalize-batch-v2":
            payload = refinalize_batch_v2(
                args.config,
                args.sweep,
                run_root=Path(args.run_root),
                source_summary_v1=Path(args.source_summary_v1),
            )
        elif args.command == "launch":
            payload = launch(
                args.config,
                args.sweep,
                run_root=Path(args.run_root),
                batch_id=args.batch_id,
                execute=bool(args.execute),
            )
        elif args.command == "status":
            payload = status(run_root=Path(args.run_root))
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (RunnerError, OperationsError, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=_json_default,
        )
    )
    return 0
