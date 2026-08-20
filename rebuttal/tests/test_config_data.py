from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import TypeGuard

import pytest

from si_rebuttal.config import (
    FIXED_COUNTS,
    FIXED_DATASETS,
    FIXED_MODELS,
    FIXED_SEED_NAMESPACE,
    FIXED_VALIDATION,
    ConfigError,
    load_base_config,
    load_sweep_config,
)
from si_rebuttal.data import (
    DataMaterializationError,
    ImmutableHFDatasetLoader,
    build_code_documents,
    build_wikipedia_documents,
    materialize_domain,
    materialize_frozen_domain,
    token_sha256,
)
from si_rebuttal.provenance import (
    DatasetBinding,
    GitBinding,
    ModelBinding,
    RuntimeBinding,
    TokenizerBinding,
    atomic_write_json,
    build_run_provenance,
    canonical_json_bytes,
    capture_git_binding,
    compact_json_seed,
    sha256_bytes,
    sha256_tree,
)


def _tokenize_by_word(text: str) -> Sequence[int]:
    return tuple(
        int.from_bytes(sha256(word.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
        for word in text.split()
    )


def _config_env(tmp_path: Path) -> dict[str, str]:
    env_paths = {
        "SI_REBUTTAL_RUNS_ROOT": "runs",
        "SI_REBUTTAL_DATA_ROOT": "data/materialized",
        "SI_REBUTTAL_MODELS_ROOT": str((tmp_path / "models").resolve()),
        "SI_REBUTTAL_LOGS_ROOT": "logs",
        "SI_REBUTTAL_CACHE_ROOT": "cache",
        "SI_REBUTTAL_MODEL_LLAMA_3_1_8B": tmp_path / "models" / "llama",
        "SI_REBUTTAL_TOKENIZER_LLAMA_3_1_8B": tmp_path / "tokenizers" / "llama",
        "SI_REBUTTAL_MODEL_MISTRAL_7B_V0_1": tmp_path / "models" / "mistral",
        "SI_REBUTTAL_TOKENIZER_MISTRAL_7B_V0_1": tmp_path / "tokenizers" / "mistral",
        "SI_REBUTTAL_MODEL_OLMO_2_7B": tmp_path / "models" / "olmo",
        "SI_REBUTTAL_TOKENIZER_OLMO_2_7B": tmp_path / "tokenizers" / "olmo",
    }
    for path in env_paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return {key: str(value) for key, value in env_paths.items()}


def _partition_bucket(dataset_revision: str, document_id: str) -> int:
    import hashlib

    digest = hashlib.sha256(f"29039|{dataset_revision}|{document_id}".encode()).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % 3


def _find_doc_ids(dataset_revision: str, make_document_id: Callable[[int], str]) -> tuple[str, str]:
    fit_id: str | None = None
    eval_id: str | None = None
    for index in range(1, 200):
        candidate = make_document_id(index)
        bucket = _partition_bucket(dataset_revision, candidate)
        if bucket == 0 and fit_id is None:
            fit_id = candidate
        if bucket != 0 and eval_id is None:
            eval_id = candidate
        if fit_id is not None and eval_id is not None:
            return fit_id, eval_id
    raise AssertionError("Unable to find fit/eval document IDs for the test fixture")


def _require_object_mapping(value: object) -> dict[str, object]:
    if not _is_object_mapping(value):
        raise TypeError(f"Expected object mapping, found {type(value).__name__}")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        key_object: object = key
        item_object: object = item
        if not isinstance(key_object, str):
            raise TypeError(f"Expected string key, found {type(key_object).__name__}")
        normalized[key_object] = item_object
    return normalized


def _require_object_list(value: object) -> list[object]:
    if not _is_object_sequence(value):
        raise TypeError(f"Expected object list, found {type(value).__name__}")
    normalized: list[object] = []
    for item in value:
        item_object: object = item
        normalized.append(item_object)
    return normalized


def _is_object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def test_base_and_sweep_configs_match_adr_constants(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    base = load_base_config(Path("rebuttal/configs/base.toml"), env=env)
    sweep = load_sweep_config(Path("rebuttal/configs/sweep.toml"))

    assert base.schema_version == 1
    assert base.seed_namespace == FIXED_SEED_NAMESPACE
    assert base.counts["fit_sequences"] == FIXED_COUNTS["fit_sequences"]
    assert base.counts["eval_sequences"] == FIXED_COUNTS["eval_sequences"]
    assert base.counts["head_bins"] == FIXED_COUNTS["head_bins"]
    assert base.validation == FIXED_VALIDATION
    assert tuple(base.datasets) == ("wikipedia", "code")
    assert tuple(base.models) == tuple(FIXED_MODELS)
    assert sweep.directions == ("wikipedia_to_code", "code_to_wikipedia")
    assert sweep.controls == ("source_kernel", "target_kernel", "offset_permutation", "norm")


def test_config_drift_is_rejected(tmp_path: Path) -> None:
    base_text = Path("rebuttal/configs/base.toml").read_text(encoding="utf-8")
    drift_path = tmp_path / "base.toml"
    drift_path.write_text(
        base_text.replace("fit_sequences = 50", "fit_sequences = 49"), encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="drifted"):
        load_base_config(drift_path, env=_config_env(tmp_path))


def test_unknown_base_keys_and_mutable_root_escapes_are_rejected(tmp_path: Path) -> None:
    base_text = Path("rebuttal/configs/base.toml").read_text(encoding="utf-8")

    unknown_path = tmp_path / "unknown.toml"
    unknown_path.write_text(base_text + "\n[unexpected]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_base_config(unknown_path, env=_config_env(tmp_path))

    env = _config_env(tmp_path)
    env["SI_REBUTTAL_RUNS_ROOT"] = "../escape"
    with pytest.raises(ConfigError, match=r"may not contain '\.\.'|escapes rebuttal root"):
        load_base_config(Path("rebuttal/configs/base.toml"), env=env)

    env = _config_env(tmp_path)
    env["SI_REBUTTAL_DATA_ROOT"] = str((tmp_path / "abs-materialized").resolve())
    with pytest.raises(ConfigError, match="must be a relative path under"):
        load_base_config(Path("rebuttal/configs/base.toml"), env=env)


def test_wikipedia_documents_ignore_prefix_and_preserve_title_boundaries() -> None:
    rows = [
        "ignored preface",
        "",
        " = Alpha = ",
        "",
        "alpha words here",
        "",
        " = Beta = ",
        "beta words here",
        "",
        "beta more words",
        "",
    ]
    documents = build_wikipedia_documents(rows, FIXED_DATASETS["wikipedia"]["revision"])

    assert [document.document_id for document in documents] == ["wiki:2:5", "wiki:6:10"]
    assert documents[0].row_indices == (2, 4)
    assert documents[0].text == " = Alpha = \n\nalpha words here"
    assert documents[1].row_indices == (6, 7, 9)


def test_code_documents_group_by_repository_name_in_row_order() -> None:
    rows = [
        {"repository_name": "", "whole_func_string": "skip"},
        {"repository_name": "repo-a", "whole_func_string": "def a(): pass"},
        {"repository_name": "repo-b", "whole_func_string": ""},
        {"repository_name": "repo-a", "whole_func_string": "def b(): pass"},
        {"repository_name": "repo-b", "whole_func_string": "def c(): pass"},
    ]
    documents = build_code_documents(rows, FIXED_DATASETS["code"]["revision"])

    assert [document.document_id for document in documents] == ["code:repo-a", "code:repo-b"]
    assert documents[0].row_indices == (1, 3)
    assert documents[0].text == "def a(): pass\n\ndef b(): pass"
    assert documents[1].row_indices == (4,)


def test_materialize_domain_is_deterministic_and_uses_noncrossing_chunks(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]
    fit_repo, eval_repo = _find_doc_ids(dataset.revision, lambda index: f"code:repo-{index}")
    rows = [
        {
            "repository_name": fit_repo.removeprefix("code:"),
            "whole_func_string": " ".join(f"fit{i}" for i in range(700)),
        },
        {
            "repository_name": eval_repo.removeprefix("code:"),
            "whole_func_string": " ".join(f"eval{i}" for i in range(1300)),
        },
    ]

    materialized_a = materialize_domain(
        dataset=dataset,
        dataset_fingerprint="fingerprint-a",
        rows=rows,
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
        chunk_length=512,
        fit_count=1,
        eval_count=2,
    )
    materialized_b = materialize_domain(
        dataset=dataset,
        dataset_fingerprint="fingerprint-a",
        rows=rows,
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
        chunk_length=512,
        fit_count=1,
        eval_count=2,
    )

    assert materialized_a.manifest_sha256 == materialized_b.manifest_sha256
    assert [
        chunk.token_count for chunk in materialized_a.fit_chunks + materialized_a.eval_chunks
    ] == [512, 512, 512]
    assert all(
        chunk.token_shape == (512,)
        for chunk in materialized_a.fit_chunks + materialized_a.eval_chunks
    )
    assert len({chunk.token_sha256 for chunk in materialized_a.fit_chunks}) == 1
    assert len({chunk.token_sha256 for chunk in materialized_a.eval_chunks}) == 2


def test_token_hashes_are_canonical_little_endian_int64_with_shape() -> None:
    tokens = [1, -2, 3, 2**40]
    digest = token_sha256(tokens)

    import numpy as np

    little = np.asarray(tokens, dtype="<i8")
    native = np.asarray(tokens, dtype=np.int64)
    big = np.asarray(tokens, dtype=">i8")
    expected = sha256_bytes(
        canonical_json_bytes({"shape": [4]}) + b"\0" + little.tobytes(order="C")
    )

    assert digest == expected
    assert digest == token_sha256(native.tolist())
    assert digest == token_sha256(big.astype(np.int64).tolist())


def test_materialize_domain_rejects_shortage_and_overlap(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]
    rows = [
        {"repository_name": "repo-fit", "whole_func_string": " ".join(f"a{i}" for i in range(600))},
        {
            "repository_name": "repo-eval",
            "whole_func_string": " ".join(f"b{i}" for i in range(100)),
        },
    ]
    with pytest.raises(DataMaterializationError, match="Required 2 chunks"):
        materialize_domain(
            dataset=dataset,
            dataset_fingerprint="fingerprint-b",
            rows=rows,
            tokenizer_name="tokenizer",
            tokenizer_tree_sha256="tok-sha",
            encode=_tokenize_by_word,
            chunk_length=512,
            fit_count=2,
            eval_count=1,
        )

    fit_repo, eval_repo = _find_doc_ids(dataset.revision, lambda index: f"code:repo-{index}")
    overlapping_rows = [
        {
            "repository_name": fit_repo.removeprefix("code:"),
            "whole_func_string": " ".join(f"x{i}" for i in range(600)),
        },
        {
            "repository_name": eval_repo.removeprefix("code:"),
            "whole_func_string": " ".join(f"y{i}" for i in range(600)),
        },
    ]
    with pytest.raises(DataMaterializationError, match="token hash overlap"):
        materialize_domain(
            dataset=dataset,
            dataset_fingerprint="fingerprint-c",
            rows=overlapping_rows,
            tokenizer_name="tokenizer",
            tokenizer_tree_sha256="tok-sha",
            encode=lambda text: tuple(range(len(text.split()))),
            chunk_length=512,
            fit_count=1,
            eval_count=1,
        )


def test_materialize_domain_allows_unused_duplicate_content_and_preserves_inventory(
    tmp_path: Path,
) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]

    fit_selected, eval_selected = _find_doc_ids(
        dataset.revision, lambda index: f"code:repo-{index}"
    )
    fit_unused, eval_unused = _find_doc_ids(dataset.revision, lambda index: f"code:unused-{index}")
    duplicate_text = " ".join(f"dup{i}" for i in range(100))
    rows = [
        {
            "repository_name": fit_selected.removeprefix("code:"),
            "whole_func_string": " ".join(f"fit{i}" for i in range(600)),
        },
        {
            "repository_name": eval_selected.removeprefix("code:"),
            "whole_func_string": " ".join(f"eval{i}" for i in range(600)),
        },
        {
            "repository_name": fit_unused.removeprefix("code:"),
            "whole_func_string": duplicate_text,
        },
        {
            "repository_name": eval_unused.removeprefix("code:"),
            "whole_func_string": duplicate_text,
        },
    ]

    expected_documents = build_code_documents(
        rows,
        dataset.revision,
        text_field=dataset.field,
        repository_field=dataset.repository_field or "repository_name",
    )
    expected_fit_documents = tuple(
        document
        for document in sorted(
            (document for document in expected_documents if document.partition == "fit"),
            key=lambda document: (document.assignment_sha256, document.document_id),
        )
    )
    expected_eval_documents = tuple(
        document
        for document in sorted(
            (document for document in expected_documents if document.partition == "eval"),
            key=lambda document: (document.assignment_sha256, document.document_id),
        )
    )

    materialized = materialize_domain(
        dataset=dataset,
        dataset_fingerprint="fingerprint-unused-duplicates",
        rows=rows,
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
        chunk_length=512,
        fit_count=1,
        eval_count=1,
    )

    assert materialized.fit_documents == expected_fit_documents
    assert materialized.eval_documents == expected_eval_documents
    assert {chunk.document_id for chunk in materialized.fit_chunks} == {fit_selected}
    assert {chunk.document_id for chunk in materialized.eval_chunks} == {eval_selected}
    assert duplicate_text in {document.text for document in materialized.fit_documents}
    assert duplicate_text in {document.text for document in materialized.eval_documents}


def test_materialize_domain_rejects_selected_cross_partition_duplicate_content(
    tmp_path: Path,
) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]
    fit_repo, eval_repo = _find_doc_ids(dataset.revision, lambda index: f"code:selected-{index}")
    duplicate_text = " ".join(f"shared{i}" for i in range(600))
    rows = [
        {
            "repository_name": fit_repo.removeprefix("code:"),
            "whole_func_string": duplicate_text,
        },
        {
            "repository_name": eval_repo.removeprefix("code:"),
            "whole_func_string": duplicate_text,
        },
    ]

    with pytest.raises(DataMaterializationError, match="Joined-document content hash overlap"):
        materialize_domain(
            dataset=dataset,
            dataset_fingerprint="fingerprint-selected-duplicates",
            rows=rows,
            tokenizer_name="tokenizer",
            tokenizer_tree_sha256="tok-sha",
            encode=_tokenize_by_word,
            chunk_length=512,
            fit_count=1,
            eval_count=1,
        )


class _FakeLoadedDataset(list[dict[str, object]]):
    _fingerprint: object
    column_names: Sequence[object]

    def __init__(
        self, rows: Sequence[dict[str, object]], *, fingerprint: str, columns: Sequence[str]
    ) -> None:
        super().__init__(rows)
        self._fingerprint = fingerprint
        self.column_names = tuple(columns)


class _FakeDatasetLoader:
    def __init__(
        self,
        rows: Sequence[dict[str, object]],
        *,
        recorded: list[tuple[str, dict[str, object | bool]]] | None = None,
    ) -> None:
        self._rows = list(rows)
        self._recorded = recorded

    def __call__(self, path: str, /, **kwargs: object) -> _FakeLoadedDataset:
        name = kwargs.get("name")
        split = kwargs.get("split")
        revision = kwargs.get("revision")
        streaming = kwargs.get("streaming", False)
        if (
            not isinstance(name, str)
            or not isinstance(split, str)
            or not isinstance(revision, str)
            or not isinstance(streaming, bool)
        ):
            raise AssertionError(
                "loader must receive string name/split/revision and bool streaming"
            )
        if self._recorded is not None:
            self._recorded.append(
                (
                    path,
                    {
                        "name": name,
                        "split": split,
                        "revision": revision,
                        "streaming": streaming,
                    },
                )
            )
        return _FakeLoadedDataset(
            self._rows,
            fingerprint="hf-fingerprint",
            columns=("repository_name", "whole_func_string"),
        )


class _MissingColumnLoader:
    def __call__(self, path: str, /, **kwargs: object) -> _FakeLoadedDataset:
        del path, kwargs
        return _FakeLoadedDataset([{"wrong": "value"}], fingerprint="fp", columns=("wrong",))


class _NoFingerprintDataset(_FakeLoadedDataset):
    def __init__(self, rows: Sequence[dict[str, object]], *, columns: Sequence[str]) -> None:
        super().__init__(rows, fingerprint="", columns=columns)


class _MissingFingerprintLoader:
    def __call__(self, path: str, /, **kwargs: object) -> _NoFingerprintDataset:
        del path, kwargs
        return _NoFingerprintDataset([{"text": " = Title = "}], columns=("text",))


def test_immutable_loader_uses_exact_hf_arguments_and_routes_rows(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]
    recorded: list[tuple[str, dict[str, object | bool]]] = []

    fit_count = 0
    eval_count = 0
    rows: list[dict[str, object]] = []
    index = 0
    while fit_count < FIXED_COUNTS["fit_sequences"] or eval_count < FIXED_COUNTS["eval_sequences"]:
        repository_name = f"repo-{index}"
        bucket = _partition_bucket(dataset.revision, f"code:{repository_name}")
        if bucket == 0 and fit_count < FIXED_COUNTS["fit_sequences"]:
            fit_count += 1
        elif bucket != 0 and eval_count < FIXED_COUNTS["eval_sequences"]:
            eval_count += 1
        else:
            index += 1
            continue
        rows.append(
            {
                "repository_name": repository_name,
                "whole_func_string": " ".join(f"{repository_name}-{token}" for token in range(600)),
            }
        )
        index += 1

    materialized = materialize_frozen_domain(
        dataset=dataset,
        dataset_loader=ImmutableHFDatasetLoader(_FakeDatasetLoader(rows, recorded=recorded)),
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
    )

    assert recorded == [
        (
            dataset.repository,
            {
                "name": dataset.config,
                "split": dataset.split,
                "revision": dataset.revision,
                "streaming": False,
            },
        )
    ]
    assert materialized.dataset_fingerprint == "hf-fingerprint"
    assert len(materialized.fit_chunks) == FIXED_COUNTS["fit_sequences"]
    assert len(materialized.eval_chunks) == FIXED_COUNTS["eval_sequences"]


def test_immutable_loader_rejects_missing_columns_and_missing_fingerprint(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["wikipedia"]

    with pytest.raises(DataMaterializationError, match="missing required columns"):
        ImmutableHFDatasetLoader(_MissingColumnLoader()).load(dataset)

    with pytest.raises(DataMaterializationError, match="non-empty _fingerprint"):
        ImmutableHFDatasetLoader(_MissingFingerprintLoader()).load(dataset)


def test_materialize_frozen_domain_rejects_dataset_drift(tmp_path: Path) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["code"]
    drifted = type(dataset)(
        domain=dataset.domain,
        repository=dataset.repository,
        revision=dataset.revision,
        config=dataset.config,
        split=dataset.split,
        field="alternate_field",
        document_title_regex=dataset.document_title_regex,
        repository_field=dataset.repository_field,
    )

    with pytest.raises(DataMaterializationError, match="does not match frozen ADR snapshot"):
        materialize_frozen_domain(
            dataset=drifted,
            dataset_loader=ImmutableHFDatasetLoader(_FakeDatasetLoader(())),
            tokenizer_name="tokenizer",
            tokenizer_tree_sha256="tok-sha",
            encode=_tokenize_by_word,
        )


def test_compact_json_seed_and_canonical_json_are_deterministic() -> None:
    seed = compact_json_seed([29039, "llama-3.1-8b", "wikipedia_to_code", "bin", 0, 1, "norm"])
    assert seed == compact_json_seed(
        [29039, "llama-3.1-8b", "wikipedia_to_code", "bin", 0, 1, "norm"]
    )
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_atomic_write_json_and_tree_hash_capture_latest_state(tmp_path: Path) -> None:
    target = tmp_path / "prov.json"
    atomic_write_json(target, {"z": 1, "a": 2})
    assert target.read_text(encoding="ascii") == '{"a":2,"z":1}\n'
    first_hash = sha256_tree(tmp_path)

    atomic_write_json(target, {"a": 2, "z": 3})
    second_hash = sha256_tree(tmp_path)
    assert first_hash != second_hash
    assert not list(tmp_path.glob("*.tmp"))


def test_run_provenance_binds_manifest_identities() -> None:
    provenance = build_run_provenance(
        seed_namespace=FIXED_SEED_NAMESPACE,
        command=("python", "-m", "si_rebuttal.runner"),
        config_identity_sha256="config-sha",
        sweep_identity_sha256="sweep-sha",
        git=GitBinding(
            commit_sha="abc",
            tracked_diff_sha256="def",
            tracked_paths=("rebuttal/src",),
            relevant_content_sha256="ghi",
            untracked_paths=("PLAN.md",),
            untracked_sha256="jkl",
        ),
        datasets=(
            DatasetBinding(
                name="wikipedia",
                repository="Salesforce/wikitext",
                revision=FIXED_DATASETS["wikipedia"]["revision"],
                config="wikitext-103-raw-v1",
                split="train",
                field="text",
                fingerprint="fp",
                manifest_sha256="manifest",
            ),
        ),
        tokenizers=(
            TokenizerBinding(
                model_name="llama-3.1-8b", tokenizer_path=Path("/tok"), tokenizer_tree_sha256="tok"
            ),
        ),
        models=(
            ModelBinding(
                model_name="llama-3.1-8b",
                weights_path=Path("/weights"),
                weights_tree_sha256="weights",
            ),
        ),
        runtime=RuntimeBinding(
            python_version="3.10",
            platform="linux",
            package_versions={"torch": "2.7.0"},
            cuda_version=None,
            driver_version=None,
        ),
    )

    payload = _require_object_mapping(asdict(provenance))
    assert payload["schema_version"] == 1
    assert payload["seed_namespace"] == FIXED_SEED_NAMESPACE
    datasets = _require_object_list(payload["datasets"])
    if not datasets:
        raise AssertionError("datasets payload must be a non-empty JSON array")
    dataset_payload = _require_object_mapping(datasets[0])
    git_payload = _require_object_mapping(payload["git"])
    assert dataset_payload["manifest_sha256"] == "manifest"
    assert git_payload["untracked_sha256"] == "jkl"


def test_capture_git_binding_binds_head_tracked_diff_and_standard_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)

    tracked_file = repo / "rebuttal" / "src" / "tracked.py"
    tracked_file.parent.mkdir(parents=True)
    tracked_file.write_text("value = 1\n", encoding="utf-8")
    plan_file = repo / "PLAN.md"
    plan_file.write_text("plan v1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "rebuttal/src/tracked.py", "PLAN.md"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True)

    clean = capture_git_binding(repo, ("rebuttal/src", "PLAN.md"))
    assert clean.commit_sha
    assert clean.untracked_paths == ()

    tracked_file.write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "rebuttal/src/tracked.py"], check=True)
    dirty = capture_git_binding(repo, ("rebuttal/src", "PLAN.md"))
    assert dirty.tracked_diff_sha256 != clean.tracked_diff_sha256
    assert dirty.relevant_content_sha256 != clean.relevant_content_sha256

    untracked_file = repo / "rebuttal" / "src" / "scratch.txt"
    untracked_file.write_text("untracked bytes\n", encoding="utf-8")
    with_untracked = capture_git_binding(repo, ("rebuttal/src", "PLAN.md"))
    assert with_untracked.untracked_paths == ("rebuttal/src/scratch.txt",)
    assert with_untracked.untracked_sha256 != clean.untracked_sha256


def test_gitignore_targets_mutable_paths_only() -> None:
    gitignore = Path("rebuttal/.gitignore").read_text(encoding="utf-8").splitlines()
    assert "/runs/" in gitignore
    assert "/logs/" in gitignore
    assert "/cache/" in gitignore
    assert "/data/materialized/" in gitignore
    assert all(entry not in gitignore for entry in ("/src/", "/configs/", "/tests/"))


def _build_bounded_wikipedia_raw_rows(
    dataset_revision: str, *, fit_count: int, eval_count: int, max_documents: int = 512
) -> tuple[tuple[dict[str, str], ...], dict[str, dict[str, object]]]:
    rows: list[dict[str, str]] = []
    expected_by_id: dict[str, dict[str, object]] = {}
    fit_documents = 0
    eval_documents = 0
    for index in range(max_documents):
        start_row = len(rows)
        title = f" = Document {index} = \n" if index % 2 == 0 else f" = Document {index} = \r\n"
        section = f" == Section {index} == \n"
        body = " ".join(f"doc{index}_token{token}" for token in range(600))
        document_rows = (
            {"text": title},
            {"text": section},
            {"text": body},
        )
        rows.extend(document_rows)
        end_row = len(rows) - 1
        document_id = f"wiki:{start_row}:{end_row}"
        bucket = _partition_bucket(dataset_revision, document_id)
        partition = "fit" if bucket == 0 else "eval"
        if partition == "fit":
            fit_documents += 1
        else:
            eval_documents += 1
        text = "\n\n".join(row["text"] for row in document_rows)
        expected_by_id[document_id] = {
            "partition": partition,
            "row_indices": (start_row, start_row + 1, start_row + 2),
            "text": text,
            "content_sha256": sha256(text.encode("utf-8")).hexdigest(),
        }
        if fit_documents >= fit_count and eval_documents >= eval_count:
            return tuple(rows), expected_by_id
    raise AssertionError(
        f"Failed to reach {fit_count} fit and {eval_count} eval documents "
        f"within {max_documents} candidates"
    )


def test_wikipedia_documents_preserve_raw_title_rows_and_section_text() -> None:
    rows = (
        "ignored preface",
        " = Alpha = \n",
        "alpha body\rkeeps carriage return",
        " == Alpha Section == \n",
        "alpha tail\nkeeps embedded newline",
        " = Beta = \r\n",
        "beta body\rkeeps carriage return",
        " === Beta Section === \r\n",
        "beta tail\nkeeps embedded newline",
    )

    expected_alpha = "\n\n".join((rows[1], rows[2], rows[3], rows[4]))
    expected_beta = "\n\n".join((rows[5], rows[6], rows[7], rows[8]))
    documents = build_wikipedia_documents(rows, FIXED_DATASETS["wikipedia"]["revision"])

    assert [document.document_id for document in documents] == ["wiki:1:4", "wiki:5:8"]
    assert [document.row_indices for document in documents] == [(1, 2, 3, 4), (5, 6, 7, 8)]
    assert [document.text for document in documents] == [expected_alpha, expected_beta]
    assert [document.content_sha256 for document in documents] == [
        sha256(expected_alpha.encode("utf-8")).hexdigest(),
        sha256(expected_beta.encode("utf-8")).hexdigest(),
    ]


def test_materialize_domain_preserves_raw_wikipedia_rows_in_selected_chunks(
    tmp_path: Path,
) -> None:
    env = _config_env(tmp_path)
    dataset = load_base_config(Path("rebuttal/configs/base.toml"), env=env).datasets["wikipedia"]
    rows, expected_by_id = _build_bounded_wikipedia_raw_rows(
        dataset.revision, fit_count=50, eval_count=100
    )

    materialized_a = materialize_domain(
        dataset=dataset,
        dataset_fingerprint="raw-row-fingerprint",
        rows=rows,
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
        chunk_length=512,
        fit_count=50,
        eval_count=100,
    )
    materialized_b = materialize_domain(
        dataset=dataset,
        dataset_fingerprint="raw-row-fingerprint",
        rows=rows,
        tokenizer_name="tokenizer",
        tokenizer_tree_sha256="tok-sha",
        encode=_tokenize_by_word,
        chunk_length=512,
        fit_count=50,
        eval_count=100,
    )

    assert len(materialized_a.fit_chunks) == 50
    assert len(materialized_a.eval_chunks) == 100
    assert materialized_a.manifest_sha256 == materialized_b.manifest_sha256

    documents_by_id = {
        document.document_id: document
        for document in materialized_a.fit_documents + materialized_a.eval_documents
    }
    assert set(documents_by_id) == set(expected_by_id)

    for chunk in materialized_a.fit_chunks + materialized_a.eval_chunks:
        expected = expected_by_id[chunk.document_id]
        document = documents_by_id[chunk.document_id]
        assert document.partition == expected["partition"]
        assert document.row_indices == expected["row_indices"]
        assert document.text == expected["text"]
        assert document.content_sha256 == expected["content_sha256"]
        assert chunk.partition == expected["partition"]
        assert chunk.source_rows == expected["row_indices"]
