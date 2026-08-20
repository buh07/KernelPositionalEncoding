from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

FIXTURE_PATH = Path("rebuttal/fixtures/wikitext_selected_overlap_v1.json")


class DatasetPayload(TypedDict):
    repository: str
    revision: str
    config: str
    split: str
    fingerprint: str
    row_count: int


class DocumentsPayload(TypedDict):
    total_count: int
    unique_content_hash_count: int
    cross_partition_content_hash_count: int


class ModelPayload(TypedDict):
    tokenizer_tree_sha256: str
    selected_fit_chunk_count: int
    selected_eval_chunk_count: int
    selected_document_id_overlap_count: int
    selected_document_content_overlap_count: int
    selected_token_hash_overlap_count: int
    selected_fit_document_ids: list[str]
    selected_eval_document_ids: list[str]


class SelectedUnionPayload(TypedDict):
    document_id_overlap_count: int
    content_hash_overlap_count: int
    source_row_overlap_count: int
    fit_document_ids: list[str]
    eval_document_ids: list[str]


class OverlapDiagnosticPayload(TypedDict):
    schema_version: int
    dataset: DatasetPayload
    documents: DocumentsPayload
    models: dict[str, ModelPayload]
    selected_union: SelectedUnionPayload


EXPECTED_DATASET: DatasetPayload = {
    "repository": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "config": "wikitext-103-raw-v1",
    "split": "train",
    "fingerprint": "7dabb830ac9ebb0d",
    "row_count": 1801350,
}

EXPECTED_DOCUMENTS: DocumentsPayload = {
    "total_count": 29443,
    "unique_content_hash_count": 29001,
    "cross_partition_content_hash_count": 146,
}

EXPECTED_MODELS: dict[str, ModelPayload] = {
    "llama-3.1-8b": {
        "tokenizer_tree_sha256": (
            "afe108ea6ee025b12beeb9fdc495ae6fd6d0936498cdd8167cfa9a08e38db952"
        ),
        "selected_fit_chunk_count": 50,
        "selected_eval_chunk_count": 100,
        "selected_document_id_overlap_count": 0,
        "selected_document_content_overlap_count": 0,
        "selected_token_hash_overlap_count": 0,
        "selected_fit_document_ids": [],
        "selected_eval_document_ids": [],
    },
    "mistral-7b-v0.1": {
        "tokenizer_tree_sha256": (
            "990bfba0e432465dfcbdfc82c6844925718ea69bd055ebff790d02d247693b2f"
        ),
        "selected_fit_chunk_count": 50,
        "selected_eval_chunk_count": 100,
        "selected_document_id_overlap_count": 0,
        "selected_document_content_overlap_count": 0,
        "selected_token_hash_overlap_count": 0,
        "selected_fit_document_ids": [],
        "selected_eval_document_ids": [],
    },
    "olmo-2-7b": {
        "tokenizer_tree_sha256": (
            "1fc4aeb0efa51200ae514a08d64ffcd6cb468ff787e798aff1b384f1378a8105"
        ),
        "selected_fit_chunk_count": 50,
        "selected_eval_chunk_count": 100,
        "selected_document_id_overlap_count": 0,
        "selected_document_content_overlap_count": 0,
        "selected_token_hash_overlap_count": 0,
        "selected_fit_document_ids": [],
        "selected_eval_document_ids": [],
    },
}


def _expect_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _expect_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise AssertionError(f"{context} must be a string")
    return value


def _expect_int(value: object, *, context: str) -> int:
    if not isinstance(value, int):
        raise AssertionError(f"{context} must be an integer")
    return value


def _expect_str_list(value: object, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise AssertionError(f"{context} must be a list")
    items = cast(list[object], value)
    result: list[str] = []
    for index, item in enumerate(items):
        result.append(_expect_str(item, context=f"{context}[{index}]"))
    return result


def _parse_dataset_payload(value: object) -> DatasetPayload:
    payload = _expect_object(value, context="dataset")
    return {
        "repository": _expect_str(payload["repository"], context="dataset.repository"),
        "revision": _expect_str(payload["revision"], context="dataset.revision"),
        "config": _expect_str(payload["config"], context="dataset.config"),
        "split": _expect_str(payload["split"], context="dataset.split"),
        "fingerprint": _expect_str(
            payload["fingerprint"],
            context="dataset.fingerprint",
        ),
        "row_count": _expect_int(payload["row_count"], context="dataset.row_count"),
    }


def _parse_documents_payload(value: object) -> DocumentsPayload:
    payload = _expect_object(value, context="documents")
    return {
        "total_count": _expect_int(
            payload["total_count"],
            context="documents.total_count",
        ),
        "unique_content_hash_count": _expect_int(
            payload["unique_content_hash_count"],
            context="documents.unique_content_hash_count",
        ),
        "cross_partition_content_hash_count": _expect_int(
            payload["cross_partition_content_hash_count"],
            context="documents.cross_partition_content_hash_count",
        ),
    }


def _parse_model_payload(value: object, *, model_name: str) -> ModelPayload:
    payload = _expect_object(value, context=f"models.{model_name}")
    return {
        "tokenizer_tree_sha256": _expect_str(
            payload["tokenizer_tree_sha256"],
            context=f"models.{model_name}.tokenizer_tree_sha256",
        ),
        "selected_fit_chunk_count": _expect_int(
            payload["selected_fit_chunk_count"],
            context=f"models.{model_name}.selected_fit_chunk_count",
        ),
        "selected_eval_chunk_count": _expect_int(
            payload["selected_eval_chunk_count"],
            context=f"models.{model_name}.selected_eval_chunk_count",
        ),
        "selected_document_id_overlap_count": _expect_int(
            payload["selected_document_id_overlap_count"],
            context=f"models.{model_name}.selected_document_id_overlap_count",
        ),
        "selected_document_content_overlap_count": _expect_int(
            payload["selected_document_content_overlap_count"],
            context=f"models.{model_name}.selected_document_content_overlap_count",
        ),
        "selected_token_hash_overlap_count": _expect_int(
            payload["selected_token_hash_overlap_count"],
            context=f"models.{model_name}.selected_token_hash_overlap_count",
        ),
        "selected_fit_document_ids": _expect_str_list(
            payload["selected_fit_document_ids"],
            context=f"models.{model_name}.selected_fit_document_ids",
        ),
        "selected_eval_document_ids": _expect_str_list(
            payload["selected_eval_document_ids"],
            context=f"models.{model_name}.selected_eval_document_ids",
        ),
    }


def _parse_selected_union_payload(value: object) -> SelectedUnionPayload:
    payload = _expect_object(value, context="selected_union")
    return {
        "document_id_overlap_count": _expect_int(
            payload["document_id_overlap_count"],
            context="selected_union.document_id_overlap_count",
        ),
        "content_hash_overlap_count": _expect_int(
            payload["content_hash_overlap_count"],
            context="selected_union.content_hash_overlap_count",
        ),
        "source_row_overlap_count": _expect_int(
            payload["source_row_overlap_count"],
            context="selected_union.source_row_overlap_count",
        ),
        "fit_document_ids": _expect_str_list(
            payload["fit_document_ids"],
            context="selected_union.fit_document_ids",
        ),
        "eval_document_ids": _expect_str_list(
            payload["eval_document_ids"],
            context="selected_union.eval_document_ids",
        ),
    }


def _load_fixture() -> OverlapDiagnosticPayload:
    payload = _expect_object(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
        context=str(FIXTURE_PATH),
    )
    models_object = _expect_object(payload["models"], context="models")
    return {
        "schema_version": _expect_int(
            payload["schema_version"],
            context="schema_version",
        ),
        "dataset": _parse_dataset_payload(payload["dataset"]),
        "documents": _parse_documents_payload(payload["documents"]),
        "models": {
            model_name: _parse_model_payload(model_payload, model_name=model_name)
            for model_name, model_payload in models_object.items()
        },
        "selected_union": _parse_selected_union_payload(payload["selected_union"]),
    }


def _assert_nonempty_unique(ids: list[str]) -> None:
    assert ids
    assert len(ids) == len(set(ids))


def test_selected_overlap_diagnostic_fixture_matches_expected_frozen_values() -> None:
    payload = _load_fixture()

    assert payload["schema_version"] == 1
    assert payload["dataset"] == EXPECTED_DATASET
    assert payload["documents"] == EXPECTED_DOCUMENTS
    assert set(payload["models"]) == set(EXPECTED_MODELS)

    fit_union_from_models: set[str] = set()
    eval_union_from_models: set[str] = set()

    for model_name, expected in EXPECTED_MODELS.items():
        model_payload = payload["models"][model_name]
        assert model_payload["tokenizer_tree_sha256"] == expected["tokenizer_tree_sha256"]
        assert model_payload["selected_fit_chunk_count"] == expected["selected_fit_chunk_count"]
        assert model_payload["selected_eval_chunk_count"] == expected["selected_eval_chunk_count"]
        assert (
            model_payload["selected_document_id_overlap_count"]
            == expected["selected_document_id_overlap_count"]
        )
        assert (
            model_payload["selected_document_content_overlap_count"]
            == expected["selected_document_content_overlap_count"]
        )
        assert (
            model_payload["selected_token_hash_overlap_count"]
            == expected["selected_token_hash_overlap_count"]
        )

        fit_ids = model_payload["selected_fit_document_ids"]
        eval_ids = model_payload["selected_eval_document_ids"]
        _assert_nonempty_unique(fit_ids)
        _assert_nonempty_unique(eval_ids)
        fit_union_from_models.update(fit_ids)
        eval_union_from_models.update(eval_ids)

    selected_union = payload["selected_union"]
    assert selected_union["document_id_overlap_count"] == 0
    assert selected_union["content_hash_overlap_count"] == 0
    assert selected_union["source_row_overlap_count"] == 0

    fit_union_ids = selected_union["fit_document_ids"]
    eval_union_ids = selected_union["eval_document_ids"]
    _assert_nonempty_unique(fit_union_ids)
    _assert_nonempty_unique(eval_union_ids)
    assert set(fit_union_ids) == fit_union_from_models
    assert set(eval_union_ids) == eval_union_from_models
