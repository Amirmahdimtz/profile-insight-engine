from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.benchmark import load_manifest
from evaluation.common import (
    DatasetSplit,
    EvaluationLabelType,
    deterministic_json,
)
from evaluation.contracts import (
    EvaluationDatasetManifest,
    EvaluationSample,
)


REPORT_SCHEMA_VERSION = "phase10-holdout-dataset-v1"
SPLIT_POLICY = "sha256_dataset_id_sample_id_lowest_rank"
_VISUAL_LABEL_TYPES = (
    EvaluationLabelType.SCENE,
    EvaluationLabelType.OBJECT,
    EvaluationLabelType.ACTIVITY,
    EvaluationLabelType.TOPIC,
)


def _rank_key(dataset_id: str, sample_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{dataset_id}\n{sample_id}".encode("utf-8")
    ).hexdigest()
    return digest, sample_id


def _visual_value(label: Any) -> str:
    value = getattr(label, "value", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return label.label


def derive_holdout_manifest(
    manifest: EvaluationDatasetManifest,
    *,
    dataset_version: str,
    train_count: int,
) -> EvaluationDatasetManifest:
    if not isinstance(manifest, EvaluationDatasetManifest):
        raise ValueError("manifest must be an EvaluationDatasetManifest")
    if isinstance(train_count, bool) or not isinstance(train_count, int):
        raise ValueError("train_count must be an integer")
    if train_count <= 0 or train_count >= len(manifest.samples):
        raise ValueError(
            "train_count must leave at least one TRAIN and one EVAL sample"
        )
    if any(
        sample.split is DatasetSplit.TRAIN
        for sample in manifest.samples
    ):
        raise ValueError(
            "source manifest already contains TRAIN samples; "
            "refuse to reshuffle an existing split"
        )

    ranked = sorted(
        manifest.samples,
        key=lambda sample: _rank_key(
            manifest.dataset_id,
            sample.sample_id,
        ),
    )
    train_ids = {
        sample.sample_id for sample in ranked[:train_count]
    }
    samples = tuple(
        EvaluationSample(
            sample_id=sample.sample_id,
            split=(
                DatasetSplit.TRAIN
                if sample.sample_id in train_ids
                else DatasetSplit.EVAL
            ),
            slices=sample.slices,
            image=sample.image,
            ground_truth=sample.ground_truth,
        )
        for sample in manifest.samples
    )
    return EvaluationDatasetManifest(
        schema_version=manifest.schema_version,
        dataset_id=manifest.dataset_id,
        dataset_version=dataset_version,
        label_schema_version=manifest.label_schema_version,
        data_policy=manifest.data_policy,
        samples=samples,
    )


def _train_taxonomy(
    manifest: EvaluationDatasetManifest,
) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {
        item.value: set() for item in _VISUAL_LABEL_TYPES
    }
    for sample in manifest.samples:
        if sample.split is not DatasetSplit.TRAIN:
            continue
        for label in sample.ground_truth.labels:
            if label.type in _VISUAL_LABEL_TYPES:
                values[label.type.value].add(_visual_value(label))
    return {
        label_type.value: tuple(
            sorted(
                values[label_type.value],
                key=lambda value: (value.casefold(), value),
            )
        )
        for label_type in _VISUAL_LABEL_TYPES
    }


def build_holdout_report(
    *,
    source_manifest: EvaluationDatasetManifest,
    derived_manifest: EvaluationDatasetManifest,
    source_manifest_fingerprint: str,
    dataset_content_fingerprint: str,
) -> Mapping[str, Any]:
    taxonomy = _train_taxonomy(derived_manifest)
    train_ids = tuple(
        sample.sample_id
        for sample in derived_manifest.samples
        if sample.split is DatasetSplit.TRAIN
    )
    eval_ids = tuple(
        sample.sample_id
        for sample in derived_manifest.samples
        if sample.split is DatasetSplit.EVAL
    )
    visual_label_counts = {
        label_type: len(values)
        for label_type, values in taxonomy.items()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "split_policy": SPLIT_POLICY,
        "source_dataset_id": source_manifest.dataset_id,
        "source_dataset_version": source_manifest.dataset_version,
        "derived_dataset_version": derived_manifest.dataset_version,
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "derived_manifest_fingerprint": derived_manifest.fingerprint(),
        "dataset_content_fingerprint": dataset_content_fingerprint,
        "train_sample_count": len(train_ids),
        "eval_sample_count": len(eval_ids),
        "train_sample_ids": list(train_ids),
        "eval_sample_ids": list(eval_ids),
        "train_visual_taxonomy": {
            key: list(values)
            for key, values in taxonomy.items()
        },
        "train_visual_taxonomy_counts": visual_label_counts,
        "train_visual_taxonomy_total": sum(
            visual_label_counts.values()
        ),
        "selection_notes": [
            "TRAIN membership is label-blind and determined only by "
            "dataset_id/sample_id SHA-256 rank.",
            "No EVAL label is used to choose TRAIN membership.",
            "The derived manifest is a new version; the source manifest "
            "is not modified.",
        ],
    }


def prepare_holdout_dataset(
    *,
    manifest_path: Path,
    dataset_root: Path,
    dataset_version: str,
    train_count: int,
    output_manifest: Path,
    output_report: Path,
    expected_source_manifest_fingerprint: str | None,
    expected_dataset_content_fingerprint: str | None,
) -> Mapping[str, Any]:
    source_manifest = load_manifest(manifest_path)
    source_fingerprint = source_manifest.fingerprint()
    if (
        expected_source_manifest_fingerprint is not None
        and source_fingerprint
        != expected_source_manifest_fingerprint
    ):
        raise ValueError(
            "source manifest fingerprint does not match expected value"
        )

    source_content_fingerprint = (
        source_manifest.validate_references(dataset_root)
    )
    if (
        expected_dataset_content_fingerprint is not None
        and source_content_fingerprint
        != expected_dataset_content_fingerprint
    ):
        raise ValueError(
            "dataset content fingerprint does not match expected value"
        )

    derived_manifest = derive_holdout_manifest(
        source_manifest,
        dataset_version=dataset_version,
        train_count=train_count,
    )
    derived_content_fingerprint = (
        derived_manifest.validate_references(dataset_root)
    )
    if derived_content_fingerprint != source_content_fingerprint:
        raise ValueError(
            "derived split changed dataset content fingerprint"
        )

    report = build_holdout_report(
        source_manifest=source_manifest,
        derived_manifest=derived_manifest,
        source_manifest_fingerprint=source_fingerprint,
        dataset_content_fingerprint=derived_content_fingerprint,
    )
    if report["train_visual_taxonomy_total"] <= 0:
        raise ValueError(
            "derived TRAIN split contains no visual taxonomy labels"
        )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        derived_manifest.to_json() + "\n",
        encoding="utf-8",
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        deterministic_json(report) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic label-blind TRAIN/EVAL holdout "
            "manifest for Phase 10 final Vision evaluation."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument(
        "--expected-source-manifest-fingerprint",
    )
    parser.add_argument(
        "--expected-dataset-content-fingerprint",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = prepare_holdout_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        dataset_version=args.dataset_version,
        train_count=args.train_count,
        output_manifest=args.output_manifest,
        output_report=args.output_report,
        expected_source_manifest_fingerprint=(
            args.expected_source_manifest_fingerprint
        ),
        expected_dataset_content_fingerprint=(
            args.expected_dataset_content_fingerprint
        ),
    )
    print(deterministic_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
