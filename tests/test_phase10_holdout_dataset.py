import unittest

from evaluation.common import (
    DATASET_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    DataPolicy,
    DatasetSlice,
    DatasetSplit,
    EvaluationLabelType,
)
from evaluation.contracts import (
    EvaluationDatasetManifest,
    EvaluationGroundTruth,
    EvaluationImageReference,
    EvaluationSample,
    ExpectedEvidenceLabel,
)
from evaluation.prepare_phase10_holdout import (
    SPLIT_POLICY,
    build_holdout_report,
    derive_holdout_manifest,
)


def _sample(
    index: int,
    *,
    value: str | None = None,
    split: DatasetSplit = DatasetSplit.EVAL,
) -> EvaluationSample:
    sample_id = f"sample_{index:02d}"
    return EvaluationSample(
        sample_id=sample_id,
        split=split,
        slices=(DatasetSlice.NO_TEXT,),
        image=EvaluationImageReference(
            image_id=f"img_{index:02d}",
            relative_path=f"images/{index:02d}.png",
        ),
        ground_truth=EvaluationGroundTruth(
            labels=(
                ExpectedEvidenceLabel(
                    label_id=f"label_{index:02d}",
                    type=EvaluationLabelType.OBJECT,
                    label="observable object",
                    value=value or f"object-{index:02d}",
                ),
            )
        ),
    )


def _manifest(samples) -> EvaluationDatasetManifest:
    return EvaluationDatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_id="phase10_holdout_test",
        dataset_version="1.0.0",
        label_schema_version=LABEL_SCHEMA_VERSION,
        data_policy=DataPolicy.CONSENTED_OR_AUTHORIZED,
        samples=tuple(samples),
    )


class Phase10HoldoutDatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_label_blind(self):
        source = _manifest(_sample(i) for i in range(1, 7))
        altered = _manifest(
            _sample(i, value=f"changed-{i:02d}")
            for i in range(1, 7)
        )

        first = derive_holdout_manifest(
            source,
            dataset_version="1.1.0",
            train_count=2,
        )
        reversed_source = _manifest(
            reversed(tuple(source.samples))
        )
        second = derive_holdout_manifest(
            reversed_source,
            dataset_version="1.1.0",
            train_count=2,
        )
        third = derive_holdout_manifest(
            altered,
            dataset_version="1.1.0",
            train_count=2,
        )

        def train_ids(manifest):
            return {
                sample.sample_id
                for sample in manifest.samples
                if sample.split is DatasetSplit.TRAIN
            }

        self.assertEqual(train_ids(first), train_ids(second))
        self.assertEqual(train_ids(first), train_ids(third))
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.dataset_version, "1.1.0")
        self.assertEqual(
            sum(
                sample.split is DatasetSplit.TRAIN
                for sample in first.samples
            ),
            2,
        )
        self.assertEqual(
            sum(
                sample.split is DatasetSplit.EVAL
                for sample in first.samples
            ),
            4,
        )

    def test_report_contains_train_taxonomy_only(self):
        source = _manifest(_sample(i) for i in range(1, 7))
        derived = derive_holdout_manifest(
            source,
            dataset_version="1.1.0",
            train_count=2,
        )
        report = build_holdout_report(
            source_manifest=source,
            derived_manifest=derived,
            source_manifest_fingerprint=source.fingerprint(),
            dataset_content_fingerprint="a" * 64,
        )

        train_values = {
            str(label.value)
            for sample in derived.samples
            if sample.split is DatasetSplit.TRAIN
            for label in sample.ground_truth.labels
        }
        eval_values = {
            str(label.value)
            for sample in derived.samples
            if sample.split is DatasetSplit.EVAL
            for label in sample.ground_truth.labels
        }
        reported = set(
            report["train_visual_taxonomy"]["object"]
        )

        self.assertEqual(report["split_policy"], SPLIT_POLICY)
        self.assertEqual(reported, train_values)
        self.assertTrue(reported.isdisjoint(eval_values))
        self.assertEqual(report["train_sample_count"], 2)
        self.assertEqual(report["eval_sample_count"], 4)

    def test_derived_manifest_requires_new_dataset_version(self):
        source = _manifest(_sample(i) for i in range(1, 4))
        with self.assertRaisesRegex(
            ValueError,
            "must differ from source",
        ):
            derive_holdout_manifest(
                source,
                dataset_version="1.0.0",
                train_count=1,
            )

    def test_existing_train_split_is_not_reshuffled(self):
        source = _manifest(
            (
                _sample(1, split=DatasetSplit.TRAIN),
                _sample(2),
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "already contains TRAIN",
        ):
            derive_holdout_manifest(
                source,
                dataset_version="1.1.0",
                train_count=1,
            )

    def test_train_count_must_leave_a_holdout(self):
        source = _manifest(_sample(i) for i in range(1, 4))
        for value in (0, 3):
            with self.subTest(train_count=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "at least one TRAIN and one EVAL",
                ):
                    derive_holdout_manifest(
                        source,
                        dataset_version="1.1.0",
                        train_count=value,
                    )


if __name__ == "__main__":
    unittest.main()
