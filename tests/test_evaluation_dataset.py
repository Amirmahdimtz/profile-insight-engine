import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.contracts import (
    DATASET_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    DataPolicy,
    DatasetSlice,
    DatasetSplit,
    EvaluationDatasetManifest,
    EvaluationGroundTruth,
    EvaluationImageReference,
    EvaluationLabelType,
    EvaluationSample,
    EvaluationValidationError,
    ExpectedEvidenceLabel,
    UnsupportedClaimAnnotation,
    UnsupportedClaimCategory,
)


class EvaluationDatasetContractTests(unittest.TestCase):
    def _sample(
        self,
        sample_id="sample_eval_1",
        image_id="img_eval_1",
        relative_path="images/eval_1.png",
        split=DatasetSplit.EVAL,
        slices=(DatasetSlice.PERSIAN_HEAVY,),
        label="football object",
    ):
        return EvaluationSample(
            sample_id=sample_id,
            split=split,
            slices=slices,
            image=EvaluationImageReference(image_id=image_id, relative_path=relative_path),
            ground_truth=EvaluationGroundTruth(
                labels=(
                    ExpectedEvidenceLabel(
                        label_id=f"label_{sample_id}",
                        type=EvaluationLabelType.OBJECT,
                        label=label,
                        value="football",
                    ),
                )
            ),
        )

    def _manifest(self, samples=None):
        return EvaluationDatasetManifest(
            schema_version=DATASET_SCHEMA_VERSION,
            dataset_id="dataset_1",
            dataset_version="1.2.3",
            label_schema_version=LABEL_SCHEMA_VERSION,
            data_policy=DataPolicy.CONSENTED_OR_AUTHORIZED,
            samples=tuple((self._sample(),) if samples is None else samples),
        )

    def test_valid_minimum_dataset(self):
        manifest = self._manifest()
        self.assertEqual(manifest.samples[0].split, DatasetSplit.EVAL)
        self.assertEqual(len(manifest.fingerprint()), 64)

    def test_topic_is_representable_in_evaluation_label_schema(self):
        self.assertEqual(EvaluationLabelType.TOPIC.value, "topic")

    def test_required_dataset_slices_are_representable(self):
        self.assertEqual(
            {item.value for item in DatasetSlice},
            {
                "persian_heavy",
                "english_heavy",
                "mixed_persian_english",
                "text_heavy",
                "low_quality",
                "screenshot",
                "meme",
                "group",
                "no_text",
                "indoor",
                "outdoor",
            },
        )

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(EvaluationValidationError):
            self._manifest(samples=())

    def test_empty_train_split_is_allowed_but_empty_eval_split_is_rejected(self):
        self._manifest(samples=(self._sample(),))
        train_only = self._sample(
            sample_id="sample_train_1",
            image_id="img_train_1",
            relative_path="images/train_1.png",
            split=DatasetSplit.TRAIN,
        )
        with self.assertRaises(EvaluationValidationError):
            self._manifest(samples=(train_only,))

    def test_malformed_dataset_version_is_rejected(self):
        with self.assertRaises(EvaluationValidationError):
            EvaluationDatasetManifest(
                schema_version=DATASET_SCHEMA_VERSION,
                dataset_id="dataset_1",
                dataset_version="v1",
                label_schema_version=LABEL_SCHEMA_VERSION,
                data_policy=DataPolicy.CONSENTED_OR_AUTHORIZED,
                samples=(self._sample(),),
            )

    def test_duplicate_sample_id_is_rejected(self):
        first = self._sample()
        second = self._sample(
            image_id="img_eval_2", relative_path="images/eval_2.png"
        )
        with self.assertRaisesRegex(EvaluationValidationError, "duplicate sample_id"):
            self._manifest(samples=(first, second))

    def test_duplicate_image_id_is_rejected(self):
        first = self._sample()
        second = self._sample(
            sample_id="sample_eval_2", relative_path="images/eval_2.png"
        )
        with self.assertRaisesRegex(EvaluationValidationError, "duplicate image_id"):
            self._manifest(samples=(first, second))

    def test_duplicate_image_reference_is_rejected(self):
        first = self._sample()
        second = self._sample(sample_id="sample_eval_2", image_id="img_eval_2")
        with self.assertRaisesRegex(
            EvaluationValidationError, "duplicate image reference"
        ):
            self._manifest(samples=(first, second))

    def test_train_eval_leakage_is_rejected(self):
        train = self._sample(
            sample_id="sample_train_1",
            split=DatasetSplit.TRAIN,
            image_id="shared_img",
            relative_path="images/train.png",
        )
        evaluation = self._sample(
            sample_id="sample_eval_2",
            split=DatasetSplit.EVAL,
            image_id="shared_img",
            relative_path="images/eval.png",
        )
        with self.assertRaisesRegex(EvaluationValidationError, "train/eval leakage"):
            self._manifest(samples=(train, evaluation))

    def test_invalid_split_is_rejected(self):
        payload = self._manifest().to_dict()
        payload["samples"][0]["split"] = "validation"
        with self.assertRaises(EvaluationValidationError):
            EvaluationDatasetManifest.from_dict(payload)

    def test_unknown_field_is_rejected(self):
        payload = self._manifest().to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(EvaluationValidationError, "unknown fields"):
            EvaluationDatasetManifest.from_dict(payload)

    def test_invalid_label_type_is_rejected(self):
        payload = self._manifest().to_dict()
        payload["samples"][0]["ground_truth"]["labels"][0]["type"] = "personality"
        with self.assertRaises(EvaluationValidationError):
            EvaluationDatasetManifest.from_dict(payload)

    def test_sensitive_person_trait_label_is_rejected_via_phase1_policy(self):
        with self.assertRaises(EvaluationValidationError):
            self._sample(label="Person religion = example")

    def test_persian_sensitive_person_trait_label_is_rejected_via_phase1_policy(self):
        with self.assertRaises(EvaluationValidationError):
            self._sample(label="مذهب شخص = نمونه")

    def test_religious_content_observation_is_allowed(self):
        label = ExpectedEvidenceLabel(
            label_id="religious_content_1",
            type=EvaluationLabelType.RELIGIOUS_CONTENT,
            label="religious-themed content present",
        )
        self.assertEqual(label.type, EvaluationLabelType.RELIGIOUS_CONTENT)

    def test_invalid_unsupported_claim_annotation_is_rejected(self):
        payload = {
            "unsupported_claims": [{"category": "medical_diagnosis"}],
        }
        with self.assertRaises(EvaluationValidationError):
            EvaluationGroundTruth.from_dict(payload)

    def test_valid_unsupported_claim_annotation_is_negative_policy_metadata(self):
        ground_truth = EvaluationGroundTruth(
            unsupported_claims=(
                UnsupportedClaimAnnotation(UnsupportedClaimCategory.PERSON_RELIGION),
            )
        )
        self.assertEqual(
            ground_truth.unsupported_claims[0].category,
            UnsupportedClaimCategory.PERSON_RELIGION,
        )

    def test_duplicate_ground_truth_identity_is_rejected_even_with_different_label_ids(self):
        first = ExpectedEvidenceLabel(
            label_id="label_1",
            type=EvaluationLabelType.BRAND,
            label="brand reference",
            value="Nike",
        )
        second = ExpectedEvidenceLabel(
            label_id="label_2",
            type=EvaluationLabelType.BRAND,
            label="brand reference",
            value="Nike",
        )
        with self.assertRaisesRegex(EvaluationValidationError, "duplicate identities"):
            EvaluationGroundTruth(labels=(first, second))

    def test_non_finite_ground_truth_value_is_rejected(self):
        with self.assertRaises(EvaluationValidationError):
            ExpectedEvidenceLabel(
                label_id="label_1",
                type=EvaluationLabelType.OBJECT,
                label="observable object",
                value=math.nan,
            )

    def test_reference_path_must_be_portable_and_relative(self):
        for value in ("../secret.png", "/absolute.png", "folder\\image.png"):
            with self.subTest(value=value), self.assertRaises(EvaluationValidationError):
                EvaluationImageReference(image_id="img_1", relative_path=value)

    def test_reference_validation_hashes_bytes_without_decoding_images(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "images" / "eval_1.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not-an-image-and-that-is-intentional-phase2")
            first_fingerprint = manifest.validate_references(temp_dir)
            self.assertEqual(len(first_fingerprint), 64)

            path.write_bytes(b"same-reference-different-bytes")
            second_fingerprint = manifest.validate_references(temp_dir)
            self.assertNotEqual(first_fingerprint, second_fingerprint)

            path.unlink()
            with self.assertRaisesRegex(EvaluationValidationError, "does not exist"):
                manifest.validate_references(temp_dir)

    def test_content_duplicate_and_cross_split_leakage_are_detected(self):
        first = self._sample(
            sample_id="sample_eval_1",
            image_id="img_eval_1",
            relative_path="images/eval_1.png",
            split=DatasetSplit.EVAL,
        )
        second = self._sample(
            sample_id="sample_eval_2",
            image_id="img_eval_2",
            relative_path="images/eval_2.png",
            split=DatasetSplit.EVAL,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            (root / "images/eval_1.png").write_bytes(b"same-content")
            (root / "images/eval_2.png").write_bytes(b"same-content")
            with self.assertRaisesRegex(EvaluationValidationError, "duplicate image content"):
                self._manifest(samples=(first, second)).validate_references(root)

            train = self._sample(
                sample_id="sample_train_1",
                image_id="img_train_1",
                relative_path="images/train_1.png",
                split=DatasetSplit.TRAIN,
            )
            (root / "images/train_1.png").write_bytes(b"same-content")
            with self.assertRaisesRegex(EvaluationValidationError, "train/eval leakage"):
                self._manifest(samples=(train, first)).validate_references(root)

    def test_manifest_serialization_and_fingerprint_are_deterministic(self):
        first = self._sample(
            sample_id="sample_eval_b",
            image_id="img_b",
            relative_path="images/b.png",
            slices=(DatasetSlice.OUTDOOR, DatasetSlice.NO_TEXT),
        )
        second = self._sample(
            sample_id="sample_eval_a",
            image_id="img_a",
            relative_path="images/a.png",
            slices=(DatasetSlice.NO_TEXT, DatasetSlice.OUTDOOR),
        )
        manifest_1 = self._manifest(samples=(first, second))
        manifest_2 = self._manifest(samples=(second, first))
        self.assertEqual(manifest_1.to_json(), manifest_2.to_json())
        self.assertEqual(manifest_1.fingerprint(), manifest_2.fingerprint())
        self.assertEqual(
            manifest_1.to_json(),
            EvaluationDatasetManifest.from_json(manifest_1.to_json()).to_json(),
        )

    def test_json_boundary_rejects_malformed_json(self):
        with self.assertRaises(EvaluationValidationError):
            EvaluationDatasetManifest.from_json("{bad json")


if __name__ == "__main__":
    unittest.main()
