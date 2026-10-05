import concurrent.futures
import dataclasses
import unittest

from src.core.profile_analysis.aggregation_contracts import (
    AggregationKind,
    AggregationValidationError,
)
from src.core.profile_analysis.contracts import Evidence, EvidenceType
from src.core.profile_analysis.embedding_contracts import (
    NearDuplicateGroup,
    SemanticThemeResult,
    ThemeSimilarityEvidence,
)
from src.core.profile_analysis.image_contracts import (
    CanonicalImage,
    DuplicateImageMapping,
    ProcessedImageBatch,
)
from src.core.services.profile_analysis.evidence_aggregation_service import (
    EvidenceAggregationService,
)


def image(image_id, content_hash=None):
    digest = content_hash or ((image_id[0] if image_id else "a") * 64)
    return CanonicalImage(
        image_id=image_id,
        content_hash=digest,
        storage_reference=f"/tmp/{digest}.png",
        mime_type="image/png",
        width=10,
        height=10,
        size_bytes=10,
        original_mime_type="image/png",
        original_width=10,
        original_height=10,
        original_size_bytes=10,
        was_resized=False,
        exif_orientation_applied=False,
    )


def batch(*images, duplicates=()):
    return ProcessedImageBatch(
        analysis_id="analysis",
        images=images,
        duplicates=duplicates,
        storage_scope_reference="/tmp/analysis",
    )


def evidence(
    evidence_id,
    image_id,
    label="football",
    *,
    evidence_type=EvidenceType.TOPIC,
    value=None,
    confidence=0.8,
    source="vision",
    metadata=None,
):
    return Evidence(
        id=evidence_id,
        image_id=image_id,
        type=evidence_type,
        label=label,
        value=label if value is None else value,
        confidence=confidence,
        source=source,
        metadata={} if metadata is None else metadata,
    )


def semantic(*items, near_duplicates=(), provider="embedding"):
    return SemanticThemeResult(
        theme_similarities=items,
        retrieval_results=(),
        near_duplicate_groups=near_duplicates,
        provider=provider,
        provider_version="1",
        model_id="model",
        model_version="revision",
        config_version="config",
        embedding_dimension=2,
    )


class EvidenceAggregationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = EvidenceAggregationService()

    def test_single_image_aggregate_has_full_coverage_and_consistency(self):
        result = self.service.aggregate(
            batch(image("a")),
            (evidence("ev-1", "a", confidence=0.7),),
        )
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item.evidence_count, 1)
        self.assertEqual(item.unique_image_count, 1)
        self.assertEqual(item.image_universe_count, 1)
        self.assertEqual(item.image_coverage, 1.0)
        self.assertEqual(item.average_confidence, 0.7)
        self.assertEqual(item.max_confidence, 0.7)
        self.assertEqual(item.cross_image_consistency, 1.0)

    def test_many_images_source_diversity_confidence_and_consistency(self):
        result = self.service.aggregate(
            batch(image("a"), image("b")),
            (
                evidence("ev-1", "a", confidence=0.6, source="vision_a"),
                evidence("ev-2", "a", confidence=0.8, source="vision_b"),
                evidence("ev-3", "b", confidence=1.0, source="vision_a"),
            ),
        )[0]
        self.assertEqual(result.evidence_count, 3)
        self.assertEqual(result.unique_image_count, 2)
        self.assertEqual(result.image_coverage, 1.0)
        self.assertAlmostEqual(result.average_confidence, 0.8)
        self.assertEqual(result.max_confidence, 1.0)
        self.assertEqual(result.source_diversity, 2)
        self.assertEqual(result.sources, ("vision_a", "vision_b"))
        self.assertEqual(result.cross_image_consistency, 0.5)

    def test_duplicate_evidence_id_counts_once_but_conflict_is_rejected(self):
        item = evidence("ev-1", "a")
        result = self.service.aggregate(batch(image("a")), (item, item))[0]
        self.assertEqual(result.evidence_count, 1)

        with self.assertRaises(AggregationValidationError):
            self.service.aggregate(
                batch(image("a")),
                (item, evidence("ev-1", "a", confidence=0.9)),
            )

    def test_exact_duplicate_images_do_not_inflate_support_or_coverage_numerator(self):
        digest = "a" * 64
        first = image("a", digest)
        duplicate = image("b", digest)
        image_batch = batch(
            first,
            duplicate,
            duplicates=(DuplicateImageMapping("b", "a", digest),),
        )
        result = self.service.aggregate(
            image_batch,
            (
                evidence("ev-a", "a", confidence=0.8),
                evidence("ev-b", "b", confidence=0.91),
            ),
        )[0]
        self.assertEqual(result.evidence_count, 1)
        self.assertEqual(result.unique_image_count, 1)
        self.assertEqual(result.image_universe_count, 2)
        self.assertEqual(result.image_coverage, 0.5)
        self.assertEqual(result.supporting_image_ids, ("a",))
        self.assertEqual(result.supporting_signal_ids, ("ev-a",))
        self.assertEqual(result.average_confidence, 0.8)

    def test_duplicate_fallback_uses_one_duplicate_when_canonical_has_no_matching_support(self):
        digest = "a" * 64
        image_batch = batch(
            image("a", digest),
            image("b", digest),
            image("c", digest),
            duplicates=(
                DuplicateImageMapping("b", "a", digest),
                DuplicateImageMapping("c", "a", digest),
            ),
        )
        result = self.service.aggregate(
            image_batch,
            (
                evidence("ev-c", "c", confidence=0.9),
                evidence("ev-b", "b", confidence=0.7),
            ),
        )[0]
        self.assertEqual(result.evidence_count, 1)
        self.assertEqual(result.supporting_signal_ids, ("ev-b",))
        self.assertEqual(result.supporting_image_ids, ("a",))

    def test_repeated_support_inside_representative_image_is_preserved(self):
        result = self.service.aggregate(
            batch(image("a")),
            (
                evidence("ev-1", "a", metadata={"position": 1}),
                evidence("ev-2", "a", metadata={"position": 2}),
            ),
        )[0]
        self.assertEqual(result.evidence_count, 2)
        self.assertEqual(result.cross_image_consistency, 1.0)

    def test_coverage_denominator_is_all_batch_image_ids(self):
        result = self.service.aggregate(
            batch(image("a"), image("b"), image("c")),
            (
                evidence("ev-a", "a"),
                evidence("ev-c", "c"),
            ),
        )[0]
        self.assertEqual(result.unique_image_count, 2)
        self.assertEqual(result.image_universe_count, 3)
        self.assertAlmostEqual(result.image_coverage, 2 / 3)

    def test_namespace_prevents_same_text_across_evidence_types_from_merging(self):
        results = self.service.aggregate(
            batch(image("a")),
            (
                evidence("ev-object", "a", "Apple", evidence_type=EvidenceType.OBJECT),
                evidence("ev-brand", "a", "apple", evidence_type=EvidenceType.BRAND),
            ),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {(item.kind, item.key) for item in results},
            {(AggregationKind.OBJECT, "apple"), (AggregationKind.BRAND, "apple")},
        )

    def test_tie_break_for_case_variants_is_stable_and_input_order_independent(self):
        image_batch = batch(image("a"), image("b"))
        first = evidence("ev-a", "a", "football")
        second = evidence("ev-b", "b", "FOOTBALL")
        forward = self.service.aggregate(image_batch, (first, second))[0]
        reverse = self.service.aggregate(image_batch, (second, first))[0]
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.key, "football")
        self.assertEqual(forward.label, "FOOTBALL")

    def test_ocr_aggregates_by_normalized_value_not_generic_label(self):
        results = self.service.aggregate(
            batch(image("a")),
            (
                evidence(
                    "ev-nike",
                    "a",
                    "ocr_text",
                    evidence_type=EvidenceType.OCR_TEXT,
                    value="Nike",
                    source="ocr",
                ),
                evidence(
                    "ev-adidas",
                    "a",
                    "ocr_text",
                    evidence_type=EvidenceType.OCR_TEXT,
                    value="Adidas",
                    source="ocr",
                ),
            ),
        )
        self.assertEqual([item.key for item in results], ["adidas", "nike"])

    def test_semantic_match_is_support_but_similarity_is_not_confidence(self):
        semantic_result = semantic(
            ThemeSimilarityEvidence("a", "Travel", 0.8, 0.5),
            ThemeSimilarityEvidence("b", "Travel", 0.2, 0.5),
        )
        result = self.service.aggregate(
            batch(image("a"), image("b")), (), semantic_result
        )[0]
        self.assertEqual(result.kind, AggregationKind.SEMANTIC_THEME)
        self.assertEqual(result.evidence_count, 1)
        self.assertEqual(result.unique_image_count, 1)
        self.assertEqual(result.image_coverage, 0.5)
        self.assertIsNone(result.average_confidence)
        self.assertIsNone(result.max_confidence)
        self.assertEqual(result.sources, ("embedding",))

    def test_conflicting_duplicate_semantic_identity_is_rejected(self):
        semantic_result = semantic(
            ThemeSimilarityEvidence("a", "Travel", 0.8, 0.5),
            ThemeSimilarityEvidence("a", "Travel", 0.9, 0.5),
        )
        with self.assertRaises(AggregationValidationError):
            self.service.aggregate(batch(image("a")), (), semantic_result)

    def test_near_duplicate_group_does_not_collapse_distinct_content_ids(self):
        semantic_result = semantic(
            near_duplicates=(NearDuplicateGroup("near-1", ("a", "b")),)
        )
        result = self.service.aggregate(
            batch(image("a", "a" * 64), image("b", "b" * 64)),
            (evidence("ev-a", "a"), evidence("ev-b", "b")),
            semantic_result,
        )[0]
        self.assertEqual(result.evidence_count, 2)
        self.assertEqual(result.unique_image_count, 2)
        self.assertEqual(result.image_coverage, 1.0)

    def test_source_diversity_uses_standard_source_not_metadata_provider_field(self):
        result = self.service.aggregate(
            batch(image("a"), image("b")),
            (
                evidence("ev-a", "a", source="vision", metadata={"provider": "x"}),
                evidence("ev-b", "b", source="vision", metadata={"provider": "y"}),
            ),
        )[0]
        self.assertEqual(result.source_diversity, 1)
        self.assertEqual(result.sources, ("vision",))

    def test_missing_standard_source_is_rejected_defensively(self):
        valid = evidence("ev-a", "a")
        invalid = object.__new__(Evidence)
        for field in dataclasses.fields(Evidence):
            object.__setattr__(
                invalid,
                field.name,
                "" if field.name == "source" else getattr(valid, field.name),
            )
        with self.assertRaises(AggregationValidationError):
            self.service.aggregate(batch(image("a")), (invalid,))

    def test_input_order_and_concurrent_calls_are_deterministic(self):
        image_batch = batch(image("a"), image("b"))
        evidence_items = (
            evidence("ev-2", "b", "FOOTBALL", confidence=0.7),
            evidence("ev-1", "a", "football", confidence=0.9),
        )
        semantic_result = semantic(
            ThemeSimilarityEvidence("b", "Travel", 0.8, 0.5),
            ThemeSimilarityEvidence("a", "Travel", 0.8, 0.5),
        )
        expected = self.service.aggregate(image_batch, evidence_items, semantic_result)
        reversed_result = self.service.aggregate(
            image_batch,
            tuple(reversed(evidence_items)),
            semantic(*reversed(semantic_result.theme_similarities)),
        )
        self.assertEqual(expected, reversed_result)
        self.assertEqual([item.key for item in expected], ["travel", "football"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = tuple(
                executor.map(
                    lambda _: self.service.aggregate(
                        image_batch, evidence_items, semantic_result
                    ),
                    range(12),
                )
            )
        self.assertTrue(all(result == expected for result in results))


if __name__ == "__main__":
    unittest.main()
