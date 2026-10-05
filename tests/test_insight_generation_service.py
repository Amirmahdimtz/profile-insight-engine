import tempfile
import unittest
from pathlib import Path

from src.core.profile_analysis.aggregation_contracts import AggregatedTheme, AggregationKind
from src.core.profile_analysis.contracts import (
    Evidence,
    EvidenceType,
    ImageAnalysisResult,
    InsightType,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
)
from src.core.profile_analysis.insight_contracts import InsightGenerationError
from src.core.services.profile_analysis.insight_generation_service import InsightGenerationService
from src.infrastructure.utils.config_reader import ConfigReader


DEFAULT_CONFIG = """\
insights:
  policy_version: phase8-test-v1
  minimum_evidence_count: 2
  minimum_unique_image_count: 2
  minimum_image_coverage: 0.1
  minimum_source_diversity: 1
  minimum_cross_image_consistency: 0.0
  minimum_average_confidence: 0.0
  source_diversity_saturation: 2
"""


def make_service(config_text=DEFAULT_CONFIG):
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
    try:
        handle.write(config_text)
        handle.close()
        return InsightGenerationService(ConfigReader(handle.name))
    finally:
        Path(handle.name).unlink(missing_ok=True)


def make_evidence(evidence_id, image_id, *, kind=EvidenceType.TOPIC, label="football", confidence=0.8, source="vision"):
    return Evidence(
        id=evidence_id,
        image_id=image_id,
        type=kind,
        label=label,
        value=label,
        confidence=confidence,
        source=source,
    )


def make_aggregate(
    *,
    kind=AggregationKind.TOPIC,
    key="football",
    label="football",
    evidence_count=2,
    unique_image_count=2,
    image_universe_count=4,
    average_confidence=0.8,
    max_confidence=0.9,
    source_diversity=1,
    consistency=1.0,
    sources=None,
    image_ids=None,
    signal_ids=None,
):
    sources = sources or tuple(f"source-{i}" for i in range(source_diversity))
    image_ids = image_ids or tuple(f"img-{i + 1}" for i in range(unique_image_count))
    signal_ids = signal_ids or tuple(f"ev-{i + 1}" for i in range(evidence_count))
    return AggregatedTheme(
        kind=kind,
        key=key,
        label=label,
        evidence_count=evidence_count,
        unique_image_count=unique_image_count,
        image_universe_count=image_universe_count,
        image_coverage=unique_image_count / image_universe_count,
        average_confidence=average_confidence,
        max_confidence=max_confidence,
        source_diversity=source_diversity,
        cross_image_consistency=consistency,
        sources=sources,
        supporting_image_ids=image_ids,
        supporting_signal_ids=signal_ids,
    )


def evidence_for(item, *, sources=("vision",)):
    return tuple(
        make_evidence(
            signal_id,
            item.supporting_image_ids[index % len(item.supporting_image_ids)],
            kind=EvidenceType(item.kind.value),
            label=item.label,
            confidence=item.average_confidence if item.average_confidence is not None else 0.5,
            source=sources[index % len(sources)],
        )
        for index, signal_id in enumerate(item.supporting_signal_ids)
    )


class InsightGenerationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = make_service()

    def test_minimum_support_boundary(self):
        below = make_aggregate(evidence_count=1, signal_ids=("ev-1",))
        exact = make_aggregate()
        above = make_aggregate(evidence_count=3, signal_ids=("ev-1", "ev-2", "ev-3"))
        evidence = (
            make_evidence("ev-1", "img-1"),
            make_evidence("ev-2", "img-2"),
            make_evidence("ev-3", "img-1"),
        )
        self.assertEqual(self.service.generate((below,), evidence).insights, ())
        self.assertEqual(len(self.service.generate((exact,), evidence).insights), 1)
        self.assertEqual(len(self.service.generate((above,), evidence).insights), 1)

    def test_low_coverage_and_single_image_are_ineligible(self):
        coverage_service = make_service(DEFAULT_CONFIG.replace("minimum_image_coverage: 0.1", "minimum_image_coverage: 0.6"))
        low_coverage = make_aggregate(unique_image_count=2, image_universe_count=4)
        self.assertEqual(coverage_service.generate((low_coverage,), evidence_for(low_coverage)).insights, ())

        single = make_aggregate(
            evidence_count=3,
            unique_image_count=1,
            image_ids=("img-1",),
            signal_ids=("ev-1", "ev-2", "ev-3"),
        )
        self.assertEqual(self.service.generate((single,), evidence_for(single)).insights, ())

    def test_recurrence_is_traceable_and_score_is_explainable(self):
        item = make_aggregate(source_diversity=1, sources=("vision",))
        insight = self.service.generate((item,), evidence_for(item)).insights[0]
        self.assertEqual(insight.type, InsightType.VISIBLE_INTEREST)
        self.assertEqual(insight.evidence_count, 2)
        self.assertEqual(insight.image_coverage, 0.5)
        self.assertEqual(insight.supporting_evidence_ids, ("ev-1", "ev-2"))
        self.assertEqual(insight.supporting_image_ids, ("img-1", "img-2"))
        self.assertAlmostEqual(insight.confidence, (0.8 + 0.5 + 1.0 + 0.5) / 4.0)
        self.assertIn("source_diversity=1", insight.explanation)
        self.assertIn("cross_image_consistency=1.000000", insight.explanation)

        diverse = make_aggregate(source_diversity=2, sources=("ocr", "vision"))
        diverse_insight = self.service.generate((diverse,), evidence_for(diverse, sources=("ocr", "vision"))).insights[0]
        self.assertGreater(diverse_insight.confidence, insight.confidence)
        self.assertLessEqual(diverse_insight.confidence, 1.0)

    def test_configured_diversity_consistency_and_confidence_thresholds_are_enforced(self):
        strict = make_service(
            DEFAULT_CONFIG
            .replace("minimum_source_diversity: 1", "minimum_source_diversity: 2")
            .replace("minimum_cross_image_consistency: 0.0", "minimum_cross_image_consistency: 0.6")
            .replace("minimum_average_confidence: 0.0", "minimum_average_confidence: 0.7")
        )
        cases = (
            make_aggregate(source_diversity=1, sources=("vision",)),
            make_aggregate(source_diversity=2, sources=("ocr", "vision"), consistency=0.5),
            make_aggregate(source_diversity=2, sources=("ocr", "vision"), consistency=0.8, average_confidence=0.69, max_confidence=0.8),
        )
        for item in cases:
            with self.subTest(item=item):
                self.assertEqual(strict.generate((item,), evidence_for(item)).insights, ())
        eligible = make_aggregate(source_diversity=2, sources=("ocr", "vision"), consistency=0.6, average_confidence=0.7, max_confidence=0.8)
        self.assertEqual(len(strict.generate((eligible,), evidence_for(eligible)).insights), 1)

    def test_no_evidence_and_support_confidence_tradeoffs(self):
        empty = self.service.generate((), ())
        self.assertEqual(empty.insights, ())
        self.assertEqual(empty.summary, "No supported insights met the configured evidence policy.")

        high_conf_low_support = make_aggregate(evidence_count=1, unique_image_count=1, average_confidence=1.0, max_confidence=1.0, image_ids=("img-1",), signal_ids=("ev-1",))
        self.assertEqual(self.service.generate((high_conf_low_support,), evidence_for(high_conf_low_support)).insights, ())

        lower_conf_high_support = make_aggregate(evidence_count=4, unique_image_count=3, average_confidence=0.2, max_confidence=0.3, image_ids=("img-1", "img-2", "img-3"), signal_ids=("ev-1", "ev-2", "ev-3", "ev-4"))
        result = self.service.generate((lower_conf_high_support,), evidence_for(lower_conf_high_support))
        self.assertEqual(len(result.insights), 1)
        self.assertGreaterEqual(result.insights[0].confidence, 0.0)
        self.assertLessEqual(result.insights[0].confidence, 1.0)

    def test_semantic_only_similarity_never_becomes_probability_confidence(self):
        semantic = make_aggregate(
            kind=AggregationKind.SEMANTIC_THEME,
            key="travel",
            label="travel",
            average_confidence=None,
            max_confidence=None,
            signal_ids=("semantic-1", "semantic-2"),
        )
        self.assertEqual(self.service.generate((semantic,), ()).insights, ())

    def test_sensitive_inference_is_suppressed_but_observable_religious_content_is_allowed(self):
        sensitive = make_aggregate(
            kind=AggregationKind.OTHER_OBSERVABLE,
            key="person_religion",
            label="Person religion = example",
        )
        evidence = (
            make_evidence("ev-1", "img-1", kind=EvidenceType.OTHER_OBSERVABLE, label="observable content"),
            make_evidence("ev-2", "img-2", kind=EvidenceType.OTHER_OBSERVABLE, label="observable content"),
        )
        self.assertEqual(self.service.generate((sensitive,), evidence).insights, ())

        allowed = make_aggregate(
            kind=AggregationKind.RELIGIOUS_CONTENT,
            key="religious content",
            label="religious-themed content",
        )
        allowed_evidence = (
            make_evidence("ev-1", "img-1", kind=EvidenceType.RELIGIOUS_CONTENT, label=allowed.label),
            make_evidence("ev-2", "img-2", kind=EvidenceType.RELIGIOUS_CONTENT, label=allowed.label),
        )
        insight = self.service.generate((allowed,), allowed_evidence).insights[0]
        self.assertEqual(insight.type, InsightType.RELIGIOUS_CONTENT)
        self.assertNotIn("person religion", insight.label.lower())

    def test_summary_cannot_add_new_fact(self):
        topic = make_aggregate()
        brand = make_aggregate(kind=AggregationKind.BRAND, key="camera_brand", label="camera_brand", signal_ids=("brand-1", "brand-2"))
        evidence = evidence_for(topic) + (
            make_evidence("brand-1", "img-1", kind=EvidenceType.BRAND, label="camera_brand"),
            make_evidence("brand-2", "img-2", kind=EvidenceType.BRAND, label="camera_brand"),
        )
        result = self.service.generate((topic, brand), evidence)
        expected = "Supported insights: " + " | ".join(item.label for item in result.insights)
        self.assertEqual(result.summary, expected)

    def test_conflicts_and_invalid_support_references_are_rejected(self):
        item = make_aggregate()
        with self.assertRaises(InsightGenerationError):
            self.service.generate((item, make_aggregate(average_confidence=0.7)), evidence_for(item))
        with self.assertRaises(InsightGenerationError):
            self.service.generate((item,), (make_evidence("ev-1", "img-1"),))
        wrong_type = (
            make_evidence("ev-1", "img-1", kind=EvidenceType.BRAND, label="brand"),
            make_evidence("ev-2", "img-2", kind=EvidenceType.BRAND, label="brand"),
        )
        with self.assertRaises(InsightGenerationError):
            self.service.generate((item,), wrong_type)
        duplicate_id = (
            make_evidence("ev-1", "img-1", confidence=0.8),
            make_evidence("ev-1", "img-1", confidence=0.9),
            make_evidence("ev-2", "img-2"),
        )
        with self.assertRaises(InsightGenerationError):
            self.service.generate((item,), duplicate_id)

    def test_exact_duplicate_canonical_support_resolves_to_actual_evidence_image_ids(self):
        item = make_aggregate(
            unique_image_count=1,
            image_universe_count=2,
            image_ids=("canonical-a",),
            signal_ids=("ev-b", "ev-c"),
        )
        service = make_service(DEFAULT_CONFIG.replace("minimum_unique_image_count: 2", "minimum_unique_image_count: 1"))
        evidence = (make_evidence("ev-b", "duplicate-b"), make_evidence("ev-c", "duplicate-b"))
        insight = service.generate((item,), evidence).insights[0]
        self.assertEqual(insight.supporting_image_ids, ("duplicate-b",))
        graph = ProfileAnalysisResult(
            analysis_id="analysis",
            status=ProfileAnalysisStatus.COMPLETED,
            images=(
                ImageAnalysisResult("analysis", "canonical-a", ()),
                ImageAnalysisResult("analysis", "duplicate-b", evidence),
            ),
            insights=(insight,),
        )
        self.assertEqual(graph.insights[0].image_coverage, 0.5)

    def test_ranking_tie_break_and_repeated_reversed_runs_are_stable(self):
        first = make_aggregate(kind=AggregationKind.BRAND, key="brand_a", label="brand_a", signal_ids=("a-1", "a-2"))
        second = make_aggregate(kind=AggregationKind.BRAND, key="brand_b", label="brand_b", signal_ids=("b-1", "b-2"))
        evidence = (
            make_evidence("a-1", "img-1", kind=EvidenceType.BRAND, label="brand_a"),
            make_evidence("a-2", "img-2", kind=EvidenceType.BRAND, label="brand_a"),
            make_evidence("b-1", "img-1", kind=EvidenceType.BRAND, label="brand_b"),
            make_evidence("b-2", "img-2", kind=EvidenceType.BRAND, label="brand_b"),
        )
        baseline = self.service.generate((second, first), evidence)
        reversed_result = self.service.generate((first, second), tuple(reversed(evidence)))
        repeated = self.service.generate((second, first), evidence)
        self.assertEqual(baseline, reversed_result)
        self.assertEqual(baseline, repeated)
        self.assertEqual([x.label for x in baseline.insights], ["Recurring brand reference: brand_a", "Recurring brand reference: brand_b"])

    def test_explicit_category_mapping(self):
        cases = (
            (AggregationKind.OBJECT, EvidenceType.OBJECT, InsightType.CONTENT_PATTERN),
            (AggregationKind.SCENE, EvidenceType.SCENE, InsightType.ENVIRONMENT),
            (AggregationKind.OCR_TEXT, EvidenceType.OCR_TEXT, InsightType.TEXTUAL_REFERENCE),
            (AggregationKind.ACTIVITY, EvidenceType.ACTIVITY, InsightType.ACTIVITY),
            (AggregationKind.ENVIRONMENT, EvidenceType.ENVIRONMENT, InsightType.ENVIRONMENT),
            (AggregationKind.TOPIC, EvidenceType.TOPIC, InsightType.VISIBLE_INTEREST),
            (AggregationKind.BRAND, EvidenceType.BRAND, InsightType.BRAND_REFERENCE),
            (AggregationKind.TEAM, EvidenceType.TEAM, InsightType.TEAM_REFERENCE),
            (AggregationKind.RELIGIOUS_CONTENT, EvidenceType.RELIGIOUS_CONTENT, InsightType.RELIGIOUS_CONTENT),
            (AggregationKind.SOCIAL_CONTEXT, EvidenceType.SOCIAL_CONTEXT, InsightType.SOCIAL_CONTEXT),
            (AggregationKind.OTHER_OBSERVABLE, EvidenceType.OTHER_OBSERVABLE, InsightType.CONTENT_PATTERN),
        )
        for index, (kind, evidence_type, expected_type) in enumerate(cases):
            item = make_aggregate(kind=kind, key=f"concept_{index}", label=f"concept_{index}", signal_ids=(f"ev-{index}-1", f"ev-{index}-2"))
            evidence = (
                make_evidence(f"ev-{index}-1", "img-1", kind=evidence_type, label=item.label),
                make_evidence(f"ev-{index}-2", "img-2", kind=evidence_type, label=item.label),
            )
            with self.subTest(kind=kind):
                self.assertEqual(self.service.generate((item,), evidence).insights[0].type, expected_type)

    def test_config_validation_rejects_invalid_ranges_and_inconsistent_diversity(self):
        invalid_probability = DEFAULT_CONFIG.replace("minimum_image_coverage: 0.1", "minimum_image_coverage: 1.1")
        with self.assertRaises(InsightGenerationError):
            make_service(invalid_probability)
        invalid_diversity = DEFAULT_CONFIG.replace("source_diversity_saturation: 2", "source_diversity_saturation: 1").replace("minimum_source_diversity: 1", "minimum_source_diversity: 2")
        with self.assertRaises(InsightGenerationError):
            make_service(invalid_diversity)


if __name__ == "__main__":
    unittest.main()
