# Phase 8 — Insight Engine

## Scope

Phase 8 converts the deterministic Phase 7 `AggregatedTheme` output into explainable, evidence-backed `ProfileInsight` values and a deterministic evidence-backed summary:

`AggregatedTheme + Evidence registry → Eligibility policy → Support scoring → ProfileInsight → deterministic summary`

This phase remains Core/evaluation/config only. It does not add HTTP endpoints, Application DTOs, repositories, SQLAlchemy models, migrations, database workflows, job/status lifecycle, or any other Phase 9 component.

## Inputs and traceability

`InsightGenerationService.generate(...)` accepts:

- a sequence of real Phase 7 `AggregatedTheme` values;
- the normalized `Evidence` registry whose IDs are referenced by eligible non-semantic aggregates.

Eligibility and scoring are driven by aggregate metrics. The Evidence registry is not re-aggregated and is used only to validate support references and produce `ProfileInsight.supporting_evidence_ids` / `supporting_image_ids` that remain valid under the existing Phase 1 `ProfileAnalysisResult` support graph.

This matters for Phase 3 exact duplicates: Phase 7 exposes canonical supporting image IDs while retaining a representative real Evidence ID. Phase 8 resolves the selected Evidence IDs back to their original Evidence image IDs so a later `ProfileAnalysisResult` can validate the graph without changing Phase 7 canonical coverage semantics.

Unknown support IDs, support IDs with an incompatible Evidence type, conflicting duplicate Evidence IDs, or conflicting duplicate aggregate identities are rejected.

## Insight category mapping

The mapping is explicit and deterministic:

| AggregationKind | InsightType | Output wording |
| --- | --- | --- |
| `object` | `content_pattern` | `Recurring object: <label>` |
| `scene` | `environment` | `Recurring scene: <label>` |
| `ocr_text` | `textual_reference` | `Recurring textual reference: <label>` |
| `activity` | `activity` | `Recurring visible activity: <label>` |
| `environment` | `environment` | `Recurring environment: <label>` |
| `topic` | `visible_interest` | `Recurring topic-related content: <label>` |
| `brand` | `brand_reference` | `Recurring brand reference: <label>` |
| `team` | `team_reference` | `Recurring team reference: <label>` |
| `religious_content` | `religious_content` | `Recurring religious-themed content: <label>` |
| `social_context` | `social_context` | `Recurring visible social context: <label>` |
| `other_observable` | `content_pattern` | `Recurring observable content: <label>` |
| `semantic_theme` | no standalone ProfileInsight | skipped |

The `visible_interest` type is used only for recurrent topic-related content. The label remains content-oriented and does not state an internal personal trait.

## Semantic-only aggregates

Phase 7 intentionally keeps semantic similarity separate from probability confidence. A semantic-only `AggregatedTheme` therefore has `average_confidence = null` and `max_confidence = null`, and its supporting signal IDs are deterministic semantic support IDs rather than Phase 1 Evidence IDs.

Phase 8 does not convert semantic similarity into probability/confidence and does not invent Evidence objects. A semantic-only aggregate therefore cannot independently become a `ProfileInsight` under the current contract. It remains available for future calibrated/fused policies only after a contract and benchmark justify that behavior.

## Eligibility policy

The candidate policy is read through the existing `ConfigReader` pattern from the `insights` section in both appsettings files. No threshold is hard-coded inside the Service.

Configured fields:

- `policy_version`
- `minimum_evidence_count`
- `minimum_unique_image_count`
- `minimum_image_coverage`
- `minimum_source_diversity`
- `minimum_cross_image_consistency`
- `minimum_average_confidence`
- `source_diversity_saturation`

An aggregate is eligible only when:

1. its `AggregationKind` has an explicit Phase 8 mapping;
2. `evidence_count` meets `minimum_evidence_count`;
3. `unique_image_count` meets `minimum_unique_image_count`;
4. `image_coverage` meets `minimum_image_coverage`;
5. `source_diversity` meets `minimum_source_diversity`;
6. `cross_image_consistency` meets `minimum_cross_image_consistency`;
7. `average_confidence` exists and meets `minimum_average_confidence`;
8. the aggregate key/label passes the existing observable-claim policy;
9. every supporting signal is a real compatible Evidence reference.

The committed values are a **Phase 8 candidate policy**, not calibrated production thresholds. `minimum_evidence_count = 2` and `minimum_image_coverage = 0.10` follow the official architecture document's example starting values; the additional cross-image requirement and other permissive bounds are explicit candidates for evaluation rather than frozen product truth. Final thresholds require real insight-level annotations and calibration evidence.

## Sensitive inference boundary

The existing Phase 1 policy remains authoritative. Phase 8 re-applies it defensively to aggregate key/label text before generating an Insight.

Allowed output describes observable content, for example:

`Recurring religious-themed content: religious-themed content`

Disallowed output includes personal sensitive-trait claims such as a person's religion, political ideology, ethnicity, mental health, sexual orientation, intelligence, honesty, family relationship, or similar internal traits.

Phase 8 does not add sensitive-trait categories or a new policy vocabulary.

## Conflict handling

The current Evidence/AggregatedTheme contracts contain no semantic conflict ontology such as mutually exclusive topic pairs. Phase 8 therefore does not invent one.

Conflict handling is limited to conflicts the current contracts can represent reliably:

- duplicate Evidence ID with a different payload is rejected;
- duplicate `(AggregationKind, key)` aggregate identity with a different aggregate payload is rejected;
- invalid/missing support references are rejected.

Low recurrence, low coverage, low source diversity, low cross-image consistency, or low aggregate confidence are handled through the configured eligibility policy and support score. They are not re-labeled as semantic contradiction.

## Insight support score / confidence semantics

`ProfileInsight.confidence` is a deterministic composite **support score**, not a calibrated probability that a claim is true.

For an eligible aggregate:

- `average_confidence` is the Phase 7 arithmetic mean of available standard Evidence confidence values;
- `image_coverage` is Phase 7 unique canonical supporting images divided by total batch image IDs;
- `cross_image_consistency` keeps the Phase 7 min/max per-image support-count semantics;
- `source_diversity_score = min(1, source_diversity / source_diversity_saturation)`.

The Phase 8 support score is the equal-weight arithmetic mean of those four bounded components. Equal weighting avoids pretending that an uncalibrated product-optimal weighting has already been learned. The source-diversity saturation point remains configurable.

`max_confidence` is preserved in the explanation for auditability but is not used in scoring, so one high-confidence outlier cannot dominate the Insight score.

No semantic similarity value is used as confidence.

## Explanation contract

Explanation is deterministic and contains only aggregate/support facts already present in the support graph:

- evidence signal count;
- supporting image count / image universe count;
- coverage;
- source diversity;
- cross-image consistency;
- average Evidence confidence;
- max Evidence confidence;
- resolved supporting image IDs.

Explanation does not introduce a new topic, trait, relationship, intent, or interpretation.

## Ranking and tie-break

Eligible candidates are sorted deterministically by:

1. support score descending;
2. image coverage descending;
3. unique supporting image count descending;
4. evidence count descending;
5. source diversity descending;
6. cross-image consistency descending;
7. `InsightType.value` ascending;
8. aggregate key ascending;
9. final label ascending.

Insight keys are deterministic SHA-256-derived lower-snake-case identifiers namespaced by Insight type and aggregate `(kind, key)`. Input order does not participate in ranking.

Repeated runs and reversed aggregate/Evidence order must produce exactly equal `InsightGenerationResult` values and serialized JSON.

## Evidence-backed summary

The summary is intentionally not a free summarizer and no generative model/provider is introduced.

For non-empty results, summary is exactly:

`Supported insights: <accepted insight label> | <accepted insight label> | ...`

using the already-ranked accepted Insights. Therefore the summary cannot add a new profile fact, category, confidence value, or interpretation.

For an empty result it states only the engine state:

`No supported insights met the configured evidence policy.`

The explicit Phase 8 rule is: **summary cannot add a new profile fact**.

## Evaluation / benchmark

`evaluation.insight_benchmark` extends the repository's deterministic evaluation approach with Phase 8 contract scenarios. Its report schema is `phase8-insight-benchmark-v1` and its metric scope is explicitly:

`synthetic_contract_scenarios_not_product_quality`

It checks:

- deterministic candidate-policy execution;
- reversed-input invariance;
- deterministic result hash;
- synthetic scenario insight precision/recall;
- false unsupported insight rate for generated contract-safe Insights;
- structural summary factuality against the accepted Insight labels.

The repository does not contain authorized Phase 8 insight-level gold annotations. Therefore the benchmark must not be reported as real product insight precision. The report keeps these unavailable metrics explicit:

- confidence calibration: unavailable without labeled Insight correctness outcomes;
- human review agreement: unavailable without independent human annotations.

Real product insight precision, calibration, human review agreement, and human factuality review remain Local/Evaluation work once an authorized insight-level annotated dataset exists. No values are fabricated for those metrics.

## Local verification

Run from the repository root after synchronizing `main`:

```powershell
git switch main
git pull --ff-only origin main
python -m compileall -q src evaluation tests
python -m unittest tests.test_insight_contracts tests.test_insight_generation_service tests.test_insight_benchmark tests.test_phase8_architecture tests.test_phase8_documentation
python -m evaluation.insight_benchmark --output phase8_insight_benchmark.json
$Report = Get-Content .\phase8_insight_benchmark.json -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,metric_scope,policy_version,aggregate_count,expected_supported_count,generated_insight_count,insight_precision,insight_recall,false_unsupported_insight_rate,summary_factuality,confidence_calibration,human_review_agreement,deterministic_rerun,reversed_input_deterministic,result_hash
$Report.limitations
if (-not $Report.deterministic_rerun -or -not $Report.reversed_input_deterministic) { throw "Phase 8 reproducibility check failed" }
if ($Report.summary_factuality -ne 1.0) { throw "Phase 8 summary factuality contract check failed" }
python -m unittest discover -s tests -p "test_*.py"
git status --short
git rev-parse HEAD
```

Expected report schema is `phase8-insight-benchmark-v1`. `metric_scope` must be `synthetic_contract_scenarios_not_product_quality`. Determinism fields and structural summary factuality must pass. `confidence_calibration` and `human_review_agreement` remain null until real annotation evidence exists.

The generated `phase8_insight_benchmark.json` may appear as an untracked local artifact; tracked production/test/documentation files must remain clean after verification.

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

Phase 8 is implemented only after the atomic implementation commit reaches `main`. It remains open until the user runs the target-machine verification commands and returns the real outputs. Phase 9 must not begin before explicit user instruction after Phase 8 closure.
