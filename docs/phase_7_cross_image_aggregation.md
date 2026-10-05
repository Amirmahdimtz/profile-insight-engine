# Phase 7 — Cross-Image Aggregation

## Scope

Phase 7 converts already-normalized per-image support into deterministic cross-image aggregates:

`Normalized OCR / visual Evidence + semantic theme matches + ProcessedImageBatch → AggregatedTheme`

This phase stops at aggregation. It does not generate profile Insights, explanations, summaries, HTTP endpoints, persistence, SQLAlchemy models, migrations, lifecycle state, or Phase 8 minimum-support policy.

## Inputs

`EvidenceAggregationService.aggregate(...)` accepts:

- `ProcessedImageBatch` from Phase 3 as the authoritative image universe and exact-duplicate mapping.
- normalized `Evidence` values produced by the existing OCR/visual contracts.
- optional `SemanticThemeResult` from Phase 6.

Raw provider payloads are not accepted. The service consumes only provider-neutral contracts already produced by prior phases.

## Evidence identity and duplicate evidence

`Evidence.id` is the primary identity for normalized Evidence. Repeating the same ID with the same image and payload contributes once. Reusing one ID with a different image or payload is rejected as conflicting input.

The existing Phase 4 and Phase 5 services already derive deterministic Evidence IDs. Phase 7 does not create a second Evidence identity scheme.

## Image identity and exact duplicate semantics

Phase 3 keeps one `CanonicalImage` for every requested image ID while recording exact canonical-content duplicates in `DuplicateImageMapping`. Phase 7 preserves that contract:

- the image universe is every image ID in `ProcessedImageBatch.images`;
- exact duplicates use `DuplicateImageMapping.canonical_image_id` for recurrence/support identity;
- an exact duplicate upload cannot increase the unique supporting-image numerator;
- for one canonical content identity plus aggregation kind/key/source, support comes from one deterministic representative original image. The Phase 3 canonical image is preferred; otherwise the lexically smallest duplicate image ID is used;
- repeated evidence occurrences inside that representative image remain distinct;
- Phase 6 semantic near-duplicate groups do **not** collapse image identity. Near-duplicate content remains distinct unless Phase 3 identified it as an exact canonical duplicate.

This separates exact content deduplication from semantic similarity and avoids inventing new duplicate-image semantics.

## Aggregation identity

Aggregation never merges values using fuzzy string similarity, stemming, embedding similarity, or provider-specific payloads.

The identity namespace is `(AggregationKind, normalized text)`:

- `AggregationKind` mirrors the existing `EvidenceType` values and adds only `semantic_theme` for Phase 6 matched theme support.
- normalization is limited to Unicode `NFC`, whitespace collapse/trim, then `casefold` for the key.
- OCR Evidence uses its normalized `Evidence.value` because the existing OCR label is the generic `ocr_text`.
- factual-caption Evidence uses its text `Evidence.value` because the existing label is the generic `factual_caption`.
- other normalized Evidence uses `Evidence.label`.
- semantic theme support uses `ThemeSimilarityEvidence.theme_label` and only contributes when `matched` is true.

Namespacing means equal strings from different evidence kinds do not merge. For example, object `apple` and brand `apple` remain separate aggregates.

## Metrics

For each `AggregatedTheme`:

- `evidence_count`: count of retained, deduplicated normalized support signals, including matched semantic-theme signals.
- `unique_image_count`: number of unique canonical supporting image IDs.
- `image_universe_count`: total number of image IDs in `ProcessedImageBatch.images`; exposed so the denominator is auditable.
- `image_coverage`: **unique canonical supporting image count / total batch image-id count**.
- `average_confidence`: arithmetic mean of available standard `Evidence.confidence` values only.
- `max_confidence`: maximum available standard `Evidence.confidence` value only.
- semantic similarity is not a probability/confidence and is never converted to one. A semantic-only aggregate therefore has `average_confidence = null` and `max_confidence = null`.
- `source_diversity`: number of unique standardized sources from `Evidence.source` and/or `SemanticThemeResult.provider`; provider metadata nested inside Evidence is not used as an extra source.
- `cross_image_consistency`: **min(per-image support count) / max(per-image support count)** across canonical supporting images. It is `1.0` for a single supporting image. This metric measures balance among supporting images; `image_coverage` separately measures how much of the image universe supports the aggregate.

No confidence threshold, minimum-support threshold, insight policy, or generated claim is introduced in Phase 7.

## Ordering and determinism

Output order is stable and independent of input iteration order. Aggregates sort by `(kind.value, normalized key)`. Sources, supporting canonical image IDs, and supporting signal IDs are sorted deterministically. Display labels use the lexically stable minimum among normalized spelling/case variants.

`math.fsum` is used for confidence summation after deterministic sorting. The service has no mutable analysis state and is safe for repeated/concurrent calls with identical immutable inputs.

## Architecture

The production additions are Core-only:

- `src/core/profile_analysis/aggregation_contracts.py`
- `src/core/services/profile_analysis/evidence_aggregation_service.py`

`EvidenceAggregationService` is a concrete `@inject` service and has no dependency requiring a new abstraction. There is no repository, persistence model, API DTO/controller, new provider, config key, or dependency in this phase.

## Performance evidence

`evaluation.aggregation_benchmark` is a deterministic contract-level benchmark. It generates only synthetic normalized contract objects; it does not fabricate model-quality ground truth and does not select or tune any ML model.

For configurable image counts and evidence density it reports:

- standard Evidence count, semantic signal count, and total input signal count;
- p50/p95 CPU aggregation latency;
- Python traced peak memory;
- process peak RAM when supported by the operating system;
- stable SHA-256 result hash;
- repeated-run determinism;
- reversed-input determinism;
- the implementation complexity profile: `O(S log S + I)` time and `O(S + I)` auxiliary space, where `S` is normalized input support signal count and `I` is image count.

The default workload covers 1/5/20/50 images with 20 standard Evidence values per image plus two semantic theme signals per image.

## Local verification

Run from the repository root on the target environment after synchronizing `main`:

```powershell
git switch main
git pull --ff-only origin main
python -m compileall -q src evaluation tests
python -m unittest tests.test_aggregation_contracts tests.test_evidence_aggregation_service tests.test_aggregation_benchmark tests.test_phase7_architecture tests.test_phase7_documentation
python -m evaluation.aggregation_benchmark --image-counts 1,5,20,50 --evidence-per-image 20 --iterations 5 --output phase7_aggregation_benchmark.json
$Report = Get-Content .\phase7_aggregation_benchmark.json -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,evidence_per_image,complexity_time,complexity_auxiliary_space
$Report.results | Select-Object image_count,standard_evidence_count,semantic_signal_count,total_input_signal_count,aggregate_count,p50_latency_ms,p95_latency_ms,traced_peak_memory_mb,process_peak_ram_mb,result_hash,deterministic_rerun,reversed_input_deterministic
if (($Report.results | Where-Object { -not $_.deterministic_rerun -or -not $_.reversed_input_deterministic }).Count -ne 0) { throw "Phase 7 reproducibility check failed" }
python -m unittest discover -s tests -p "test_*.py"
git status --short
```

Expected benchmark schema: `phase7-aggregation-benchmark-v1`. Every result row must report `deterministic_rerun = true` and `reversed_input_deterministic = true`. Unit/architecture/documentation tests and the full repository regression must pass. `git status --short` may show the generated untracked benchmark JSON; it must not show unexpected tracked changes.

## Verified local evidence

Target-Windows verification was completed against the Phase 7 implementation commit `40d580b52b9ec98e51423dcf0226353f12e23537`.

- verified source commit before closure: `40d580b52b9ec98e51423dcf0226353f12e23537`
- compile/import check: passed
- Phase 7 focused unit/architecture/documentation suite: 29 tests passed in 0.689 seconds
- full repository regression: 246 tests passed with no failures or errors in 6.415 seconds
- benchmark schema: `phase7-aggregation-benchmark-v1`
- benchmark image counts: `1, 5, 20, 50`
- standard evidence per image: `20`
- reported complexity: `O(S log S + I)` time and `O(S + I)` auxiliary space
- 1 image / 22 input signals: p50 `12.878600000476581 ms`, p95 `16.89726000022347 ms`, traced peak `0.03421211242675781 MB`, process peak RAM `20.71484375 MB`, result hash `fd7923208fc1a6786a269e693c326b1fc75e8c6f4918cdf5b9f989f2df104a46`
- 5 images / 110 input signals: p50 `34.219799999846146 ms`, p95 `40.48322000016924 ms`, traced peak `0.0752725601196289 MB`, process peak RAM `21.1953125 MB`, result hash `0e184ab162e5a5d150bea8af0f262308622ae74ad0afad3ee7d9e891cd896b7c`
- 20 images / 440 input signals: p50 `139.3674999999348 ms`, p95 `172.83910000005562 ms`, traced peak `0.3124961853027344 MB`, process peak RAM `22.1953125 MB`, result hash `1945ccca52525f1f06042e7bbab902f3c1ab2844217caa0d4efa8907617748ea`
- 50 images / 1100 input signals: p50 `312.9602999997587 ms`, p95 `383.0551600000035 ms`, traced peak `0.7606830596923828 MB`, process peak RAM `24.05859375 MB`, result hash `2e08a7f7ba6f90a1ef95ff86bd489c76fbb45ac2f8d1949235cc8255d0af73b6`
- `deterministic_rerun = true` for every benchmark scale
- `reversed_input_deterministic = true` for every benchmark scale
- PowerShell reproducibility guard completed without throwing
- working tree contained only untracked benchmark/evidence JSON artifacts; no tracked production, test, or documentation changes were pending

These results satisfy the Phase 7 determinism, traceability, denominator, duplicate-handling, CPU latency, memory, and reproducibility acceptance requirements. No Phase 8 insight policy or higher-phase behavior is introduced by this closure.

## Status

`COMPLETE / READY_FOR_NEXT_PHASE`

Phase 7 is locally verified on the target Windows environment. Its cross-image aggregation contracts, exact-duplicate semantics, deterministic ordering, coverage/confidence/source-diversity/consistency calculations, concurrency behavior, and scale benchmark evidence are complete for this phase. Phase 8 must not begin until the user explicitly requests the next phase.
