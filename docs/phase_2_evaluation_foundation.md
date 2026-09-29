# Phase 2 — Evaluation Foundation

## Scope

Phase 2 adds deterministic evaluation tooling before any OCR, Vision/VLM, Embedding, image-ingestion, API, persistence, or model runtime is introduced. The tooling lives under root-level `evaluation/` and is intentionally outside the business runtime.

The implementation uses only the Python standard library and the verified Phase 1 policy contract. It does not add a dependency manager, framework, provider abstraction, model download, GPU runtime, SQLAlchemy, FastAPI, or future-phase service.

## Dataset contract

The manifest contract is `EvaluationDatasetManifest` in `evaluation/contracts.py`.

- `schema_version`: fixed Phase 2 dataset schema version (`1.0.0`).
- `dataset_id`: stable dataset identity.
- `dataset_version`: required numeric `MAJOR.MINOR.PATCH` version.
- `label_schema_version`: fixed Phase 2 label schema version (`1.1.0`).
- `data_policy`: curator declaration `consented_or_authorized`. This is an auditable dataset assertion; software cannot independently prove consent.
- `samples`: deterministic collection of `EvaluationSample` values.

Each sample has:

- stable `sample_id`;
- primary split: `train` or `eval`;
- one or more dataset slices;
- stable `image_id` plus portable relative image reference;
- structured ground truth.

Supported dataset slices are:

`persian_heavy`, `english_heavy`, `mixed_persian_english`, `text_heavy`, `low_quality`, `screenshot`, `meme`, `group`, `no_text`, `indoor`, `outdoor`.

An empty train split is valid. An empty eval split is invalid. The manifest itself may not be empty.

### Integrity rules

Validation enforces:

- unique sample IDs;
- unique image IDs;
- unique image references;
- train/eval leakage detection at image-ID and image-reference level;
- safe relative POSIX references (no absolute paths, `..`, or backslashes);
- optional filesystem existence validation without decoding images;
- SHA-256 byte hashing when `--dataset-root` is supplied, including duplicate-content and cross-split content-leakage detection;
- strict unknown-field rejection;
- stable ordering and serialization;
- finite JSON numeric values;
- dataset and label schema version compatibility.

Phase 2 computes SHA-256 over referenced file bytes only when a dataset root is explicitly supplied for evaluation validation. This is an evaluation-integrity hash: it detects duplicate bytes, train/eval content leakage, and dataset mutation without decoding images. It is **not** the Phase 3 `CanonicalImage` content hash and is not used by the production ingestion pipeline.

The manifest fingerprint is SHA-256 over the deterministic UTF-8 JSON representation of the complete manifest. Reports may additionally carry `dataset_content_fingerprint`, derived from the referenced file bytes, so annotations and raw evaluation inputs can be audited independently.

## Label schema and policy boundary

`EvaluationLabelType` supports observable evaluation targets:

- object;
- scene;
- OCR/text;
- activity;
- environment;
- topic;
- non-sensitive visible interest;
- brand;
- team;
- religious-themed content;
- recurring observable content;
- visible social/group context.

Free-form ground-truth labels are validated through the same Phase 1 observable-claim policy. The policy implementation is not copied into the evaluation layer.

Allowed example:

`religious-themed content present`

Rejected example:

`Person religion = example`

`UnsupportedClaimAnnotation` is negative policy metadata. It can identify a category that a system must not infer, such as `person_religion` or `political_orientation`; it never stores a person's value for that trait. This distinction prevents the evaluation dataset from turning a prohibited personal inference into ground truth.

## Prediction contract

`PredictionSet` contains predictions for the `eval` split only and must exactly cover that split. It records:

- observable structured label predictions with `type`, `label`, structured `value`, and confidence in `[0, 1]`;
- optional OCR text;
- generic predicted claims used to measure unsupported-claim rate.

Observable label predictions themselves remain policy-safe. Generic claims are allowed to contain unsupported output specifically so the evaluator can measure the unsupported-claim rate using the Phase 1 policy.

Duplicate structured label predictions are rejected to avoid ambiguous metric counting. Classification identity is `(type, label, canonical value)`, so a correct label with the wrong structured value is counted as incorrect rather than silently accepted.

## Metrics

The generic calculators in `evaluation/metrics.py` are runtime- and provider-independent.

### OCR

- CER uses Unicode code points. There is no lowercasing, Unicode normalization, Persian/Arabic character normalization, or punctuation normalization.
- WER uses whitespace tokenization only. It does not lowercase or Unicode-normalize tokens.
- Empty reference and empty prediction => `0.0`.
- Empty reference and non-empty prediction => `1.0`.
- CER/WER are non-negative and may exceed `1.0` for non-empty references when insertions exceed reference length.

### Classification

`precision_recall_f1` uses set semantics, so duplicate generic inputs do not multiply counts.

- both expected and predicted empty => precision/recall/F1 `1.0`;
- expected non-empty, predicted empty => `0.0` F1;
- expected empty, predicted non-empty => `0.0` F1.

`macro_f1` supports an explicit class universe. Classes absent from both ground truth and predictions are excluded; predicted-only classes are included and therefore penalize false positives. If no class is present in either ground truth or predictions, the result is `1.0`.

### Retrieval

- `recall_at_k` requires `k > 0`;
- ranked results must not contain duplicates;
- no relevant items => Recall@K `0.0`;
- Average Precision with no relevant items => `0.0`;
- mAP requires identical query-ID graphs for relevant and ranked mappings.

These calculators are implemented now, but the benchmark runner does not wire retrieval metrics to dataset fields because the Phase 6 retrieval/evaluation-pair runtime contract does not exist yet.

### Confidence calibration

Expected Calibration Error uses a caller-selected positive bin count. Confidence `1.0` is placed in the final bin. Invalid probabilities, NaN, and Infinity are rejected. Empty observations => `0.0`.

### Unsupported-claim rate

Unsupported-claim rate is `unsupported claims / total claims`, where support is decided by the reused Phase 1 policy validator. Zero claims => `0.0`.

### System metrics

`SystemMetrics` represents p50 latency, p95 latency, RAM, and optional VRAM with finite non-negative validation and `p95 >= p50`. The benchmark CLI can accept measured values through `--system-metrics <json>`.

Phase 2 does not invent runtime measurements. System metrics are included only when an external benchmark step supplies measured values.

## Benchmark runner

Stable command:

```text
python -m evaluation.benchmark --manifest <manifest.json> --predictions <predictions.json> --output <report.json>
```

Optional flags:

- `--dataset-root <path>` validates references, hashes raw file bytes without image decoding, detects byte-identical duplicates/leakage, and records `dataset_content_fingerprint`;
- `--calibration-bins <positive integer>` controls ECE binning and is recorded in reproducibility metadata;
- `--report-kind benchmark|infrastructure_sanity_baseline` explicitly distinguishes ordinary reports from the test-only infrastructure baseline;
- `--system-metrics <json>` attaches externally measured p50/p95 latency and RAM/optional VRAM values without fabricating measurements.

The runner currently wires metrics that can be derived correctly from the Phase 2 sample/prediction contract:

- mean per-labeled-sample OCR CER;
- mean per-labeled-sample OCR WER;
- visual label micro precision/recall/F1 across `(sample, type, label, canonical value)` identities;
- confidence ECE over predicted observable labels;
- unsupported-claim rate.

Generic Macro F1, Recall@K, and mAP calculators are present and unit-tested, but are intentionally not connected to fabricated theme/retrieval runtime structures before their later phases define real inputs.

## Reproducibility

Reports contain:

- dataset ID and dataset version;
- deterministic manifest SHA-256 fingerprint;
- optional dataset-content SHA-256 fingerprint when referenced files are available;
- evaluator version;
- explicit metric configuration;
- stable sample, label, prediction, and metric ordering;
- deterministic JSON serialization.

No timestamp is embedded in the deterministic report payload. An external audit process may record wall-clock execution time without changing the evaluation result identity.

No random behavior is used, so no random seed is required in Phase 2.

## Infrastructure sanity baseline

`evaluation/baselines/infrastructure_sanity_baseline.json` is generated from the synthetic files under `evaluation/fixtures/`.

It exists only to prove that manifest parsing, prediction parsing, policy checks, metrics, fingerprinting, report generation, and deterministic reruns work together. It is **not** a product accuracy baseline, gold dataset result, candidate-model benchmark, or production model-selection artifact.

Recreate it with:

```text
python -m evaluation.benchmark --manifest evaluation/fixtures/sanity_manifest.json --predictions evaluation/fixtures/sanity_predictions.json --report-kind infrastructure_sanity_baseline --output evaluation/baselines/infrastructure_sanity_baseline.json
```

## Real dataset workflow

A real project dataset is not committed in Phase 2. Keep authorized data outside the repository if it is private. A compatible local layout is:

```text
<dataset-root>/
  manifest.json
  images/
    ...
```

Each manifest image `relative_path` is interpreted relative to `<dataset-root>`.

Validate structure, references, duplicate bytes, cross-split byte leakage, and compute the dataset-content fingerprint:

```text
python -m evaluation.validate_dataset --manifest <dataset-root>/manifest.json --dataset-root <dataset-root>
```

Generate candidate predictions using the provider/runtime from the appropriate later phase, then run:

```text
python -m evaluation.benchmark --manifest <dataset-root>/manifest.json --predictions <predictions.json> --dataset-root <dataset-root> --output <report.json>
```

A real OCR/Vision/Embedding baseline cannot exist until authorized project data and an actual candidate runtime are available. No model performance is synthesized in this phase.

## Phase boundary

Phase 2 adds no image decoding, MIME validation, resize/orientation/EXIF work, `CanonicalImage`, production ingestion hashing/deduplication, OCR/VLM/Embedding provider, model loading, CUDA/GPU runtime, aggregation runtime, insight generation, HTTP API, persistence, SQLAlchemy, Alembic, queue/worker, deployment, production model selection, or production threshold tuning. Evaluation-only byte hashing is limited to dataset integrity checks.
