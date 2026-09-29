# Phase 4 — OCR & Text Evidence

## Scope

Phase 4 adds only `CanonicalImage → OCR Provider → OCR/Text Evidence`. It does not add Visual/VLM understanding, object or scene detection, embeddings, semantic themes, cross-image aggregation, Insight generation, HTTP API, persistence, SQLAlchemy, Alembic, workers, or any Phase 5+ component.

## Contract

`IOcrProvider` is a real replaceable provider boundary because OCR engines/models are benchmarkable external dependencies. The concrete production implementation is `TesseractOcrProvider` in Infrastructure. DI discovery now maps a discovered `@inject` implementation to an interface base whose name starts with `I`, matching the repository architecture convention; there is no feature-specific manual registration.

The provider returns only provider-neutral `OcrProviderResult` / `OcrProviderBlock` values. Tesseract TSV itself never crosses the Infrastructure boundary. `OcrEvidenceService` validates the provider result, performs deterministic normalization, attaches observable script/language hints, and emits standard Phase 1 `Evidence(type=ocr_text)` with deterministic IDs and direct `image_id` traceability.

Each OCR block preserves both `raw_text` and `normalized_text`, a bounded confidence, deterministic order, and an optional rectangular region when the provider supplies word geometry. The final Evidence value is the normalized text; the raw OCR text is retained in Evidence metadata so accuracy evaluation can use the unmodified provider text.

## Text normalization

Normalization is intentionally limited and deterministic:

- Unicode NFC normalization.
- CRLF/CR line endings become LF.
- NBSP/narrow-NBSP become ordinary spaces.
- Arabic Yeh/Alef-Maksura/Kaf forms `ي`, `ى`, `ك` normalize to Persian `ی`, `ی`, `ک`.
- Horizontal whitespace is collapsed within each line and surrounding whitespace is removed.
- ZWNJ is preserved.

There is no translation, spelling correction, semantic rewriting, LLM rewrite, guessed token insertion, or sensitive/person-level inference. Raw OCR text is always preserved separately. The benchmark reports both raw and normalized CER/WER so normalization cannot hide provider errors.

`script_hint` is a Unicode-script observation (`arabic`, `latin`, `mixed_arabic_latin`, `numeric`, `other`). `language_hint` is deliberately only a hint within the configured Persian/English scope: Arabic-script text maps to `fa`, Latin text to `en`, mixed text to `mixed`, and numeric/other text has no language hint.

## Tesseract provider

The production provider invokes the local Tesseract CLI without a Python wrapper dependency and requests UTF-8 TSV output. Word rows are grouped deterministically into line blocks; confidence is the arithmetic mean of valid word confidences and the region is the union of word boxes. Missing TSV columns, malformed numeric fields, invalid confidence, non-UTF-8 output, timeout, missing executable, or non-zero provider exit are converted to sanitized `OcrProviderError` messages. OCR text and canonical image bytes are not logged.

Default runtime configuration is read through `ConfigReader`:

- `ocr.provider`
- `ocr.executable`
- `ocr.languages`
- `ocr.page_segmentation_mode`
- `ocr.timeout_seconds`
- `ocr.model_id`
- `ocr.config_version`

The default candidate uses `fas+eng` and PSM 1 so orientation/script detection can participate when the local Tesseract installation includes `osd.traineddata`. These are Phase 4 candidate settings, not a frozen Phase 10 production choice.

## Provider/model decision

Tesseract is implemented as the first real local/offline provider because it supports Persian (`fas`) and English (`eng`) trained data and exposes TSV text, confidence and bounding geometry. Provider/model selection is **not frozen** before project-specific benchmark evidence.

The Phase 4 benchmark accepts multiple Tesseract tessdata candidates in one run. The intended first comparison is official `tessdata_fast` versus `tessdata_best` for `fas`, `eng`, and `osd`. A candidate name and tessdata directory are report metadata; no automatic winner is promoted into runtime config.

## Evaluation and benchmark

`evaluation.ocr_benchmark` reuses the verified Phase 2 manifest, reference validation/fingerprints, CER/WER calculators, Phase 3 canonicalization, and the production Phase 4 OCR service/provider path. It includes only eval samples whose `ground_truth.ocr_text` is not null and reports actual OCR-ground-truth coverage by Phase 2 slice instead of assuming annotation coverage.

For each candidate it records:

- raw CER/WER;
- normalized CER/WER;
- per-slice raw CER/WER for covered slices;
- p50/p95 OCR latency per image;
- Python process peak RSS;
- deterministic rerun status;
- provider version, model ID, config version and language config;
- GPU requirement (`false`, because this provider path is CPU-only).

On Windows, stdlib process accounting cannot reliably include the Tesseract child process CPU, so `cpu_usage_percent` is explicitly `null` rather than fabricated. The report also states that process RSS is the Python process metric and does not claim Tesseract child peak memory. A future resource measurement dependency is not added solely to manufacture Phase 4 numbers.

The command accepts expected manifest/content fingerprints and fails on mismatch. The authorized dataset itself is never modified by this benchmark.

## Failure and privacy behavior

Provider errors are deterministic sanitized exceptions. No raw image bytes, canonical image bytes, OCR text, TSV output, or sensitive OCR content is written to operational logs by Phase 4 production code. OCR only emits observable text evidence; it does not produce claims about religion, politics, ethnicity, mental health, sexual orientation, intelligence, honesty, family relationships, or other sensitive/internal traits.

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

Phase 4 must not be marked complete until the real authorized dataset fingerprints, OCR coverage, candidate comparison, full regression, architecture tests, and local Tesseract execution are verified on the target Windows environment.
