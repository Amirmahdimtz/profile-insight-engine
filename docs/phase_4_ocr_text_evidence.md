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

Official references:

- Windows installation: https://tesseract-ocr.github.io/tessdoc/Installation.html#windows
- `tessdata_fast`: https://github.com/tesseract-ocr/tessdata_fast
- `tessdata_best`: https://github.com/tesseract-ocr/tessdata_best

## Windows local verification prerequisites

Tesseract installation and traineddata installation are separate prerequisites. Do not run the OCR benchmark until both are available.

PowerShell variables, from the repository root:

```powershell
$env:DATASET_ROOT = "E:\phase2-real-dataset"
$env:MODEL_ROOT = "E:\profile-insight-models"
$env:TESSDATA_FAST = Join-Path $env:MODEL_ROOT "tessdata_fast"
$env:TESSDATA_BEST = Join-Path $env:MODEL_ROOT "tessdata_best"
$env:PHASE4_REPORT = Join-Path (Get-Location) "phase4_ocr_benchmark.json"
$env:EXPECTED_MANIFEST_FINGERPRINT = "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc"
$env:EXPECTED_DATASET_CONTENT_FINGERPRINT = "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e"
```

Verify the Tesseract executable first:

```powershell
$TesseractCommand = Get-Command tesseract -ErrorAction SilentlyContinue
if (-not $TesseractCommand) {
    $DefaultTesseractExe = "C:\Program Files\Tesseract-OCR\tesseract.exe"
    if (Test-Path $DefaultTesseractExe) {
        $env:PATH = "$(Split-Path $DefaultTesseractExe);$env:PATH"
        $TesseractCommand = Get-Command tesseract -ErrorAction SilentlyContinue
    }
}
if (-not $TesseractCommand) {
    throw "Tesseract 5 is not installed or is not on PATH. Install the Windows build referenced by the official Tesseract documentation, then reopen PowerShell."
}
tesseract --version
```

Create the candidate model directories and download only the required official traineddata files:

```powershell
New-Item -ItemType Directory -Force -Path $env:TESSDATA_FAST,$env:TESSDATA_BEST | Out-Null
foreach ($Language in @("fas","eng","osd")) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$Language.traineddata" -OutFile (Join-Path $env:TESSDATA_FAST "$Language.traineddata")
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$Language.traineddata" -OutFile (Join-Path $env:TESSDATA_BEST "$Language.traineddata")
}
```

Fail fast if any required model is missing or empty:

```powershell
foreach ($Directory in @($env:TESSDATA_FAST,$env:TESSDATA_BEST)) {
    foreach ($Language in @("fas","eng","osd")) {
        $ModelPath = Join-Path $Directory "$Language.traineddata"
        if (-not (Test-Path $ModelPath -PathType Leaf)) { throw "Missing traineddata file: $ModelPath" }
        if ((Get-Item $ModelPath).Length -le 0) { throw "Empty traineddata file: $ModelPath" }
    }
}
```

Verify Tesseract can load each candidate:

```powershell
tesseract --tessdata-dir "$env:TESSDATA_FAST" --list-langs
tesseract --tessdata-dir "$env:TESSDATA_BEST" --list-langs
```

Both lists must contain `fas`, `eng`, and `osd`.

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

PowerShell benchmark command, after the prerequisite checks above:

```powershell
python -m evaluation.ocr_benchmark --manifest "$env:DATASET_ROOT\manifest.json" --dataset-root "$env:DATASET_ROOT" --candidate "tessdata_fast=$env:TESSDATA_FAST" --candidate "tessdata_best=$env:TESSDATA_BEST" --iterations 2 --expected-manifest-fingerprint "$env:EXPECTED_MANIFEST_FINGERPRINT" --expected-dataset-content-fingerprint "$env:EXPECTED_DATASET_CONTENT_FINGERPRINT" --output "$env:PHASE4_REPORT"
```

Inspect the report:

```powershell
$Report = Get-Content "$env:PHASE4_REPORT" -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,dataset_id,dataset_version,manifest_fingerprint,dataset_content_fingerprint,eval_sample_count,ocr_ground_truth_sample_count
$Report.ocr_ground_truth_coverage_by_slice | Format-List
$Report.candidates | Select-Object candidate,provider,provider_version,model_id,sample_count,raw_cer,raw_wer,normalized_cer,normalized_wer,p50_latency_ms,p95_latency_ms,process_peak_ram_mb,cpu_usage_percent,gpu_required,deterministic_rerun | Format-Table -AutoSize
```

## Failure and privacy behavior

Provider errors are deterministic sanitized exceptions. No raw image bytes, canonical image bytes, OCR text, TSV output, or sensitive OCR content is written to operational logs by Phase 4 production code. OCR only emits observable text evidence; it does not produce claims about religion, politics, ethnicity, mental health, sexual orientation, intelligence, honesty, family relationships, or other sensitive/internal traits.

## Verified local evidence

Target Windows verification completed with Tesseract `v5.4.0.20240606` and official `tessdata_fast` / `tessdata_best` candidates using `fas+eng+osd`.

Verified dataset identity:

- `dataset_id=profile_insight_real_eval`
- `dataset_version=1.0.0`
- manifest fingerprint: `06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc`
- dataset-content fingerprint: `5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e`
- 18 eval samples, 15 OCR-ground-truth samples
- OCR-ground-truth coverage: Persian-heavy 5, English-heavy 1, mixed Persian/English 1, low-quality 1, no-text 8, text-heavy 5, meme 5

Candidate evidence:

| Candidate | Raw CER | Raw WER | p50 ms | p95 ms | Peak Python RSS MB | Deterministic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tessdata_fast` | 0.3433123420 | 0.3680080483 | 1490.10 | 4927.42 | 36.02 | yes |
| `tessdata_best` | 0.3365972274 | 0.3594232059 | 1545.42 | 6774.15 | 36.02 | yes |

Relevant slice evidence:

- Persian-heavy: fast CER/WER `0.7271 / 0.7857`; best `0.7069 / 0.7571`.
- Mixed Persian/English: both CER `0.5143`; fast WER `0.5915`, best WER `0.6056`.
- English-heavy / low-quality: both candidates measured CER/WER `1.0 / 1.0` on the current single shared sample. This limitation is retained as benchmark evidence rather than hidden by normalization or threshold tuning.
- No-text: both candidates measured CER/WER `0.0 / 0.0` across 8 annotated samples.

The benchmark does not establish one tessdata candidate as universally dominant: `tessdata_best` has slightly better overall and Persian accuracy, while `tessdata_fast` has materially lower p95 latency and slightly better mixed WER on the current dataset. Phase 4 therefore verifies the Tesseract provider and keeps the traineddata choice non-final for the broader Phase 10 benchmark rather than prematurely freezing a production model.

Full local regression previously passed 145 tests, including Phase 4 architecture, provider failure, traineddata-only candidate, normalization, mixed-script, deterministic, and DI/discovery coverage. The dataset-only annotation additions did not modify repository runtime code.

## Status

`COMPLETE / READY_FOR_NEXT_PHASE`

Phase 4 is locally verified on the target Windows environment. Provider output normalization, deterministic failure behavior, privacy constraints, real Persian/English/mixed/no-text evaluation coverage, candidate comparison, and reproducibility fingerprints are all evidenced. Final production model/provider tuning remains a Phase 10 responsibility.
