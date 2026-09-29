# Phase 3 — Image Ingestion & Preprocessing

## Scope

Phase 3 adds only safe, bounded, deterministic image ingestion and canonical preprocessing. It does not add OCR, Vision/VLM, Embedding, cross-image aggregation, Insight generation, HTTP API, persistence, database models, migrations, workers, or model selection.

## Input and output

`ImageProcessingService.process_async()` accepts the verified Phase 1 `ProfileAnalysisRequest` plus one `RawImageInput` per request image ID. Image IDs must match exactly. Raw bytes are held only for processing and are never written to operational logs or persisted by Phase 3.

The service returns a `ProcessedImageBatch` containing one `CanonicalImage` per input, deterministic `DuplicateImageMapping` values, and an opaque temporary storage-scope reference. Each canonical image records validated dimensions, detected source MIME type, canonical dimensions/size, SHA-256 canonical content identity, resize/orientation flags, and the reference to a canonical PNG.

## Validation and canonicalization

Configured limits are read through `ConfigReader` from `src/host/res/appsettings*.yaml`:

- `profile_analysis.max_images`
- `profile_analysis.max_image_size_mb`
- `profile_analysis.max_image_pixels`
- `image.max_dimension`
- `image.allowed_mime_types`

The configured numeric values are operational Phase 3 limits, not model-selection thresholds or final Phase 10 tuning.

Processing is fail-fast and validates file size before decode, declared MIME, decoded format, optional filename extension, pixel count, single-frame input, full decoder verification, and full decode. JPEG, PNG, and WebP are the configured raster allowlist. A missing filename extension is accepted because trust is based on decoded content; a present extension must agree with the decoded format.

EXIF orientation is applied with `ImageOps.exif_transpose`. Other EXIF metadata is not retained. Alpha-bearing inputs are composited onto an opaque white background, all canonical pixels become RGB, and oversize images are resized with deterministic dimensions and Lanczos resampling. Canonical output is encoded as PNG without copying source EXIF metadata.

The canonical content hash is SHA-256 over a versioned preimage containing canonicalization version, RGB mode, canonical dimensions, and canonical pixel bytes. The hash is deliberately independent of upload filename, raw container bytes, temporary storage path, and PNG encoder metadata.

## Deduplication

Deduplication uses the canonical content hash after orientation and resize. Request order is authoritative: the first image with a hash owns the canonical temporary file, and each later image with the same hash maps deterministically to that first image ID. Duplicate inputs reuse the same canonical storage reference.

## Temporary storage lifecycle

`LocalImageStorage` is a concrete implementation because Phase 3 has no second storage backend. It creates an isolated per-processing temporary scope under the operating system temporary directory and writes only canonical PNG files with restrictive permissions. Raw uploads are never written to that scope.

On any processing failure, the service deletes the scope before propagating the sanitized error. On success, the caller must invoke `ImageProcessingService.release_async(batch)` after downstream consumers no longer need the canonical files. Phase 9 will own the broader analysis lifecycle; no persistence contract is introduced here.

## DI and configuration

The repository had no runtime DI/config implementation before Phase 3. The minimal convention-based infrastructure required by this phase is now present: `@inject` constructor injection, package discovery via `bootstrap_di()`, `ConfigReader`, and `LocalImageStorage`. Providers are discovered from decorated classes; no feature-specific manual provider registration is used.

## Benchmark evidence

Use the real authorized Phase 2 dataset.

PowerShell:

```powershell
$env:DATASET_ROOT = "C:\\absolute\\path\\to\\authorized\\evaluation-dataset"
$env:MANIFEST_PATH = Join-Path $env:DATASET_ROOT "manifest.json"

python -m evaluation.validate_dataset --manifest "$env:MANIFEST_PATH" --dataset-root "$env:DATASET_ROOT"

$env:PHASE3_REPORT = Join-Path $env:TEMP "phase3_image_preprocessing_benchmark.json"

python -m evaluation.image_preprocessing_benchmark --manifest "$env:MANIFEST_PATH" --dataset-root "$env:DATASET_ROOT" --batch-sizes 1,5,20,50 --iterations 3 --output "$env:PHASE3_REPORT"
```

POSIX shell:

```sh
export DATASET_ROOT="/absolute/path/to/authorized/evaluation-dataset"
export MANIFEST_PATH="$DATASET_ROOT/manifest.json"

python -m evaluation.image_preprocessing_benchmark \
  --manifest "$MANIFEST_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --batch-sizes 1,5,20,50 \
  --iterations 3 \
  --output /tmp/phase3_image_preprocessing_benchmark.json
```

The benchmark revalidates the Phase 2 dataset-content fingerprint, runs full Phase 3 processing for each requested batch size, verifies canonical hash stability across reruns, records p50/p95 batch latency and process peak RSS, measures decode throughput, and records an isolated resize-cost measurement using the same resize geometry/resampling settings. It fails instead of silently skipping a requested batch size when the dataset is too small.

Performance numbers are evidence, not hard-coded acceptance thresholds in Phase 3. Final operational tuning remains a Phase 10 responsibility.
