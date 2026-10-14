# Profile Insight Engine

Evidence-first profile image analysis service built with FastAPI, PostgreSQL, deterministic image preprocessing, Tesseract OCR, cross-image aggregation, and explainable evidence-backed insights.

> **Release status:** COMPLETE / PROJECT ACCEPTED / READY FOR BACKEND-DEVOPS HANDOFF  
> **Accepted deployment profile:** `degraded_ocr_only`  
> **Production Vision:** disabled by benchmark evidence  
> **Production OCR:** Tesseract + `tessdata_best` for Persian/English  
> **Validated target:** Windows 11, CPU-only, single Uvicorn worker

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Current production scope](#current-production-scope)
3. [Core product principles](#core-product-principles)
4. [Architecture](#architecture)
5. [Repository structure](#repository-structure)
6. [Requirements](#requirements)
7. [Quick start](#quick-start)
8. [Configuration](#configuration)
9. [Running the API](#running-the-api)
10. [API reference](#api-reference)
11. [Response and evidence model](#response-and-evidence-model)
12. [Image validation and preprocessing](#image-validation-and-preprocessing)
13. [Privacy, persistence, and safety boundaries](#privacy-persistence-and-safety-boundaries)
14. [Lifecycle and recovery](#lifecycle-and-recovery)
15. [Testing and verification](#testing-and-verification)
16. [Benchmarks and evaluation](#benchmarks-and-evaluation)
17. [Production deployment checklist](#production-deployment-checklist)
18. [Troubleshooting](#troubleshooting)
19. [Known limitations and deferred capabilities](#known-limitations-and-deferred-capabilities)
20. [Project documentation](#project-documentation)
21. [License](#license)

---

## What this project does

Profile Insight Engine receives a set of profile images and turns observable image evidence into traceable profile insights.

The intended evidence-first flow is:

```text
Images
  -> Per-image evidence
  -> Cross-image aggregation
  -> Supported insights
  -> Evidence + confidence + explanation
```

The service is designed so that an insight is only produced when enough observable evidence exists. Final insights carry support metadata such as:

- evidence count;
- image coverage;
- confidence;
- supporting evidence IDs;
- supporting image IDs;
- a deterministic explanation of the support.

The service also enforces an explicit product boundary: observable content may be analyzed, but sensitive or internal personal traits must not be inferred from appearance.

---

## Current production scope

The accepted release is intentionally conservative.

### Enabled in the accepted production profile

- FastAPI HTTP API;
- multipart image upload;
- deterministic image validation and preprocessing;
- JPEG, PNG, and WebP support;
- duplicate-image handling;
- Tesseract OCR using Persian and English;
- OCR normalization;
- cross-image aggregation;
- evidence-backed insight generation;
- PostgreSQL persistence;
- lifecycle/status/result endpoints;
- startup recovery;
- Alembic migrations;
- privacy sanitization before persistence;
- reproducible verification and benchmark tooling.

### Intentionally disabled or inactive

#### Vision

Production Vision is disabled:

```yaml
vision:
  enabled: false
```

The available local Vision candidates were benchmarked on the target CPU-only machine and rejected by evidence. The final holdout-safe SigLIP2 candidate completed reliably but produced zero holdout F1 because the TRAIN-derived taxonomy had zero coverage of the EVAL vocabulary.

The accepted release therefore does **not** claim scene, object, activity, or topic Vision capability.

Do not enable Vision in production without a new benchmark and evidence-review cycle.

#### Embedding / Semantic Themes

The embedding implementation and benchmark tooling remain in the repository, but the current HTTP workflow does not inject `SemanticThemeService`. No embedding runtime is required for the accepted deployment.

### What this means operationally

The accepted production path is:

```text
Client
  -> FastAPI
  -> ImageProcessingService
  -> Tesseract OCR
  -> EvidenceAggregationService
  -> InsightGenerationService
  -> privacy sanitization
  -> PostgreSQL
```

No llama.cpp server, GPU runtime, or embedding service is required for this release.

---

## Core product principles

### Evidence first

The service is built around the rule:

```text
No sufficient evidence -> no insight
```

Insights are derived from structured evidence rather than free-form model summaries.

### Explainability

Each supported insight is linked back to evidence and images and contains a support explanation.

### Determinism

Where a stage is deterministic, the same input/configuration should produce the same output.

### Privacy by design

Raw image content and raw OCR text are not treated as normal persistent application data.

### Sensitive-inference boundary

The service must not infer personal sensitive/internal traits such as:

- a person's religion;
- political orientation or ideology;
- ethnicity;
- mental-health state;
- sexual orientation;
- intelligence;
- honesty;
- family relationships;
- personality or inner personality traits.

Observable content may still be reported. For example, visible religious-themed content can be described as content; it must not be converted into a claim about the person's religion.

---

## Architecture

The required dependency direction is:

```text
Host -> Application -> Core -> Infrastructure
```

The standard request dependency flow is:

```text
Controller -> Service -> Repository -> PsqlDbContext
```

### Responsibilities

#### Host

Application startup and dependency bootstrap.

```text
src/host/
```

#### Application

HTTP/controller and DTO concerns.

```text
src/application/
```

Controllers own:

- HTTP request handling;
- DTO validation;
- response mapping;
- HTTP status codes;
- `HTTPException` mapping.

Business rules do not belong in controllers.

#### Core

Product contracts and business workflow.

```text
src/core/
```

This includes:

- image processing orchestration;
- OCR evidence normalization;
- evidence aggregation;
- insight generation;
- lifecycle rules;
- observable/sensitive-inference boundaries.

#### Infrastructure

Runtime-specific concerns.

```text
src/infrastructure/
```

This includes:

- PostgreSQL and SQLAlchemy;
- repositories;
- Alembic;
- local image storage;
- Tesseract;
- model/runtime providers;
- configuration;
- DI/discovery implementation.

### Dependency injection

The project uses constructor injection and runtime discovery. Dependency-bearing components use the repository's `@inject` convention.

Do not add manual provider registration when discovery already applies.

---

## Repository structure

```text
profile-insight-engine/
├─ src/
│  ├─ host/                  # app bootstrap and appsettings
│  ├─ application/           # FastAPI controllers and DTOs
│  ├─ core/                  # contracts, services, business policy
│  └─ infrastructure/        # DB, repositories, providers, DI, Alembic
├─ evaluation/               # dataset contracts and benchmark runners
├─ scripts/                  # local verification / hardware collection
├─ tests/                    # unit, integration, architecture, regression tests
├─ docs/                     # phase-by-phase implementation documentation
├─ requirements.txt
└─ README.md
```

---

## Requirements

### Accepted/validated environment

Final acceptance evidence was produced on:

- Windows 11;
- CPython 3.12.10;
- 8 logical CPU threads;
- approximately 16 GB RAM;
- no NVIDIA GPU;
- Docker 29.x for disposable PostgreSQL verification;
- Tesseract 5.4.x;
- PostgreSQL 17 in verification.

The code may run on other environments, but those environments are not part of the current accepted benchmark profile.

### Runtime requirements

Required for the accepted production profile:

- Python and dependencies from `requirements.txt`;
- PostgreSQL;
- Tesseract 5 available on `PATH`;
- official `tessdata_best` traineddata for:
  - `fas`;
  - `eng`;
  - `osd`;
- two environment variables:
  - `PROFILE_INSIGHT_DATABASE_URL`;
  - `PROFILE_INSIGHT_TESSDATA_DIR`.

Docker is optional for the application itself. It is used by the repository verification scripts to create disposable PostgreSQL instances.

---

## Quick start

The commands below use PowerShell because the accepted deployment was verified on Windows.

### 1. Clone the repository

```powershell
git clone https://github.com/Amirmahdimtz/profile-insight-engine.git
cd profile-insight-engine
```

### 2. Create a Python virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Install Tesseract

Install Tesseract 5 and make sure the executable is available:

```powershell
tesseract --version
```

If the command is not found, add the Tesseract installation directory to `PATH` before starting the service.

### 4. Download the selected OCR traineddata

The accepted OCR profile uses official `tessdata_best` files.

Example:

```powershell
$ModelRoot = "E:\profile-insight-models"
$TessdataBest = Join-Path $ModelRoot "tessdata_best"

New-Item -ItemType Directory -Force -Path $TessdataBest | Out-Null

foreach ($Language in @("fas","eng","osd")) {
    Invoke-WebRequest `
      -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$Language.traineddata" `
      -OutFile (Join-Path $TessdataBest "$Language.traineddata")
}

$env:PROFILE_INSIGHT_TESSDATA_DIR = $TessdataBest

tesseract `
  --tessdata-dir "$env:PROFILE_INSIGHT_TESSDATA_DIR" `
  --list-langs
```

The language list must contain `fas`, `eng`, and `osd`.

### 5. Start PostgreSQL

You can use an existing PostgreSQL instance.

For a disposable local instance with Docker:

```powershell
$DbPassword = [Guid]::NewGuid().ToString("N")

docker run `
  --name profile-insight-postgres `
  -e "POSTGRES_USER=postgres" `
  -e "POSTGRES_PASSWORD=$DbPassword" `
  -e "POSTGRES_DB=profile_insight" `
  -p "127.0.0.1:5432:5432" `
  -d postgres:17-alpine

$env:PROFILE_INSIGHT_DATABASE_URL = `
  "postgresql+asyncpg://postgres:$DbPassword@127.0.0.1:5432/profile_insight"
```

For a real deployment, use your normal secret-management mechanism. Do not commit credentials.

### 6. Apply migrations

```powershell
python -m alembic `
  -c src/infrastructure/alembic.ini `
  upgrade head

python -m alembic `
  -c src/infrastructure/alembic.ini `
  check
```

Expected current migration head:

```text
20261005_0001
```

### 7. Start the API

```powershell
python -m uvicorn `
  src.host.app:app `
  --host 127.0.0.1 `
  --port 5000 `
  --workers 1
```

Use exactly one worker for the accepted topology. Startup recovery ownership is intentionally single-worker.

### 8. Smoke test

In another shell:

```powershell
Invoke-RestMethod `
  http://127.0.0.1:5000/openapi.json |
  Select-Object -ExpandProperty info
```

The FastAPI title is:

```text
Profile Insight Engine
```

Interactive OpenAPI documentation is normally available at:

```text
http://127.0.0.1:5000/docs
```

---

## Configuration

Configuration files:

```text
src/host/res/appsettings.yaml
src/host/res/appsettings.development.yaml
```

By default the service loads `appsettings.yaml`.

Set:

```powershell
$env:env_type = "development"
```

before process startup to load `appsettings.development.yaml`.

### Important production defaults

| Setting | Accepted value |
|---|---|
| `profile_analysis.max_images` | 50 |
| `profile_analysis.max_image_size_mb` | 12 MB per image |
| `profile_analysis.max_image_pixels` | 40,000,000 |
| `image.max_dimension` | 2048 |
| `image.allowed_mime_types` | JPEG, PNG, WebP |
| `ocr.provider` | Tesseract |
| `ocr.languages` | fas + eng |
| `ocr.page_segmentation_mode` | 1 |
| `ocr.timeout_seconds` | 30 |
| `ocr.model_id` | tessdata_best-fas-eng |
| `ocr.tessdata_dir_env` | PROFILE_INSIGHT_TESSDATA_DIR |
| `vision.enabled` | false |
| `api.prefix` | /api/v1 |
| `database.url_env` | PROFILE_INSIGHT_DATABASE_URL |

### Important

Candidate Vision and Embedding settings remain in appsettings for benchmark/history purposes.

Their presence does not mean they are active.

For the accepted release:

```text
vision.enabled = false
SemanticThemeService is not in the HTTP workflow
```

---

## Running the API

Base URL for the default local setup:

```text
http://127.0.0.1:5000
```

API prefix:

```text
/api/v1
```

Profile-analysis routes:

```text
POST /api/v1/profile_analysis/
GET  /api/v1/profile_analysis/{analysis_id}
GET  /api/v1/profile_analysis/{analysis_id}/result
```

There is intentionally no DELETE endpoint in the current API surface.

---

## API reference

### Create and run an analysis

```text
POST /api/v1/profile_analysis/
Content-Type: multipart/form-data
```

The request contains:

1. a form field named `request_json`;
2. one or more repeated file fields named `images`.

The JSON contract is:

```json
{
  "analysis_id": "analysis-demo-001",
  "image_ids": [
    "img-001",
    "img-002"
  ]
}
```

Rules:

- `analysis_id` must be non-empty and trimmed;
- image IDs must be non-empty, trimmed, and unique;
- the number of uploaded files must exactly match `image_ids`;
- uploaded files are associated with image IDs by multipart order;
- the current maximum is 50 images;
- an existing `analysis_id` cannot be reused.

#### curl example

```powershell
$request = '{"analysis_id":"analysis-demo-001","image_ids":["img-001","img-002"]}'

curl.exe -X POST `
  "http://127.0.0.1:5000/api/v1/profile_analysis/" `
  -F "request_json=$request" `
  -F "images=@C:\images\profile-1.jpg;type=image/jpeg" `
  -F "images=@C:\images\profile-2.png;type=image/png"
```

The create endpoint currently executes the analysis synchronously and returns the completed sanitized result with HTTP 201 when successful.

### Get analysis status

```text
GET /api/v1/profile_analysis/{analysis_id}
```

Example:

```powershell
curl.exe `
  "http://127.0.0.1:5000/api/v1/profile_analysis/analysis-demo-001"
```

Example response:

```json
{
  "analysis_id": "analysis-demo-001",
  "status": "completed",
  "image_ids": [
    "img-001",
    "img-002"
  ]
}
```

Possible statuses:
```text
pending
in_progress
completed
failed
```

### Get completed result

```text
GET /api/v1/profile_analysis/{analysis_id}/result
```

Example:

```powershell
curl.exe `
  "http://127.0.0.1:5000/api/v1/profile_analysis/analysis-demo-001/result"
```

If the analysis exists but is not completed, the endpoint returns HTTP 409.

### Main HTTP error behavior

| Status | Typical cause |
|---|---|
| 201 | analysis completed successfully |
| 404 | analysis ID not found |
| 409 | duplicate analysis ID, invalid lifecycle transition, or result not ready |
| 413 | uploaded image exceeds configured file-size limit |
| 422 | invalid DTO, image count mismatch, corrupt/invalid image, unsupported input |
| 500 | sanitized internal analysis/persistence execution failure |

---

## Response and evidence model

The result DTO contains:

```json
{
  "analysis_id": "analysis-demo-001",
  "status": "completed",
  "images": [],
  "themes": [],
  "insights": [],
  "insight_policy_version": "phase8-candidate-v1",
  "summary": "No supported insights met the configured evidence policy."
}
```

Each image can contain structured evidence.

Example persisted OCR evidence:

```json
{
  "id": "ocr-...",
  "image_id": "img-001",
  "type": "ocr_text",
  "label": "ocr_text",
  "value": null,
  "confidence": 0.91,
  "source": "tesseract",
  "metadata": {
    "block_order": 0,
    "script_hint": "arabic",
    "language_hint": "fa",
    "provider_version": "tesseract 5.x",
    "model_id": "tessdata_best-fas-eng",
    "config_version": "phase10-ocr-tessdata-best-v1"
  }
}
```

The `value` is intentionally `null` after persistence because normalized OCR content is redacted before storage.

### Insight contract

When an insight is supported, it includes:

```json
{
  "key": "textual_reference_...",
  "type": "textual_reference",
  "label": "Recurring textual reference: ...",
  "explanation": "Supported by ...",
  "confidence": 0.82,
  "evidence_count": 2,
  "image_coverage": 0.5,
  "supporting_evidence_ids": [
    "ocr-...",
    "ocr-..."
  ],
  "supporting_image_ids": [
    "img-001",
    "img-002"
  ]
}
```

With the current default policy an aggregate must have at least:

- 2 evidence signals;
- support from 2 unique images;
- 10% image coverage;
- at least 1 source.

This is why a single OCR observation does not independently become a profile insight.

---

## Image validation and preprocessing

The preprocessing path is deterministic and defensive.

Current limits:

- maximum 50 images per analysis;
- maximum 12 MB per image;
- maximum 40,000,000 decoded pixels;
- maximum canonical dimension 2048;
- JPEG, PNG, and WebP only.

The service also:

- validates decoded image format rather than trusting MIME alone;
- checks file-extension/decoded-format consistency;
- rejects corrupt images;
- applies EXIF orientation;
- converts to canonical RGB;
- flattens transparency against white;
- resizes while preserving aspect ratio when needed;
- encodes deterministic canonical PNG;
- hashes canonical image content;
- detects exact duplicate canonical content;
- prevents exact duplicate uploads from artificially increasing cross-image support.

Canonical image files are temporary and are released after analysis.

---

## Privacy, persistence, and safety boundaries

### Raw image persistence

Raw upload bytes are not stored in PostgreSQL.

Canonicalized image files are temporary local artifacts owned by the processing scope.

### OCR persistence

Before persistence:

- OCR evidence `value` is replaced with `null`;
- `metadata.raw_text` is removed.

Traceability fields such as evidence ID, image ID, confidence, provider metadata, and final insight support references are preserved.

### Logging

Production logic must not log:

- raw image bytes;
- canonical image bytes;
- raw provider payloads;
- raw OCR text;
- database credentials.

### Final textual insights

Supported final textual-reference insights may be persisted because they are validated product output derived from the evidence graph. This is different from persisting block-level raw OCR text.

---

## Lifecycle and recovery

The persistence lifecycle is:

```text
pending -> in_progress -> completed
                    \-> failed
```

Completed rows require an atomic persisted result.

Failed analyses do not keep a partial completed-result payload.

### Startup recovery

On application startup:

1. stale local image-processing scopes are removed;
2. unfinished database rows are recovered;
3. leftover `pending`/`in_progress` rows are marked `failed`.

Interrupted work is not resumed automatically.

This recovery design is intentionally validated for one Uvicorn worker.

Do not deploy the accepted release with:

```text
--workers > 1
```

without redesigning and revalidating startup/recovery ownership.

---

## Testing and verification

### Full regression suite

```powershell
python -m unittest `
  discover `
  -s tests `
  -p "test_*.py" `
  -q
```

### Phase 10 implementation verifier

```powershell
python scripts/verify_phase10.py
```

This verifier checks, among other things:

- Docker availability;
- disposable password-authenticated PostgreSQL;
- compile/import;
- Alembic upgrade;
- Alembic metadata drift;
- focused Phase 10 tests;
- PostgreSQL recovery regression;
- hardware/runtime evidence collection;
- fail-closed final-report scaffold behavior;
- full repository regression;
- tracked Git cleanliness.

Important: the verifier's internal acceptance report is intentionally a scaffold with incomplete evidence. Real project acceptance comes from the separately aggregated benchmark evidence.

### Accepted verification baseline

The final accepted runtime/evidence baseline was verified before the documentation-only closure commit:

- focused Phase 10: 63 tests passed;
- recovery regression: 5 tests passed;
- full repository regression: 362 tests passed;
- Alembic check: passed;
- E2E failure rate: 0.0;
- unsupported claim rate: 0.0;
- structural summary factuality: true;
- final `acceptance_ready=true`;
- final `missing_evidence=[]`.

The final evidence run was produced against commit:

```text
01fcf80ade2ffbc701b066281e262d25b1689b11
```

Subsequent closure/documentation changes did not change the accepted runtime behavior.

---

## Benchmarks and evaluation

Evaluation tooling lives under:

```text
evaluation/
```

Important runners include:

| Runner | Purpose |
|---|---|
| `evaluation.ocr_benchmark` | OCR quality/performance comparison |
| `evaluation.vision_benchmark` | earlier Vision evaluation |
| `evaluation.transformers_vision_benchmark` | structured Transformers Vision candidate |
| `evaluation.prepare_phase10_holdout` | deterministic label-blind Phase 10 TRAIN/EVAL derivation |
| `evaluation.siglip_zero_shot_vision_benchmark` | final zero-shot Vision holdout evaluation |
| `evaluation.embedding_benchmark` | embedding/retrieval/theme evidence |
| `evaluation.aggregation_benchmark` | cross-image aggregation |
| `evaluation.insight_benchmark` | insight policy contract |
| `evaluation.phase9_api_persistence_benchmark` | API/database evidence |
| `evaluation.phase10_end_to_end_benchmark` | real selected service-path E2E evidence |
| `evaluation.phase10_benchmark` | final acceptance evidence aggregator |

Generated benchmark JSON and local diagnostic logs are evidence artifacts. They do not need to be committed to source control unless your organization explicitly chooses to archive them there.

### Final accepted evidence profile

Final acceptance uses:

```text
deployment_profile = degraded_ocr_only
```

Final production selections:

```text
OCR        = tesseract_tessdata_best_fas_eng
Vision     = disabled_rejected_by_holdout_evidence
Embedding  = not_in_active_http_workflow
Thresholds = phase8_versioned_policy_no_posthoc_tuning
```

Measured target-host snapshot:

- E2E failure rate: 0.0;
- unsupported-claim rate after validation: 0.0;
- structural summary factuality: true;
- p50 total latency: approximately 26.0 seconds;
- p95 total latency: approximately 89.8 seconds;
- throughput: approximately 0.495 images/second;
- peak service-process RAM: approximately 80.27 MB.

These figures describe the accepted target workstation and are not universal performance guarantees.

End-to-end accuracy, insight precision, and confidence calibration remain unavailable because the authorized dataset does not contain the annotation types required to measure them. The project intentionally does not fabricate these metrics.

---

## Production deployment checklist

Before serving traffic:

- [ ] Python dependencies installed from `requirements.txt`;
- [ ] Tesseract 5 installed and on PATH;
- [ ] `tessdata_best` contains `fas`, `eng`, and `osd`;
- [ ] `PROFILE_INSIGHT_TESSDATA_DIR` set;
- [ ] PostgreSQL available;
- [ ] `PROFILE_INSIGHT_DATABASE_URL` set using `postgresql+asyncpg://`;
- [ ] Alembic upgraded to head;
- [ ] Alembic metadata check passes;
- [ ] `vision.enabled=false` remains unchanged;
- [ ] API started with one Uvicorn worker;
- [ ] `/openapi.json` responds;
- [ ] POST/status/result profile-analysis flow smoke-tested;
- [ ] database credentials are managed outside the repository;
- [ ] machine-specific model paths are not committed.

Recommended production command for the accepted topology:

```powershell
python -m uvicorn `
  src.host.app:app `
  --host 127.0.0.1 `
  --port 5000 `
  --workers 1
```

Bind to a different interface only according to your organization's network/reverse-proxy/security design.

---

## Troubleshooting

### Tesseract executable was not found

Check:

```powershell
Get-Command tesseract
tesseract --version
```

Make sure the Tesseract installation directory is on PATH.

### OCR tessdata environment variable is not set

Set:

```powershell
$env:PROFILE_INSIGHT_TESSDATA_DIR = "E:\profile-insight-models\tessdata_best"
```

Then verify:

```powershell
tesseract `
  --tessdata-dir "$env:PROFILE_INSIGHT_TESSDATA_DIR" `
  --list-langs
```

### Required languages are missing

The selected profile expects:

```text
fas
eng
osd
```

Download the official `tessdata_best` files and verify the configured directory.

### Database URL error

The application expects the environment-variable name configured by:

```yaml
database:
  url_env: PROFILE_INSIGHT_DATABASE_URL
```

The URL must use the async SQLAlchemy dialect:

```text
postgresql+asyncpg://
```

Example shape:

```text
postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
```

### Alembic reports schema drift

Run:

```powershell
python -m alembic -c src/infrastructure/alembic.ini upgrade head
python -m alembic -c src/infrastructure/alembic.ini check
```

Do not manually patch the database schema around Alembic.

### HTTP 413

An uploaded image exceeds the configured 12 MB per-image limit.

### HTTP 422

Typical causes:

- invalid `request_json`;
- duplicate image IDs;
- image count mismatch;
- unsupported MIME/format;
- corrupt image;
- filename extension does not match decoded format;
- pixel-count or other image validation failure.

### HTTP 409 on POST

The `analysis_id` already exists or a lifecycle transition conflicted.

Use a new unique analysis ID.

### HTTP 409 on result endpoint

The analysis exists but is not completed.

Inspect the status endpoint first.

### Analysis becomes failed after restart

This can be expected.

The accepted recovery contract marks interrupted `pending`/`in_progress` rows as `failed` instead of attempting unsafe implicit resume.

### Should I start llama.cpp?

No, not for the accepted release.

Vision is intentionally disabled and llama.cpp is not a production dependency.

---

## Known limitations and deferred capabilities

### Vision is intentionally unavailable in production

The repository contains Vision-provider implementations and benchmark history, but the accepted release does not expose Vision-derived scene/object/activity/topic evidence.

Re-enabling it requires new quality evidence and a new production-selection review.

### Semantic embedding themes are not in the current API workflow

The embedding provider exists and has evaluation coverage, but the production HTTP contract currently has no accepted source of caller-supplied theme labels.

### Insight quality metrics are partially unavailable

There is no authorized insight-level labeled dataset for product-level insight precision or confidence calibration.

Those values remain explicitly unavailable.

### Create-analysis is synchronous

The POST request performs the analysis in the request lifecycle. There is currently no external queue or asynchronous job broker.

### Single-worker topology

Startup recovery is validated for one process worker. Multi-worker deployment requires a deliberate recovery/ownership design change.

### Target-platform validation

The accepted deployment was benchmarked on Windows 11 CPU-only hardware. Other operating systems or hardware profiles require their own operational validation.

---

## Project documentation

Detailed implementation history and contracts are in `docs/`:

- `docs/phase_1_analysis_contract.md`
- `docs/phase_2_evaluation_foundation.md`
- `docs/phase_3_image_ingestion_preprocessing.md`
- `docs/phase_4_ocr_text_evidence.md`
- `docs/phase_5_visual_understanding.md`
- `docs/phase_6_embedding_semantic_themes.md`
- `docs/phase_7_cross_image_aggregation.md`
- `docs/phase_8_insight_engine.md`
- `docs/phase_9_api_persistence_lifecycle.md`
- `docs/phase_10_benchmark_hardening_local_deployment.md`

For deployment and final production-selection evidence, start with:

```text
docs/phase_10_benchmark_hardening_local_deployment.md
```

---

## License

This repository currently does not contain a LICENSE file.

Before redistribution, external publication, or third-party commercial use, confirm licensing terms with the repository owner.