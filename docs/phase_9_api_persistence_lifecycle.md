# Phase 9 — API, Persistence & Lifecycle

## Scope

Phase 9 exposes the verified Phase 3–8 pipeline through the repository's standard layered runtime:

`HTTP multipart request → ProfileAnalysisController → ProfileAnalysisService → prior Core services → ProfileAnalysisRepository → PostgreSQL`

This phase adds the Application layer, FastAPI host wiring, PostgreSQL persistence, lifecycle enforcement, and Alembic migration required by the official Phase 9 contract. It does not add Phase 10 deployment/hardening features, a queue framework, a retry scheduler, a production-final ML model choice, or new inference categories.

## API contract

Routes are discovered through `src/application/web.py` and mounted under the configured `api.prefix` plus the feature folder:

- `POST /api/v1/profile_analysis/` — multipart request with `request_json` and repeated `images`; runs the existing pipeline and returns a completed, persisted result.
- `GET /api/v1/profile_analysis/{analysis_id}` — returns lifecycle status and input image IDs.
- `GET /api/v1/profile_analysis/{analysis_id}/result` — returns the completed result or HTTP 409 while the analysis is not completed.

`DELETE` is intentionally absent because the official Phase 9 plan makes it conditional on an explicit requirement and the repository has no deletion/lifecycle requirement for analyses.

The multipart `request_json` object contains exactly:

```json
{"analysis_id":"analysis-1","image_ids":["img-1","img-2"]}
```

Uploaded files are associated positionally with `image_ids`. The controller enforces the configured per-image upload byte limit while reading `UploadFile`; Phase 3 independently revalidates MIME, decoded format, dimensions, pixels, corruption, and image-count limits.

## Lifecycle

Phase 9 uses the existing Phase 1 `ProfileAnalysisStatus` vocabulary only:

`pending → in_progress → completed`

or

`pending → in_progress → failed`

Terminal states have no outgoing transition. `ProfileAnalysisService` owns the transition rules. `ProfileAnalysisRepository` only performs conditional compare-and-set updates against the expected persisted status, so stale/concurrent transitions fail deterministically.

A completed transition requires its result payload in the same SQL statement. Non-completed states are constrained to `result_payload IS NULL`. A failed analysis therefore cannot expose a partial result as successful.

## Orchestration of prior phases

`ProfileAnalysisService.create_async()` reuses the existing services directly:

1. `ImageProcessingService.process_async()`
2. `OcrEvidenceService.extract_async()`
3. `EvidenceExtractionService.extract_async()`
4. `EvidenceAggregationService.aggregate()`
5. `InsightGenerationService.generate()`
6. validated `ProfileAnalysisResult`
7. privacy sanitization and atomic result persistence
8. `ImageProcessingService.release_async()` for temporary image scope cleanup before completion is committed

`SemanticThemeService` is not invoked by the Phase 9 HTTP workflow. Its verified Phase 6 contract requires caller-supplied candidate observable theme labels, while no production runtime source for those labels exists in the current API/product contract. Phase 7 already makes semantic input optional, and Phase 8 does not promote semantic-only aggregates to `ProfileInsight`. Deriving labels from OCR/model output here would create an unbenchmarked policy outside Phase 9, so no such policy is invented.

## Persistence contract

One `profile_analysis` row stores lifecycle state and a versioned JSONB completed-result snapshot:

- `analysis_id` primary key
- `status`
- `image_ids`
- nullable `result_payload`
- creation/modification timestamps

The result snapshot schema version is `phase9-analysis-result-v1`. It contains the validated `ProfileAnalysisResult`, Phase 8 insight policy version, and deterministic Phase 8 summary. No separate parallel evidence or insight schema is introduced.

The snapshot preserves `ProfileInsight.evidence_count`, `image_coverage`, `confidence`, `supporting_evidence_ids`, `supporting_image_ids`, and `explanation`. Deserialization reconstructs the Core contracts, which revalidates the support graph and sensitive-inference boundary before a persisted result is returned. It also requires the persisted summary to equal the deterministic Phase 8 summary derived from the accepted Insight labels, so storage corruption or tampering cannot introduce a new claim through the summary field.

Raw image bytes, canonical image bytes, storage references, and raw provider payloads are not persisted in PostgreSQL.

OCR content receives an additional persistence privacy boundary: normalized OCR `Evidence.value` is replaced with `null` and `metadata.raw_text` is removed before persistence. Evidence ID, image ID, confidence, source, non-content metadata, and final supported Insight references remain available for traceability. The API returns this sanitized persisted form. A final validated Phase 8 textual-reference Insight (and the evidence-backed summary built from it) is still retained as product output; this redaction targets block-level OCR raw/normalized text, not supported final Insights.

## Database and migration

Database configuration stores no credential in repository configuration. Both appsettings files define only:

```yaml
database:
  url_env: PROFILE_INSIGHT_DATABASE_URL
```

The named environment variable must contain a SQLAlchemy async PostgreSQL URL using `postgresql+asyncpg://`.

Alembic revision `20261005_0001` creates the table, status index, status vocabulary constraint, and result/status consistency constraint. The migration has a real downgrade that drops the Phase 9 table and index.

## Benchmark / evidence

`python -m evaluation.phase9_api_persistence_benchmark` records:

- API status/result p50/p95 for controller/DTO serialization using a deterministic in-process service fixture;
- PostgreSQL atomic completion write p50/p95;
- PostgreSQL read p50/p95;
- serialized analysis graph byte size;
- concurrent PostgreSQL read behavior.

Its schema is `phase9-api-persistence-benchmark-v1` and its scope is explicitly `phase9_contract_and_persistence_not_end_to_end_ml`. It does not claim full ML pipeline latency; end-to-end performance/hardening belongs to Phase 10.

## Local verification

Run from the repository root in PowerShell.

### 1. Synchronize and install

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -r requirements.txt
```

### 2. Configure a disposable Phase 9 PostgreSQL database

Use an existing disposable PostgreSQL database, or start one with Docker. Do not use a production database for migration downgrade/integration tests.

```powershell
docker rm -f -v profile-insight-phase9-postgres 2>$null | Out-Null

$env:PHASE9_DB_PASSWORD = (
  [guid]::NewGuid().ToString("N") +
  [guid]::NewGuid().ToString("N")
)

docker run --name profile-insight-phase9-postgres `
  --env "POSTGRES_USER=postgres" `
  --env "POSTGRES_PASSWORD=$env:PHASE9_DB_PASSWORD" `
  --env "POSTGRES_DB=profile_insight_phase9_test" `
  -p 55432:5432 `
  -d postgres:17-alpine
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL container startup failed" }

$DatabaseReady = $false
for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
  docker exec profile-insight-phase9-postgres pg_isready -U postgres -d profile_insight_phase9_test | Out-Null
  if ($LASTEXITCODE -eq 0) {
    $DatabaseReady = $true
    break
  }
  Start-Sleep -Seconds 1
}
if (-not $DatabaseReady) {
  docker logs profile-insight-phase9-postgres
  throw "PostgreSQL did not become ready"
}

$env:PROFILE_INSIGHT_DATABASE_URL = "postgresql+asyncpg://postgres:$env:PHASE9_DB_PASSWORD@127.0.0.1:55432/profile_insight_phase9_test"
$env:PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL = $env:PROFILE_INSIGHT_DATABASE_URL

python -m alembic -c src/infrastructure/alembic.ini current
if ($LASTEXITCODE -ne 0) {
  docker logs profile-insight-phase9-postgres
  throw "PostgreSQL authentication preflight failed"
}
```

If PostgreSQL is already available, set the same two environment variables to the disposable database URL instead.

### 3. Compile/import and focused non-DB tests

```powershell
python -m compileall -q src evaluation tests
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
python -m unittest tests.test_phase9_contracts tests.test_profile_analysis_service tests.test_profile_analysis_controller tests.test_phase9_architecture tests.test_phase9_di_discovery tests.test_phase9_documentation -v
if ($LASTEXITCODE -ne 0) { throw "Phase 9 focused tests failed" }
```

### 4. Migration metadata, upgrade and check

```powershell
python -m alembic -c src/infrastructure/alembic.ini current
python -m alembic -c src/infrastructure/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { throw "Alembic upgrade failed" }
python -m alembic -c src/infrastructure/alembic.ini current
python -m alembic -c src/infrastructure/alembic.ini check
if ($LASTEXITCODE -ne 0) { throw "Alembic metadata is not in sync" }
```

### 5. PostgreSQL integration tests

These tests use the already-migrated Phase 9 table in the database referenced by `PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL` and delete only their test rows before/after the suite. The database must still be disposable because the later migration downgrade is destructive.

```powershell
python -m unittest tests.test_profile_analysis_repository_integration -v
if ($LASTEXITCODE -ne 0) { throw "Phase 9 PostgreSQL integration tests failed" }
```

### 6. Destructive migration downgrade/upgrade verification — disposable DB only

The next command drops the `profile_analysis` table. Run it only against the disposable Phase 9 database configured above.

```powershell
python -m alembic -c src/infrastructure/alembic.ini downgrade base
if ($LASTEXITCODE -ne 0) { throw "Alembic downgrade failed" }
python -m alembic -c src/infrastructure/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { throw "Alembic re-upgrade failed" }
python -m alembic -c src/infrastructure/alembic.ini current
```

### 7. DI/discovery and route smoke check

```powershell
python -c "from src.infrastructure.di.bootstrap import bootstrap_di; bootstrap_di(); from src.application.web import WebService; from src.infrastructure.di.inject import resolve; app=resolve(WebService).create_app(); spec=app.openapi(); methods={'get','post','put','patch','delete'}; print(sorted((method.upper(), path) for path, operations in spec['paths'].items() for method in operations if method in methods and path.startswith('/api/')))"
if ($LASTEXITCODE -ne 0) { throw "DI/discovery route smoke check failed" }
```

Expected application routes include:

- `POST /api/v1/profile_analysis/`
- `GET /api/v1/profile_analysis/{analysis_id}`
- `GET /api/v1/profile_analysis/{analysis_id}/result`

### 8. Phase 9 API/persistence benchmark

```powershell
python -m evaluation.phase9_api_persistence_benchmark --iterations 20 --concurrency 8 --output phase9_api_persistence_benchmark.json
if ($LASTEXITCODE -ne 0) { throw "Phase 9 benchmark failed" }
$Report = Get-Content .\phase9_api_persistence_benchmark.json -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,metric_scope,iterations,analysis_graph_size_bytes
$Report.api | Format-List
$Report.database | Format-List
```

Expected `schema_version` is `phase9-api-persistence-benchmark-v1` and `metric_scope` is `phase9_contract_and_persistence_not_end_to_end_ml`.

### 9. Full regression and repository state

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { throw "full regression failed" }
git status --short
git rev-parse HEAD
```

Generated benchmark JSON may appear as an untracked local artifact. No tracked production/test/config/documentation change should remain after verification.

When verification is complete, remove the disposable container and its anonymous volume with `docker rm -f -v profile-insight-phase9-postgres`. Clear the three Phase 9 database environment variables from the shell afterward.

## Expected result

- compile/import checks pass;
- focused API/DTO/service/lifecycle/architecture tests pass;
- repository integration tests pass on the disposable PostgreSQL database;
- Alembic upgrade/check/downgrade/re-upgrade passes;
- DI discovery exposes exactly the Phase 9 profile-analysis routes without manual provider registration;
- persisted results preserve the Insight support graph while raw image bytes and block-level OCR raw/normalized text are absent from persistence; validated final textual-reference Insights may remain as supported product output;
- invalid/stale lifecycle transitions fail deterministically;
- failed analyses have no result payload;
- full repository regression passes;
- benchmark reports Phase 9 API/DB latency, graph size and concurrent-read evidence without claiming Phase 10 end-to-end ML performance.

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

Phase 9 implementation is committed only after repository-side checks and diff review. It must not be marked `COMPLETE / READY_FOR_NEXT_PHASE` until the target local verification evidence above is returned and accepted.
