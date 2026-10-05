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

Run from the repository root. The verification path intentionally avoids shell-managed database credentials because repeated local runs showed that stale/mismatched PostgreSQL credentials can make every later DB check fail with the same secondary error.

Prerequisites:

- Python dependencies from `requirements.txt` are installed.
- Docker Desktop / Docker Engine is running.
- The current branch is `main` and tracked files are clean.

### 1. Synchronize and install

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -r requirements.txt
```

### 2. Run the single fail-fast verifier

```powershell
python scripts/verify_phase9.py
if ($LASTEXITCODE -ne 0) { throw "Phase 9 local verification failed" }
```

The verifier owns the disposable database lifecycle and stops at the first failing check. It:

1. verifies `main` and rejects tracked local modifications;
2. verifies the Docker daemon is reachable;
3. chooses an unused loopback port and starts a uniquely named `postgres:17-alpine` container;
4. enables `POSTGRES_HOST_AUTH_METHOD=trust` **only inside that disposable test container**, binds PostgreSQL only to `127.0.0.1`, and does not create/store a database password;
5. supplies `PROFILE_INSIGHT_DATABASE_URL` and `PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL` only to verifier child processes;
6. performs an actual host-side Alembic connection preflight before any DB-dependent tests;
7. runs compile/import checks and focused Phase 9 contract/service/controller/architecture/DI/documentation/verifier tests;
8. applies the real Alembic migration and checks metadata drift;
9. runs PostgreSQL repository integration tests;
10. verifies destructive downgrade followed by re-upgrade on the disposable database;
11. verifies the discovered HTTP surface from the generated OpenAPI document;
12. runs the Phase 9 API/persistence benchmark and validates its schema/scope;
13. runs the entire repository regression suite in quiet mode so successful test names do not flood the terminal;
14. verifies tracked Git state again and always removes the disposable PostgreSQL container in `finally`.

The test-only `trust` setting is deliberately scoped to a short-lived container published on loopback only. Production/runtime database configuration remains unchanged and still comes from the configured environment variable.

If a step fails, do not continue with later commands manually. The verifier exits immediately with a section name such as `Database host-connection preflight`, `PostgreSQL integration tests`, or `Full repository regression`; return that output for diagnosis. This prevents one infrastructure failure from producing a long cascade of misleading secondary errors.

The benchmark report is written to `phase9_api_persistence_benchmark.json`. It may remain as an untracked local evidence artifact; tracked files must remain clean.

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
