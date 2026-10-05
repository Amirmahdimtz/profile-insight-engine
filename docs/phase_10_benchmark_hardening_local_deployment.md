# Phase 10 — Benchmark, Hardening & Local Deployment

## Starting baseline

Phase 10 starts from `main` commit `594187e15b692944e126c485b7d9c1b4383cce1d`, the verified Phase 9 closure commit. Phase 9 is `COMPLETE / READY_FOR_NEXT_PHASE`. Its benchmark scope is `phase9_contract_and_persistence_not_end_to_end_ml`, so those API/DB measurements are not treated as Phase 10 end-to-end ML evidence.

## Scope and architecture

The production dependency direction remains:

`Host → Application → Core → Infrastructure`

The current HTTP workflow remains:

`multipart input → ImageProcessingService → OCR → Vision → Cross-Image Aggregation → InsightGenerationService → privacy sanitization → PostgreSQL`

`SemanticThemeService` is intentionally not inserted into the HTTP workflow. Phase 6 requires caller-supplied observable theme labels and the production API still has no benchmarked source for them. Inventing one in Phase 10 would change the product contract.

## Gate 0

| Input | Status | Evidence / gap |
| --- | --- | --- |
| Full pipeline | `AVAILABLE` | Phase 9 exposes the verified ingestion/OCR/Vision/aggregation/insight/persistence path. |
| Evaluation dataset | `PARTIAL` | Authorized images are outside the repository. Dataset identity/fingerprints are known, but insight-level correctness annotations are unavailable. |
| Candidate model configs | `PARTIAL` | OCR, Vision and Embedding candidates exist but none is production-final. |
| Target hardware | `PARTIAL` | Prior Windows evidence exists; a versioned hardware/runtime profile still must be returned for Phase 10. |

Authorized dataset identity carried forward from prior verified phases:

- dataset: `profile_insight_real_eval` v`1.0.0`
- manifest fingerprint: `06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc`
- content fingerprint: `5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e`

## Candidate evidence carried forward

OCR Phase 4 did not establish a universal winner: `tessdata_best` was slightly better on overall/Persian CER/WER, while `tessdata_fast` had materially lower p95 latency and slightly better mixed-script WER.

Vision Phase 5 is negative selection evidence for the current candidates. Gemma 3 4B failed the offline-load preflight. Qwen2.5-VL 3B completed a reduced evaluation but had 13/18 request timeouts, overall F1 `0.0`, p50 around `276993 ms`, and p95 around `298460 ms`. The current Vision candidate has previously shown very high latency/timeouts and is not production-final.

Embedding Phase 6 measured strong retrieval for `google/siglip2-base-patch16-224` (Recall@5 about `0.9907`, mAP about `0.9183`) but theme Macro F1 at threshold `0.25` was `0.0`. Model/threshold selection therefore remains unresolved.

Phase 8 has deterministic contract evidence and structural summary factuality, but no authorized insight-level correctness labels. Product insight precision and confidence calibration remain `unavailable`.

## Benchmark contracts

`evaluation.phase10_end_to_end_benchmark` executes the real production service path using the authorized dataset, real configured OCR/Vision providers and PostgreSQL. It uses no fake provider.

Default workload sizes are `1, 5, 20, 50, 100`. With current `profile_analysis.max_images=50`, the 100-image case is recorded as `expected_rejection_by_config`, not as fabricated inference performance.

The E2E report records per-workload p50/p95, throughput, failure behavior, deterministic reruns, post-validation unsupported-claim rate, structural summary factuality, concurrent requests/request isolation and service-process RAM where measurable. Metrics lacking valid labels or hardware telemetry remain `unavailable`.

Schema: `phase10-end-to-end-benchmark-v1`.

`evaluation.phase10_benchmark` aggregates OCR, Vision, Embedding, Insight, API/persistence, E2E and hardware evidence into schema `phase10-final-benchmark-v1`. It never auto-promotes a model/config. Missing metrics remain machine-readable `unavailable`; production selections remain `unresolved_requires_phase10_evidence_review` until evidence is reviewed.

Required failure matrix includes GPU unavailable, OCR unavailable/timeout, Vision timeout/crash, Embedding failure, partial pipeline failure, DB/runtime failure, concurrency, restart recovery and request isolation. Existing provider-contract tests supply provider-specific injected failures; Phase 10 adds restart recovery and final acceptance-report coverage.

## Hardening

A process crash can leave canonical-image temp scopes and rows in `pending`/`in_progress`.

Before FastAPI serves requests:

1. `LocalImageStorage.cleanup_all_scopes_async()` removes stale files under the repository-owned temp root.
2. `ProfileAnalysisService.recover_interrupted_async()` delegates lifecycle recovery to the repository.
3. `ProfileAnalysisRepository.fail_interrupted_async()` atomically changes only unfinished rows with no result payload to `failed`.

This contract is intentionally single-worker. Do not use `--workers > 1`; multi-worker startup would make process-lifetime recovery ownership ambiguous.

Existing privacy boundaries remain: raw upload bytes are not persisted; canonical files are temporary; persisted OCR block content is redacted; provider/raw image/OCR payloads are not logged; HTTP failures remain sanitized; and input byte/MIME/pixel/corruption limits remain enforced.

## Local deployment configuration

The verified topology is a local FastAPI process, PostgreSQL, Tesseract OCR, local llama.cpp Vision runtime, plus the separately benchmarked Transformers embedding runtime. Core/Application do not depend on GGUF, Safetensors, PyTorch or ONNX details.

No new application Docker/Compose topology is introduced because the current target is a Windows workstation with an external loopback llama.cpp runtime; containerizing the API would change that verified topology without evidence. Docker is used only for reproducible PostgreSQL verification.

## Operational runbook

Run from repository root in PowerShell.

### First-time setup

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -r requirements.txt
```

Set `PROFILE_INSIGHT_DATABASE_URL` to an async PostgreSQL URL in the shell/environment. Do not commit credentials.

### Database migration

```powershell
python -m alembic -c src/infrastructure/alembic.ini upgrade head
python -m alembic -c src/infrastructure/alembic.ini check
```

### AI runtime startup

Verify Tesseract:

```powershell
tesseract --version
tesseract --list-langs
```

After the candidate model is already acquired locally, start the current candidate runtime offline. This command is candidate evidence, not a production-final selection:

```powershell
llama-server -hf "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M" --offline --host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1 --log-disable
```

In another shell verify llama.cpp readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

### Application startup and smoke check

```powershell
python -m uvicorn src.host.app:app --host 127.0.0.1 --port 5000 --workers 1
```

In another shell:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/openapi.json | Select-Object -ExpandProperty info
```

### Hardware/runtime evidence

```powershell
python scripts/collect_phase10_hardware.py --output phase10_hardware_profile.json
```

If `llama-server` is not on PATH, use `--llama-server-executable` with its local executable path.

### End-to-end benchmark

Set `DATASET_ROOT` to the authorized Phase 2 dataset directory, then run:

```powershell
python -m evaluation.phase10_end_to_end_benchmark `
  --manifest "$env:DATASET_ROOT\manifest.json" `
  --dataset-root "$env:DATASET_ROOT" `
  --iterations 2 `
  --concurrency 2 `
  --expected-manifest-fingerprint "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc" `
  --expected-dataset-content-fingerprint "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e" `
  --output phase10_end_to_end_benchmark.json
```

Slow/time-out behavior from the current Vision candidate is valid evidence and must be returned rather than hidden.

### Final evidence report

After the prior phase benchmark JSON files are available locally:

```powershell
python -m evaluation.phase10_benchmark `
  --ocr-report .\phase4_ocr_benchmark.json `
  --vision-report .\phase5_vision_benchmark.json `
  --embedding-report .\phase6_embedding_benchmark.json `
  --insight-report .\phase8_insight_benchmark.json `
  --api-persistence-report .\phase9_api_persistence_benchmark.json `
  --e2e-report .\phase10_end_to_end_benchmark.json `
  --hardware-report .\phase10_hardware_profile.json `
  --output .\phase10_final_benchmark.json
```

If local report filenames differ, pass the actual generated paths; do not rename or fabricate report contents.

### Fail-fast implementation verification

```powershell
python scripts/verify_phase10.py
if ($LASTEXITCODE -ne 0) { throw "Phase 10 implementation verification failed" }
```

This checks compile/import, focused hardening/benchmark/architecture tests, disposable PostgreSQL migration/recovery, hardware collection, final-report schema behavior and full regression. It intentionally asserts that acceptance is not claimed without real benchmark evidence.

### Shutdown

Stop Uvicorn and llama.cpp normally with `Ctrl+C`; stop PostgreSQL according to the deployment method used. Normal analysis completion/failure cleans its temp scope.

### Restart/recovery

1. Ensure the prior Uvicorn process is stopped.
2. Ensure PostgreSQL is reachable and llama.cpp is healthy.
3. Start Uvicorn with exactly one worker.
4. Startup cleanup removes stale temp scopes and changes leftover `pending`/`in_progress` rows to `failed`.
5. Repeat the OpenAPI smoke check.

### Troubleshooting

- Missing Tesseract/traineddata: fix the real OCR runtime; do not bypass it with a fake provider.
- Vision timeout/crash: retain it as benchmark evidence; do not merely increase timeouts and claim a quality fix.
- Embedding offline load failure: pre-acquire the exact model/revision, then rerun offline.
- DB failure: verify the environment URL, PostgreSQL readiness and Alembic head.
- Recovery-created `failed` rows: they were unfinished in the previous single-worker process lifetime and are intentionally not resumed.

## Production model/config selection status

`UNRESOLVED`

No OCR traineddata choice, Vision model/config, Embedding model/config, semantic threshold or Insight threshold is production-final in this implementation commit. Existing appsettings remain candidate configuration.

## Acceptance evidence still required

Target hardware/runtime profile, current OCR/Vision/Embedding comparison reports, target-host E2E workload/resource/failure evidence, offline startup/restart evidence, and full regression after any final selected config change are still required. Metrics requiring unavailable insight-level labels must remain `unavailable`; they cannot be fabricated to close the phase.

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

Phase 10 is not `COMPLETE / PROJECT ACCEPTED`. Final selection and closure require the real target-hardware evidence to be returned and reviewed.
