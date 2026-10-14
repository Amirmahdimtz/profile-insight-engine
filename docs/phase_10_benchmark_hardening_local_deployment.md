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
| Target hardware | `AVAILABLE` | Target-Windows hardware/runtime profile was returned during verified Phase 10 local implementation verification. |

Authorized dataset identity carried forward from prior verified phases:

- dataset: `profile_insight_real_eval` v`1.0.0`
- manifest fingerprint: `06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc`
- content fingerprint: `5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e`

## Candidate evidence carried forward

OCR Phase 4 did not establish a universal winner: `tessdata_best` was slightly better on overall/Persian CER/WER, while `tessdata_fast` had materially lower p95 latency and slightly better mixed-script WER.

Vision Phase 5 is negative selection evidence for the current candidates. Gemma 3 4B failed the offline-load preflight. Qwen2.5-VL 3B completed a reduced evaluation but had 13/18 request timeouts, overall F1 `0.0`, p50 around `276993 ms`, and p95 around `298460 ms`. The current Vision candidate has previously shown very high latency/timeouts and is not production-final.

Phase 10 target-host diagnostics additionally rejected the llama.cpp path on the CPU-only workstation: Qwen2-VL 2B, SmolVLM2 2.2B and SmolVLM2 256M all exceeded the 120-second request budget on a clean single-image probe. Direct Transformers inference with `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` identified image splitting as the dominant bottleneck: the default processor produced 17 image tiles and a first-token latency around 102 seconds, while disabling image splitting and using one 512px tile reduced first-token latency to about 13.8 seconds and completed a 32-token generation in about 14.3 seconds. Prompt-only JSON remained fail-closed because a 256-token response ended with an unclosed JSON object. `lm-format-enforcer==0.11.3` was rejected because its Transformers integration is incompatible with the repository's pinned `transformers==5.18.0`.

The final target-host Transformers candidate benchmark has now been returned and reviewed. For model snapshot `067788b187b95ebe7b2e040b3e4299e342e5b8fd` with Transformers `5.18.0`, Outlines `1.3.3`, CPU execution, one 512px image tile and `max_tokens=256`:

- 18 samples were attempted and only 9 completed successfully;
- failure rate was `0.5`;
- 5 failures were malformed structured output;
- 4 failures were structured output truncated at the token limit;
- measured p50 latency was about `45531 ms` and p95 about `56951 ms` on successful samples;
- deterministic reruns were stable on the configured 3-sample subset;
- overall Precision/Recall/F1 and each evaluated visual label type were `0.0`;
- unsupported-claim rate before policy and after validation was `0.0`.

This is valid benchmark evidence, not an acceptable production selection. In this benchmark family, `benchmark_valid=true` means the benchmark completed with enough measurable output to produce metrics; it does **not** mean the candidate is production-ready. The reviewed SmolVLM2-256M Transformers candidate is therefore rejected for the current target hardware/configuration.

The project-finalization path does not introduce another generative VLM. It reuses the already benchmarked local `google/siglip2-base-patch16-224` vision-language encoder as a zero-shot visual classifier. The first target-host run demonstrated strong runtime behavior: 18/18 samples completed, failure rate was `0.0`, deterministic reruns passed, p50 latency was about `1210 ms`, and the closed-set report measured precision about `0.909`, recall about `0.476`, and F1 about `0.625` for resolved snapshot `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.

That first quality score is diagnostic-only because the candidate vocabulary was assembled from the complete versioned manifest, including EVAL annotations. The report did not expose the correct per-image assignment, but its class universe still contained the answer vocabulary from the holdout set and therefore is not sufficient evidence for production selection. The selection-safe protocol now builds the candidate taxonomy from TRAIN labels only and evaluates all EVAL labels unchanged. EVAL labels absent from the TRAIN taxonomy remain in the recall denominator. This prevents benchmark-answer vocabulary leakage while retaining the valid target-host runtime evidence. The provider emits no synthetic caption, uses independent SigLIP sigmoid scores as explicitly uncalibrated confidence, and remains benchmark-only until holdout-safe evidence is reviewed.

Embedding Phase 6 measured strong retrieval for `google/siglip2-base-patch16-224` (Recall@5 about `0.9907`, mAP about `0.9183`) but theme Macro F1 at threshold `0.25` was `0.0`. Those retrieval results justify evaluating the same already-local model as a zero-shot Vision classifier; they do not pre-approve its Vision quality or threshold.

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

The current production-wired topology is a local FastAPI process, PostgreSQL, Tesseract OCR, local llama.cpp Vision runtime, plus the separately benchmarked Transformers embedding runtime. Phase 10 also contains an explicit, non-DI-bound Transformers Vision candidate used only by the candidate benchmark until evidence review selects a provider. Core/Application do not depend on GGUF, Safetensors, PyTorch, Outlines or ONNX details.

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

### Transformers Vision candidate benchmark

The Phase 10 candidate uses `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` through the pinned Transformers runtime, a single 512px image tile, and Outlines JSON-schema constrained decoding. It is intentionally not registered as the active `IVisionProvider` yet; the existing `vision.provider` setting remains unchanged until this benchmark evidence is reviewed.

If the model is not already cached, acquire it once while network access is available:

```powershell
python -c "import torch; from transformers import AutoProcessor, AutoModelForMultimodalLM; m='HuggingFaceTB/SmolVLM2-256M-Video-Instruct'; AutoProcessor.from_pretrained(m, revision='main'); AutoModelForMultimodalLM.from_pretrained(m, revision='main', dtype=torch.float32); print('Transformers Vision candidate cache ready')"
```

Then run the real 18-sample evaluation offline from repository root:

```powershell
$DatasetRoot = "E:\phase2-real-dataset"

python -m evaluation.transformers_vision_benchmark `
  --manifest "$DatasetRoot\manifest.json" `
  --dataset-root "$DatasetRoot" `
  --iterations 2 `
  --determinism-sample-count 3 `
  --expected-manifest-fingerprint "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc" `
  --expected-dataset-content-fingerprint "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e" `
  --output .\phase10_transformers_vision_benchmark.json

if ($LASTEXITCODE -ne 0) { throw "Transformers Vision benchmark failed" }
```

The report records the resolved Hugging Face snapshot, provider/runtime versions, per-type precision/recall/F1, unsupported-claim rates, p50/p95 latency, failure rate and deterministic reruns. Raw generated model text and raw image bytes are not written to the benchmark report.

### SigLIP2 zero-shot Vision finalization benchmark

This is the final local/offline Vision candidate under the current CPU-only constraint. It uses the existing `google/siglip2-base-patch16-224` model through Transformers zero-shot image classification semantics. The versioned manifest supplies the visual class universe from the TRAIN split only; EVAL labels are never used to construct the candidate taxonomy and sample-specific ground-truth mappings are never passed to the provider.

The repository candidate configuration is CPU-only, `local_files_only=true`, `minimum_score=0.10`, at most one observation per visual kind, and the standard zero-shot hypothesis template `This is a photo of {}.`. The threshold is a candidate setting recorded in the report, not a production-final value.

If the SigLIP2 checkpoint is not yet cached, acquire it once while network access is available:

```powershell
python -c "from transformers import AutoModel, AutoProcessor; m='google/siglip2-base-patch16-224'; AutoProcessor.from_pretrained(m, revision='main'); AutoModel.from_pretrained(m, revision='main'); print('SigLIP2 zero-shot cache ready')"
```

Then disconnect/network-disable if required and run the real target-host benchmark:

```powershell
$DatasetRoot = "E:\phase2-real-dataset"

python -m evaluation.siglip_zero_shot_vision_benchmark `
  --manifest "$DatasetRoot\manifest.json" `
  --dataset-root "$DatasetRoot" `
  --iterations 2 `
  --determinism-sample-count 3 `
  --expected-manifest-fingerprint "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc" `
  --expected-dataset-content-fingerprint "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e" `
  --output .\phase10_siglip_zero_shot_vision_benchmark.json

if ($LASTEXITCODE -ne 0) { throw "SigLIP2 zero-shot Vision benchmark failed" }
```

The candidate benchmark fails before model loading if the TRAIN split contains no visual taxonomy labels, and fails closed if any evaluation image has a runtime failure. A zero exit code means the complete holdout-safe benchmark produced reliable metrics; it does not automatically promote the candidate. The report includes the resolved model snapshot, TRAIN-derived taxonomy and taxonomy fingerprint, EVAL taxonomy-coverage diagnostics, configured threshold, Precision/Recall/F1, unsupported-claim rates, score diagnostics, p50/p95 latency, failure rate and deterministic reruns. Raw image bytes are never written to the report.

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

Slow/time-out behavior from the current Vision candidate is valid evidence and must be returned rather than hidden. The Vision benchmark also waits for llama.cpp slots to become idle after a failed/timed-out request before advancing to the next sample, so cancellation cleanup and queue time cannot contaminate later sample measurements.

### Final evidence report

After the prior phase benchmark JSON files are available locally:

```powershell
python -m evaluation.phase10_benchmark `
  --ocr-report .\phase4_ocr_benchmark.json `
  --vision-report .\phase10_siglip_zero_shot_vision_benchmark.json `
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
- `training split must contain at least one visual taxonomy label`: stop the Vision benchmark. This means the authorized dataset is valid for Phase 2 but insufficient for leakage-free zero-shot model selection. Add authorized TRAIN visual annotations or provide an independent versioned product taxonomy; do not copy EVAL labels into the candidate vocabulary.

## Production model/config selection status

`UNRESOLVED`

The target-host generative Vision evidence has been reviewed. The current Transformers SmolVLM2-256M candidate is explicitly rejected because half of the evaluation samples failed structured inference and measured exact-match F1 was `0.0`. Earlier llama.cpp candidates were already rejected by timeout/load evidence. No generative local Vision model/provider configuration is acceptable on the current CPU-only workstation.

The finalization candidate is the non-generative SigLIP2 zero-shot adapter described above. Its first closed-set target-host run established valid runtime evidence, but the holdout-safe rerun exposed the remaining evaluation gap: the current authorized dataset has no TRAIN visual taxonomy labels. The candidate deliberately remains unbound from DI and the active appsettings provider is not changed to any rejected or unverified candidate. No OCR traineddata choice, Embedding model/config, semantic threshold or Insight threshold is production-final yet.

Production Vision selection now requires an independent candidate vocabulary before another quality run is meaningful. The acceptable sources are authorized visual annotations on the TRAIN split or an independent versioned product taxonomy defined outside EVAL ground truth. EVAL labels must not be copied into that taxonomy. Until one of those evidence sources exists, the runtime viability of SigLIP2 remains valid but its production quality cannot be selected without leakage.

## Acceptance evidence still required

The target hardware profile, rejected generative Vision benchmark, SigLIP2 runtime evidence and full regression evidence are available. The holdout-safe quality benchmark is blocked before model loading because the current dataset contains no TRAIN visual taxonomy labels. Remaining acceptance work is therefore: supply authorized TRAIN visual labels or an independent versioned product taxonomy, rerun the holdout-safe SigLIP2 benchmark once, freeze the selected Vision snapshot/taxonomy/config only if supported by that evidence, then run the required OCR/Embedding/Insight/API-persistence/E2E evidence aggregation, offline startup/restart checks and final full regression. Metrics requiring unavailable annotations remain `unavailable`; they cannot be fabricated to close the phase.

## Verified local implementation evidence

Target-Windows implementation verification was completed against commit `9e8d16192b913eb90fc8b39b392b518b26081034`.

- targeted Phase 10 documentation/local-verifier/SigLIP holdout-safe suite: 13 tests passed;
- focused Phase 10 suite: 57 tests passed;
- PostgreSQL recovery regression: 5 tests passed;
- full repository regression: 354 tests passed with no failures or errors;
- Alembic upgrade and metadata drift check: passed;
- disposable PostgreSQL startup/readiness: passed with runtime password authentication for the host connection;
- hardware profile: Windows 11, CPython 3.12.10, 8 logical CPU threads, approximately 16 GB RAM, no NVIDIA GPU;
- implementation verifier completed with `PHASE 10 IMPLEMENTATION VERIFICATION PASSED`;
- tracked repository status was clean at verification time;
- local `HEAD` matched `9e8d16192b913eb90fc8b39b392b518b26081034`.

This closes implementation/local-verification work. It does not close Phase 10 acceptance. The final acceptance scaffold still correctly reports missing OCR, Vision, Embedding, Insight, API/persistence and E2E evidence inputs, and production selections remain unresolved.

## Status

`IMPLEMENTED / LOCAL VERIFICATION PASSED / ACCEPTANCE BLOCKED`

Phase 10 is not `COMPLETE / PROJECT ACCEPTED`. The remaining blocker is acceptance evidence, led by the missing independent visual taxonomy/TRAIN annotations required for leakage-free Vision quality selection, followed by final provider/config selection and E2E evidence aggregation.
