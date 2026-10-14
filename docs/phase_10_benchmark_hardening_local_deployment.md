# Phase 10 — Benchmark, Hardening & Local Deployment

## Starting baseline

Phase 10 starts from `main` commit `594187e15b692944e126c485b7d9c1b4383cce1d`, the verified Phase 9 closure commit. Phase 9 is `COMPLETE / READY_FOR_NEXT_PHASE`. Its benchmark scope is `phase9_contract_and_persistence_not_end_to_end_ml`, so those API/DB measurements are not treated as Phase 10 end-to-end ML evidence.

## Scope and architecture

The production dependency direction remains:

`Host → Application → Core → Infrastructure`

The current HTTP workflow remains:

`multipart input → ImageProcessingService → OCR → Vision disabled by selected profile → Cross-Image Aggregation → InsightGenerationService → privacy sanitization → PostgreSQL`

`SemanticThemeService` is intentionally not inserted into the HTTP workflow. Phase 6 requires caller-supplied observable theme labels and the production API still has no benchmarked source for them. Inventing one in Phase 10 would change the product contract.

## Gate 0

| Input | Status | Evidence / gap |
| --- | --- | --- |
| Full pipeline | `AVAILABLE` | Phase 9 exposes the verified ingestion/OCR/Vision/aggregation/insight/persistence path. |
| Evaluation dataset | `PARTIAL` | Authorized images are outside the repository. Dataset identity/fingerprints are known, but insight-level correctness annotations are unavailable. |
| Candidate model configs | `SELECTED` | Degraded production profile is frozen: Tesseract `tessdata_best` for OCR, Vision disabled by rejected holdout evidence, Embedding outside the active HTTP workflow. |
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

`evaluation.phase10_end_to_end_benchmark` executes the real selected production service path using the authorized dataset, configured Tesseract OCR and PostgreSQL. With `vision.enabled=false`, the workflow does not call `IVisionProvider`; it uses no fake provider and emits no fabricated Vision evidence.

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

The selected deployment topology is a local FastAPI process, PostgreSQL and Tesseract OCR with Vision explicitly disabled by configuration. The rejected llama.cpp and Transformers Vision adapters remain in Infrastructure for benchmark/history and future reevaluation but are not required to run the selected production profile. The separately benchmarked Embedding runtime is not in the active HTTP workflow. Core/Application do not depend on GGUF, Safetensors, PyTorch, Outlines or ONNX details.

No new application Docker/Compose topology is introduced because the selected target remains a local Windows workstation running the API directly. The degraded production profile has no llama.cpp runtime dependency. Docker is used only for reproducible PostgreSQL verification.

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

The selected production profile does not start llama.cpp because `vision.enabled=false`. Verify Tesseract and point the service at the selected `tessdata_best` directory:

```powershell
$env:PROFILE_INSIGHT_TESSDATA_DIR = "E:\profile-insight-models\tessdata_best"
tesseract --version
tesseract --tessdata-dir "$env:PROFILE_INSIGHT_TESSDATA_DIR" --list-langs
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

The hardware collector may report llama.cpp details when a local executable is available, but llama.cpp is not a prerequisite for the selected production profile.

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

### Deterministic Phase 10 holdout preparation

The original authorized dataset contains 18 EVAL samples and no TRAIN split. To avoid inventing a product taxonomy or copying all EVAL answers into the candidate vocabulary, Phase 10 now derives a new versioned holdout manifest from the same authorized images. TRAIN membership is selected without reading labels: samples are ranked by SHA-256 over `dataset_id + sample_id`, the first 6 become TRAIN, and the remaining 12 stay EVAL. The source manifest is never modified.

This is a precommitted evaluation protocol, not model tuning. The candidate model, threshold and hypothesis template must remain unchanged after the split is created.

```powershell
$DatasetRoot = "E:\phase2-real-dataset"
$HoldoutManifest = "$DatasetRoot\manifest.phase10-holdout.json"
$HoldoutReport = ".\phase10_holdout_dataset.json"

python -m evaluation.prepare_phase10_holdout `
  --manifest "$DatasetRoot\manifest.json" `
  --dataset-root "$DatasetRoot" `
  --dataset-version "1.1.0" `
  --train-count 6 `
  --expected-source-manifest-fingerprint "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc" `
  --expected-dataset-content-fingerprint "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e" `
  --output-manifest $HoldoutManifest `
  --output-report $HoldoutReport

if ($LASTEXITCODE -ne 0) { throw "Phase 10 holdout preparation failed" }

$Holdout = Get-Content $HoldoutReport | ConvertFrom-Json
$Holdout | ConvertTo-Json -Depth 8
```

Expected invariants: `train_sample_count=6`, `eval_sample_count=12`, a non-zero `train_visual_taxonomy_total`, unchanged dataset-content fingerprint, and a new derived manifest fingerprint. The generated report contains only sample IDs and annotation-derived taxonomy metadata; it contains no raw image bytes.

### SigLIP2 zero-shot Vision finalization benchmark

This is the final local/offline Vision candidate under the current CPU-only constraint. It uses the existing `google/siglip2-base-patch16-224` model through Transformers zero-shot image classification semantics. The versioned manifest supplies the visual class universe from the TRAIN split only; EVAL labels are never used to construct the candidate taxonomy and sample-specific ground-truth mappings are never passed to the provider.

The repository candidate configuration is CPU-only, `local_files_only=true`, `minimum_score=0.10`, at most one observation per visual kind, and the standard zero-shot hypothesis template `This is a photo of {}.`. The threshold is a candidate setting recorded in the report, not a production-final value.

If the SigLIP2 checkpoint is not yet cached, acquire it once while network access is available:

```powershell
python -c "from transformers import AutoModel, AutoProcessor; m='google/siglip2-base-patch16-224'; AutoProcessor.from_pretrained(m, revision='main'); AutoModel.from_pretrained(m, revision='main'); print('SigLIP2 zero-shot cache ready')"
```

After the deterministic holdout manifest has been prepared, disconnect/network-disable if required and run the real target-host benchmark without changing the candidate configuration:

```powershell
$DatasetRoot = "E:\phase2-real-dataset"
$HoldoutManifest = "$DatasetRoot\manifest.phase10-holdout.json"
$Holdout = Get-Content .\phase10_holdout_dataset.json | ConvertFrom-Json

python -m evaluation.siglip_zero_shot_vision_benchmark `
  --manifest $HoldoutManifest `
  --dataset-root "$DatasetRoot" `
  --iterations 2 `
  --determinism-sample-count 3 `
  --expected-manifest-fingerprint $Holdout.derived_manifest_fingerprint `
  --expected-dataset-content-fingerprint $Holdout.dataset_content_fingerprint `
  --output .\phase10_siglip_zero_shot_vision_benchmark.json

$VisionExitCode = $LASTEXITCODE
if (Test-Path .\phase10_siglip_zero_shot_vision_benchmark.json) {
  Get-Content .\phase10_siglip_zero_shot_vision_benchmark.json
}
Write-Host "Vision benchmark exit code: $VisionExitCode"
```

Do not tune `minimum_score`, the hypothesis template, TRAIN count or taxonomy after seeing this holdout result. Return the report even if the candidate quality is insufficient.

The candidate benchmark fails before model loading if the TRAIN split contains no visual taxonomy labels, and fails closed if any evaluation image has a runtime failure. A zero exit code means the complete holdout-safe benchmark produced reliable metrics; it does not automatically promote the candidate. The report includes the resolved model snapshot, TRAIN-derived taxonomy and taxonomy fingerprint, EVAL taxonomy-coverage diagnostics, configured threshold, Precision/Recall/F1, unsupported-claim rates, score diagnostics, p50/p95 latency, failure rate and deterministic reruns. Raw image bytes are never written to the report.

### End-to-end benchmark

Set `DATASET_ROOT` to the authorized Phase 2 dataset directory. Ensure `PROFILE_INSIGHT_TESSDATA_DIR` is set and do not start llama.cpp; the E2E runner reads the selected `vision.enabled=false` profile from appsettings. Then run:

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

Rejected Vision timeout/failure behavior remains preserved in the historical candidate benchmark reports. The selected E2E production run does not invoke Vision because `vision.enabled=false`; its latency and failure metrics therefore measure the exact degraded production profile being handed off.

### Final evidence report

After the prior phase benchmark JSON files are available locally:

```powershell
python -m evaluation.phase10_benchmark `
  --deployment-profile degraded_ocr_only `
  --ocr-report .\phase4_ocr_benchmark.json `
  --vision-report .\phase10_siglip_zero_shot_vision_benchmark.json `
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

Stop Uvicorn normally with `Ctrl+C` and stop PostgreSQL according to the deployment method used. The selected degraded profile has no llama.cpp production process. Normal analysis completion/failure cleans its temp scope.

### Restart/recovery

1. Ensure the prior Uvicorn process is stopped.
2. Ensure PostgreSQL is reachable and `PROFILE_INSIGHT_TESSDATA_DIR` points to the selected `tessdata_best` directory.
3. Start Uvicorn with exactly one worker.
4. Startup cleanup removes stale temp scopes and changes leftover `pending`/`in_progress` rows to `failed`.
5. Repeat the OpenAPI smoke check.

### Troubleshooting

- Missing Tesseract/traineddata: fix the real OCR runtime; do not bypass it with a fake provider.
- Vision timeout/crash: retain it as benchmark evidence; do not merely increase timeouts and claim a quality fix.
- Embedding offline load failure: pre-acquire the exact model/revision, then rerun offline.
- DB failure: verify the environment URL, PostgreSQL readiness and Alembic head.
- Recovery-created `failed` rows: they were unfinished in the previous single-worker process lifetime and are intentionally not resumed.
- `training split must contain at least one visual taxonomy label`: run the deterministic Phase 10 holdout preparation first. If the derived TRAIN split still contains no visual labels, stop and return the holdout report; do not copy EVAL labels manually or alter the split after seeing results.

## Production model/config selection status

`SELECTED_DEGRADED_PROFILE`

The target-host generative Vision evidence has been reviewed. The current Transformers SmolVLM2-256M candidate is explicitly rejected because half of the evaluation samples failed structured inference and measured exact-match F1 was `0.0`. Earlier llama.cpp candidates were already rejected by timeout/load evidence. No generative local Vision model/provider configuration is acceptable on the current CPU-only workstation.

The finalization SigLIP2 zero-shot adapter is now rejected by the returned holdout-safe evidence and remains unbound from DI. The selected production profile disables Vision extraction with `vision.enabled=false`. OCR is production-selected as Tesseract with official `tessdata_best` for `fas+eng` because it measured better overall and Persian CER/WER on the authorized dataset. Its higher p95 latency versus `tessdata_fast` is accepted for this quality-first Persian deployment profile. Embedding is not in the active HTTP workflow, so its model/semantic threshold are not deployment blockers. Insight policy remains the verified Phase 8 versioned configuration; unavailable product-quality calibration metrics remain explicitly unavailable.

The deterministic derived-holdout benchmark has now been returned. The derived dataset is v`1.1.0` with manifest fingerprint `d4417e50f2436c71ce9e9201646a5e060684377f52fdbabad42df4f487316833`, 6 TRAIN samples and 12 EVAL samples. The TRAIN taxonomy contained 7 labels, but its EVAL coverage was `0.0` for scene, object, activity and topic. SigLIP2 snapshot `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` completed all 12 EVAL samples with failure rate `0.0` and deterministic reruns, but measured Precision/Recall/F1 were all `0.0`. The frozen zero-shot candidate is therefore rejected for production quality; its exit code `0` only means the benchmark itself completed successfully.

To finish the deployable project without promoting failed ML evidence, the production deployment profile explicitly disables Vision extraction with `vision.enabled=false`. `ProfileAnalysisService` skips the Vision call and continues with OCR-backed evidence, aggregation and Insights. This produces no fake Vision evidence and removes llama.cpp from the required production runtime. Visual scene/object/activity/topic capability is a documented deferred capability, not silently claimed as available.

## Final acceptance evidence

Target-Windows final verification was completed against commit `01fcf80ade2ffbc701b066281e262d25b1689b11`.

- focused Phase 10 suite: 63 tests passed;
- PostgreSQL recovery regression: 5 tests passed;
- full repository regression: 362 tests passed with no failures or errors;
- Alembic upgrade and metadata drift check: passed;
- tracked repository state was clean;
- implementation verifier completed with `PHASE 10 IMPLEMENTATION VERIFICATION PASSED`;
- final evidence aggregation used deployment profile `degraded_ocr_only`;
- `acceptance_ready=true`;
- `missing_evidence=[]`;
- OCR selection evidence valid: `true`;
- Vision rejection evidence valid: `true`;
- degraded-profile E2E evidence valid: `true`;
- measured E2E failure rate: `0.0`;
- measured unsupported-claim rate after validation: `0.0`;
- structural summary factuality: `true`;
- measured p50 total latency: approximately `26009 ms`;
- measured p95 total latency: approximately `89823 ms`;
- measured throughput: approximately `0.495 images/second`;
- measured peak service-process RAM: approximately `80.27 MB`.

Authorized end-to-end accuracy, insight precision and confidence calibration remain `unavailable` because the supplied dataset does not contain the annotations required to measure those values. They are not estimated or fabricated and are not required by the selected degraded acceptance profile.

The final production selections are frozen as:

- OCR: `tesseract_tessdata_best_fas_eng`;
- Vision: `disabled_rejected_by_holdout_evidence`;
- Embedding: `not_in_active_http_workflow`;
- Insight thresholds: `phase8_versioned_policy_no_posthoc_tuning`.

Generated benchmark JSON/log files are local evidence artifacts and may remain untracked. Production source, configuration, tests and documentation must remain clean and version-controlled.

## Backend/DevOps handoff

The accepted deployment scope is the degraded OCR-backed profile. Do not advertise scene/object/activity/topic Vision capability for this release; `vision.enabled=false` is intentional and evidence-backed.

Required runtime inputs:

- `PROFILE_INSIGHT_DATABASE_URL`: PostgreSQL SQLAlchemy async URL using `postgresql+asyncpg://`;
- `PROFILE_INSIGHT_TESSDATA_DIR`: directory containing the selected official `tessdata_best` traineddata used by Tesseract for the configured `fas+eng` OCR runtime;
- Tesseract 5 available on `PATH`;
- PostgreSQL reachable before application startup.

Deployment sequence from repository root:

```powershell
python -m pip install -r requirements.txt
python -m alembic -c src/infrastructure/alembic.ini upgrade head
python -m alembic -c src/infrastructure/alembic.ini check
tesseract --version
tesseract --tessdata-dir "$env:PROFILE_INSIGHT_TESSDATA_DIR" --list-langs
python -m uvicorn src.host.app:app --host 127.0.0.1 --port 5000 --workers 1
```

Use exactly one Uvicorn worker because startup recovery ownership is defined for the verified single-worker topology. No llama.cpp, GPU runtime or Embedding service is required for the accepted release. Keep database credentials and machine-specific paths out of repository configuration.

After startup, verify `/openapi.json` and exercise the POST/status/result profile-analysis flow. Database migrations must be applied before serving traffic. Restart behavior intentionally marks interrupted `pending`/`in_progress` analyses as `failed` and removes stale temporary image scopes.

## Status

`COMPLETE / PROJECT ACCEPTED / READY_FOR_BACKEND_DEVOPS_HANDOFF`

Phase 10 is complete for the accepted `degraded_ocr_only` production scope. The repository is ready to be handed to Backend/DevOps for integration and deployment. Vision remains a documented deferred capability and must not be enabled without a new benchmark/evidence-review cycle.
