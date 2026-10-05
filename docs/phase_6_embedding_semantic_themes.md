# Phase 6 — Embedding & Semantic Themes

## Scope

Phase 6 adds only the independent multimodal semantic signal required by the phase plan:

`CanonicalImage + candidate observable labels → image/text embeddings → similarity → theme matches / retrieval / near-duplicate groups`

It does **not** add Phase 7 cross-image evidence aggregation, Phase 8 insight generation, Phase 9 API/persistence/lifecycle, or any production-final model/threshold decision.

## Contracts and architecture

`IEmbeddingProvider` is a real replaceable provider boundary because the embedding model/runtime is benchmarkable and replaceable. The concrete Infrastructure implementation is `TransformersEmbeddingProvider`. `SemanticThemeService` is a Core workflow service and depends only on `IEmbeddingProvider`, provider-neutral embedding contracts, and `ConfigReader`.

DI follows the existing convention: provider and service use `@inject`, dependencies are constructor-injected, and no feature-specific manual registration is added.

Application/Core do not know whether weights are Safetensors, PyTorch, ONNX, or another runtime format. Model loading, tensor handling, device selection and processor behavior remain in Infrastructure.

## Embedding contract

Every embedding vector:

- has a stable input `item_id`;
- contains only finite numeric values;
- rejects NaN/Inf;
- rejects zero norm;
- is L2-normalized;
- exposes its vector dimension;
- participates in a batch with one consistent dimension;
- preserves provider/model/config traceability.

Image and text batches must preserve caller order and must come from exactly the same provider, provider version, model identity/version and config version before similarity is calculated.

The concrete provider records the resolved Hugging Face commit hash when Transformers exposes it, so a `main` candidate revision is still traceable in benchmark output. A production-final model revision is not frozen in Phase 6.

## Semantic theme behavior

`SemanticThemeService.analyze_async()` validates candidate theme labels against the existing Phase 1 observable-claim policy before embedding them. Person-level sensitive inference remains rejected; semantic similarity cannot bypass the safety contract.

Cosine similarity is calculated only from normalized vectors. Theme matches use the configurable `embedding.theme_similarity_threshold`. Retrieval is deterministic: score descending, then `image_id` ascending as the tie-break.

Near-duplicate grouping uses image-image cosine similarity and the configurable `embedding.near_duplicate_similarity_threshold`. Pairwise matches are converted into deterministic connected components. Group IDs are deterministic hashes of the ordered image IDs. This is only near-duplicate detection; no Phase 7 evidence aggregation or recurrence scoring is performed.

## Candidate runtime/model

The default configuration is a **benchmark candidate**, not a production selection:

- provider: Hugging Face Transformers
- candidate model: `google/siglip2-base-patch16-224`
- execution device: CPU by default for portability

SigLIP2 is used as an initial candidate because it is a multilingual vision-language encoder with image/text feature support. Final model, device policy and thresholds remain benchmark decisions.

Python runtime dependencies are pinned in `requirements.txt`. Model weights are downloaded to the normal Hugging Face cache and are not committed to the repository.

## Benchmark

`evaluation.embedding_benchmark` uses the existing Phase 2 evaluation manifest and Phase 3 canonicalization path. It derives candidate observable theme labels from authorized eval ground-truth labels; it does not create model-derived ground truth.

For one model candidate it reports:

- dataset ID/version and manifest/content fingerprints;
- provider/version and resolved model version;
- embedding dimension;
- theme Macro F1 at the configured candidate threshold;
- mean Recall@K;
- mAP;
- Persian-heavy, English-heavy and mixed Persian/English theme Macro F1 only when that slice has authorized ground-truth labels; otherwise the value is `null` rather than a fabricated perfect score;
- one explicit warm-up/model-load latency measurement;
- p50/p95 full-eval batch latency measured only after warm-up, so download/model-load time is not mixed into inference latency;
- similarity diagnostics (match count and score ranges) for threshold interpretation without silently retuning the threshold;
- process peak RAM where the OS exposes it;
- process VRAM when `nvidia-smi` reports memory for the current process;
- deterministic rerun equality when at least two iterations are used;
- candidate threshold values;
- detected near-duplicate group count.

Run the same benchmark command for multiple model candidates and compare reports. Phase 6 does not automatically promote a winner.

The existing Phase 2 manifest does not contain explicit human-labelled near-duplicate pairs. Therefore near-duplicate logic is covered deterministically by unit tests, but model-quality evaluation for near-duplicate precision/recall remains dependent on an authorized pair annotation set. No pair labels are fabricated from model outputs.

## Windows local verification prerequisites

Run all commands from the repository root in PowerShell.

Phase 5 is already verified on `main`. Phase 6 is implemented directly on the latest `main`; local verification must therefore synchronize `main` before interpreting any result.

```powershell
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "Unable to fast-forward local main. Review tracked local changes before continuing." }
git rev-parse HEAD
git status --short
```

Create and activate the project virtual environment if needed, then install pinned dependencies:

```powershell
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }
```

Set the authorized Phase 2 dataset and report locations. Do not place private datasets or model caches inside the Git repository.

```powershell
$env:DATASET_ROOT = "E:\phase2-real-dataset"
$env:HF_HOME = "E:\profile-insight-models\huggingface"
$env:PHASE6_REPORT_ROOT = Join-Path (Get-Location) "phase6_reports"
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:PHASE6_REPORT_ROOT | Out-Null
```

The first candidate run may require network access to populate the Hugging Face cache. After a candidate is cached, set `embedding.local_files_only: true` in a local uncommitted config copy if strict offline verification is required. Do not commit machine-specific cache paths or secrets.

The current Phase 2 manifest does not encode authorized translation-equivalence pairs or human-labelled near-duplicate pairs. The benchmark therefore reports Persian-heavy, English-heavy and mixed-slice performance only where those slices contain authorized labels, while true paired cross-language equivalence quality and near-duplicate precision/recall remain dataset limitations until authorized annotations exist. Model outputs are never promoted to ground truth.

`phase6-embedding-benchmark-v1` reports are superseded for final Phase 6 acceptance because the first timed iteration included model acquisition/load and an unlabeled language slice could be reported as Macro F1 `1.0`. The corrected `phase6-embedding-benchmark-v2` performs an explicit warm-up before measured full-batch iterations and emits `null` for language slices without ground-truth coverage.

## Local verification commands

Compile/import-safe syntax check:

```powershell
python -m compileall -q src evaluation tests
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
```

Run the full regression suite:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { throw "test suite failed" }
```

Run the default SigLIP2 candidate twice for deterministic rerun evidence:

```powershell
python -m evaluation.embedding_benchmark `
  --manifest "$env:DATASET_ROOT\manifest.json" `
  --dataset-root "$env:DATASET_ROOT" `
  --model-id "google/siglip2-base-patch16-224" `
  --model-revision "main" `
  --iterations 2 `
  --recall-k 5 `
  --output "$env:PHASE6_REPORT_ROOT\siglip2-base-patch16-224.json"
if ($LASTEXITCODE -ne 0) { throw "Phase 6 embedding benchmark failed" }
```

Inspect the report:

```powershell
$Report = Get-Content "$env:PHASE6_REPORT_ROOT\siglip2-base-patch16-224.json" -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,dataset_id,dataset_version,manifest_fingerprint,dataset_content_fingerprint,model_id,model_version,provider,provider_version,embedding_dimension,eval_sample_count,theme_label_count,theme_macro_f1,mean_recall_at_k,recall_k,map,warmup_latency_ms,latency_scope,p50_batch_latency_ms,p95_batch_latency_ms,peak_process_ram_mb,process_vram_mb,deterministic_rerun,near_duplicate_group_count
$Report.cross_language_theme_macro_f1 | ConvertTo-Json -Depth 4
$Report.theme_similarity_diagnostics | ConvertTo-Json -Depth 4
$Report.candidate_thresholds | ConvertTo-Json -Depth 4
```

For candidate comparison, rerun the same command with each approved multimodal image-text encoder candidate and a separate output file. Do not change the dataset, thresholds, `recall-k`, or iteration count between candidates unless the comparison explicitly records that change.

## Expected result

- Compile and full unit/integration/architecture/evaluation regression are green.
- DI discovery resolves `IEmbeddingProvider` to `TransformersEmbeddingProvider` and resolves `SemanticThemeService` without manual registration.
- NaN/Inf, zero-norm and inconsistent dimensions are rejected.
- Batch order is preserved and checked.
- Duplicate-content images with distinct IDs remain distinct inputs and can form deterministic near-duplicate groups.
- Theme and near-duplicate thresholds come from config and boundary tests pass.
- Sensitive person-level candidate labels are rejected before semantic matching.
- Benchmark output contains dataset/model/config traceability, Recall@K, mAP, theme F1, cross-language slice F1, latency and resource fields.
- No candidate or threshold is promoted automatically.

## Verified local evidence

Target-Windows verification was completed on the authorized Phase 2 evaluation dataset after the final Phase 6 naming correction.

- verified source commit before closure: `7cdc076e6da7a7e8df53aff9a73458def3fee5b3`
- compile/import check: passed
- Phase 6 architecture suite: 7 tests passed
- full repository regression: 217 tests passed with no failures or errors
- dataset_id: `profile_insight_real_eval`
- dataset_version: `1.0.0`
- manifest fingerprint: `06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc`
- dataset-content fingerprint: `5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e`
- report schema: `phase6-embedding-benchmark-v2`
- model candidate: `google/siglip2-base-patch16-224`
- provider/version: `transformers 5.18.0`
- embedding dimension: `768`
- mean Recall@5: `0.9907407407407407`
- mAP: `0.9182539682539683`
- theme Macro F1 at candidate threshold `0.25`: `0.0`
- deterministic rerun: `true`
- warm-up/model-load latency: `145587.7361 ms`
- post-warmup full-eval p50 batch latency: `37288.23 ms`
- post-warmup full-eval p95 batch latency: `42332.6675 ms`
- peak process RAM: `2132.6171875 MB`
- process VRAM: unavailable/not applicable for the CPU candidate run
- near-duplicate groups detected on the current eval set: `0`
- mixed Persian/English slice Macro F1: unavailable (`null`) because that slice lacks authorized ground-truth labels in the current dataset

Similarity diagnostics explain the zero thresholded theme F1 rather than hiding it: the highest observed expected-label cosine similarity was `0.14266678429109328`, below the candidate threshold `0.25`, so the benchmark produced zero threshold matches. Retrieval quality remained strong. This result is retained as evidence and does not justify post-hoc threshold tuning from the eval set.

The candidate model and both similarity thresholds therefore remain configurable and are **not** frozen as production-final. Broader candidate comparison and final production model/threshold selection remain a Phase 10 hardening/benchmark responsibility. Phase 6 closes because the provider-neutral contract, deterministic semantic/retrieval behavior, threshold configurability, safety validation, target-machine reproducibility, and required benchmark evidence have all been demonstrated.

The local working tree contained only untracked benchmark JSON artifacts; no tracked production/test/documentation changes were pending at verification time.

## Status

`COMPLETE / READY_FOR_NEXT_PHASE`

Phase 6 is locally verified on the target Windows environment. Its embedding contracts, provider boundary, semantic theme workflow, deterministic retrieval and near-duplicate logic, safety checks, resource measurements, and real-dataset benchmark evidence are complete for this phase. No embedding model or similarity threshold is declared production-final by this closure.
