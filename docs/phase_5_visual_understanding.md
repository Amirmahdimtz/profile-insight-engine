# Phase 5 — Visual Understanding

## Scope

Phase 5 adds only `CanonicalImage → Vision Provider → strict structured validation → Visual Evidence`.
It produces observable `scene`, `object`, `activity`, `topic`, and a short factual caption. It does not add embeddings, image similarity, semantic themes, near-duplicate detection, cross-image aggregation, Insight generation, API, persistence, SQLAlchemy, queues/workers, or any Phase 6+ component.

## Contract and architecture

`IVisionProvider` is a real replaceable provider boundary because the local runtime/model is benchmarkable and replaceable. `LlamaCppVisionProvider` is the concrete Infrastructure implementation. Its structured-output request uses llama.cpp server's native `response_format.type=json_schema` plus direct `response_format.schema`; provider validation remains fail-closed if the runtime returns malformed or truncated JSON. `EvidenceExtractionService` is the Core orchestration/normalization boundary and depends only on `IVisionProvider` plus provider-neutral Phase 5 contracts.

DI follows the existing discovery mechanism: `@inject` on the concrete provider and Core service, constructor injection, and no feature-specific manual registration.

The concrete provider uses llama.cpp's local OpenAI-compatible server. Model/runtime-specific HTTP payloads, base64 image transport, schema-constrained JSON, and response parsing remain in Infrastructure and do not cross into Core. The provider validates the exact structured response shape before returning a `VisionProviderResult`.

Every emitted standard `Evidence` remains directly traceable to `image_id`, provider, provider build/version, model identifier/version, config version, and confidence semantics. Evidence IDs are deterministic hashes of image/type/normalized value.

## Visual output validation

The structured provider response contains exactly:

- `scenes[]`
- `objects[]`
- `activities[]`
- `topics[]`
- optional `caption`

Each observation contains only a trimmed label and a finite confidence in `[0,1]`. Unknown fields, malformed JSON, missing fields, invalid confidence, empty strings, duplicate observations, invalid result types, mismatched `image_id`, non-vision runtime capability, timeout, runtime failure, and malformed responses fail deterministically with sanitized errors.

The Phase 1 observable-claim validator remains the policy source of truth. Phase 5 does not define a second sensitive-trait policy. Sensitive/person-level inferences are rejected before they can become standard Evidence. Raw model responses and image bytes are not written to operational logs.

## Confidence semantics

The candidate VLMs do not expose a calibrated classifier probability for these generated labels. Phase 5 therefore records provider confidence as:

`model_self_reported_uncalibrated`

The value is bounded and traceable, but it is **not** treated as calibrated probability and no production confidence threshold is frozen in this phase. Calibration/threshold decisions require project benchmark evidence.

## Local runtime strategy

The first concrete runtime is `llama.cpp` because the current repository has no Python ML runtime dependency and llama.cpp provides a local/offline server, multimodal image input, and schema-constrained JSON without adding PyTorch/Transformers/ONNX to the service process.

The default config points at a benchmark candidate only; it is not a production-final model selection.

Official references:

- llama.cpp server: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- llama.cpp model/cache docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md
- Qwen2.5-VL 3B GGUF: https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF
- Qwen2.5-VL 7B GGUF: https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF
- Gemma 3 4B GGUF: https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF

Initial benchmark candidates:

| Candidate | Quantized model spec | Reason for inclusion | License/source note |
| --- | --- | --- | --- |
| `qwen2.5-vl-3b` | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M` | smaller local VLM baseline; official llama.cpp usage is published on the model page | Apache-2.0 |
| `gemma3-4b` | `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M` | second model family in the small/medium range | Gemma license; review deployment terms before production use |
| `qwen2.5-vl-7b` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M` | larger Qwen comparison point for quality/resource trade-off | Apache-2.0 |

No candidate is declared the winner before the real project benchmark.

## Dataset audit

The repository does not contain the real authorized Phase 2 dataset; the verified Phase 4 documentation establishes the local dataset identity and fingerprints, but not the current visual-label coverage.

`evaluation.vision_benchmark` therefore audits the actual eval manifest before model comparison and reports:

- eval sample count;
- object/scene/activity/topic label counts;
- per-slice visual label counts for all Phase 2 slices;
- missing required visual label types.

The current Phase 2 manifest schema has no caption-ground-truth field. Phase 5 tests caption schema, safety and deterministic mapping, but does not fabricate caption accuracy. If object/scene/activity/topic coverage is insufficient, additional human/authorized annotations are required; model output must not be copied into ground truth.

## Benchmark

`evaluation.vision_benchmark` reuses the verified Phase 2 manifest/reference fingerprints, Phase 3 canonicalization path, the production `LlamaCppVisionProvider`, `EvidenceExtractionService`, and the existing Phase 2 Precision/Recall/F1 and unsupported-claim metric.

For each candidate it records, where measurable:

- dataset ID/version and fingerprints;
- provider build info;
- model source spec, local model path, and SHA-256 when the runtime exposes a readable GGUF path;
- object/scene/activity/topic Precision, Recall and F1 plus overall metrics;
- unsupported-claim rate on strictly parsed pre-policy provider claims, post-validation Evidence, and failed/rejected sample count;
- p50/p95 per-image latency;
- model acquisition/warm-up time;
- model load time measured from a second offline startup after acquisition;
- sampled llama.cpp process RAM peak;
- sampled NVIDIA process VRAM peak when `nvidia-smi` is available, otherwise `null`;
- deterministic rerun status (`null` when fewer than two iterations are requested);
- sanitized per-candidate failure reasons and a `benchmark_valid` flag;
- visual-label coverage and missing-annotation gaps.

Candidate processes are run sequentially with the same dataset, decode settings, iterations, and metric configuration. Before measurement, each candidate is started once with network-enabled `-hf` so llama.cpp can populate `LLAMA_CACHE` with the model and multimodal projector. That acquisition/warm-up has its own configurable timeout. The measured server is then restarted with `--offline`; therefore `model_load_time_ms` excludes network download time and represents local cached startup. Failed samples remain in the expected-label denominator so provider failures cannot artificially improve recall. If every eval sample fails for a candidate, the report is still written with sanitized failure reasons, `benchmark_valid` is false, and the CLI exits non-zero. Diagnostic runs may override `request_timeout_seconds` and `max_tokens` from the benchmark CLI without changing the production vision configuration; the effective values are recorded in each candidate report. When the candidate artifacts are already cached, diagnostic runs may also use `--skip-model-acquisition` to avoid the network-enabled warm-up/reload cycle; final benchmark runs must leave acquisition enabled so acquisition and offline load timings remain measurable. For repeated local diagnostics, `--existing-runtime-base-url` can reuse one already-running local llama.cpp server so model acquisition and model load are not repeated; this mode is diagnostic-only and does not provide startup/RAM/VRAM measurements.

## Windows local verification prerequisites

Run commands from the repository root in a fresh PowerShell session.

First synchronize the local `main` branch before interpreting any verification result. Existing untracked benchmark reports do not need to be deleted, but tracked local modifications must be reviewed before pulling.

```powershell
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "Unable to fast-forward local main. Review local tracked changes before continuing." }
git rev-parse HEAD
```

The HEAD printed above must match the latest Phase 5 verification commit documented in the current instructions.

Set the authorized dataset, model cache and report paths. The model cache is intentionally outside the Git repository. `LLAMA_CACHE` is an official llama.cpp cache control.

```powershell
$env:DATASET_ROOT = "E:\phase2-real-dataset"
$env:MODEL_ROOT = "E:\profile-insight-models"
$env:LLAMA_CACHE = Join-Path $env:MODEL_ROOT "llama-cache"
$env:PHASE5_REPORT = Join-Path (Get-Location) "phase5_vision_benchmark.json"
$env:EXPECTED_MANIFEST_FINGERPRINT = "06fcf3e196d3f3d4fea9d284fa162c0d59b14a85aa662beb2bd34d42db0a15bc"
$env:EXPECTED_DATASET_CONTENT_FINGERPRINT = "5ac729a000b6d0170ea74d7ac1ed688a472771c02ca27ca2c59664c105fb7d8e"
New-Item -ItemType Directory -Force -Path $env:LLAMA_CACHE | Out-Null
```

Install Python dependencies already pinned by the repository:

```powershell
python -m pip install -r requirements.txt
```

Install/verify llama.cpp. The benchmark uses the `llama-server` executable directly. WinGet installs it as a portable package, so verification resolves the actual executable path instead of depending only on PATH propagation.

```powershell
$LlamaServerCommand = Get-Command llama-server -ErrorAction SilentlyContinue

if ($LlamaServerCommand) {
    $env:LLAMA_SERVER_EXE = $LlamaServerCommand.Source
} else {
    $WinGetCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\llama-server.exe"),
        "C:\Program Files\WinGet\Links\llama-server.exe",
        "C:\Program Files (x86)\WinGet\Links\llama-server.exe"
    )

    $env:LLAMA_SERVER_EXE = $WinGetCandidates |
        Where-Object { Test-Path $_ -PathType Leaf } |
        Select-Object -First 1

    if (-not $env:LLAMA_SERVER_EXE) {
        $WinGetRoots = @(
            (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"),
            "C:\Program Files\WinGet\Packages",
            "C:\Program Files (x86)\WinGet\Packages"
        ) | Where-Object { Test-Path $_ -PathType Container }

        $env:LLAMA_SERVER_EXE = $WinGetRoots |
            ForEach-Object {
                Get-ChildItem -Path $_ -Filter "llama-server.exe" -File -Recurse -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -match "ggml\.llamacpp" } |
                    Select-Object -ExpandProperty FullName
            } |
            Select-Object -First 1
    }
}

if (-not $env:LLAMA_SERVER_EXE) {
    throw "llama-server.exe could not be resolved. Verify the ggml.llamacpp WinGet installation."
}
if (-not (Test-Path $env:LLAMA_SERVER_EXE -PathType Leaf)) {
    throw "Resolved llama-server executable does not exist: $env:LLAMA_SERVER_EXE"
}

& $env:LLAMA_SERVER_EXE --version
if ($LASTEXITCODE -ne 0) { throw "llama-server version check failed" }

Write-Host "Resolved llama-server: $env:LLAMA_SERVER_EXE"
```

The explicit executable path is a local verification input only; no machine-specific path is committed to application config.

Detect NVIDIA GPU availability without assuming CUDA/GPU support. The llama.cpp provider can run on CPU; GPU usage is not a Phase 5 prerequisite.

```powershell
$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    nvidia-smi
} else {
    Write-Host "nvidia-smi not found; benchmark will record VRAM as null and may run on CPU/other llama.cpp backend."
}
```

The `-hf` model specs used by the benchmark download/cache weights and the available multimodal projector through llama.cpp under `$env:LLAMA_CACHE`; no model weights are committed to the repository. `vision.model_acquisition_timeout_seconds` controls only this pre-measurement acquisition/warm-up budget; `vision.startup_timeout_seconds` remains the measured offline startup budget.

## Local verification commands

Before compile/tests, confirm the synchronized commit and resolved runtime:

```powershell
git status --short
git rev-parse HEAD
if (-not $env:LLAMA_SERVER_EXE) { throw "LLAMA_SERVER_EXE is not set. Run the llama.cpp resolver above first." }
if (-not (Test-Path $env:LLAMA_SERVER_EXE -PathType Leaf)) { throw "Resolved llama-server executable does not exist: $env:LLAMA_SERVER_EXE" }
& $env:LLAMA_SERVER_EXE --version
if ($LASTEXITCODE -ne 0) { throw "llama-server version check failed" }
```

An untracked `phase4_ocr_benchmark.json` is allowed and does not invalidate Phase 5 verification. Do not delete it solely to make `git status` empty.

Compile/import-safe syntax check:

```powershell
python -m compileall -q src evaluation tests
if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
```

Run the full regression suite, including Phases 1–5:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { throw "test suite failed" }
```

Run the real Phase 5 candidate benchmark:

```powershell
python -m evaluation.vision_benchmark `
  --manifest "$env:DATASET_ROOT\manifest.json" `
  --dataset-root "$env:DATASET_ROOT" `
  --candidate "qwen2.5-vl-3b=ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M" `
  --candidate "gemma3-4b=ggml-org/gemma-3-4b-it-GGUF:Q4_K_M" `
  --candidate "qwen2.5-vl-7b=ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M" `
  --iterations 2 `
  --expected-manifest-fingerprint "$env:EXPECTED_MANIFEST_FINGERPRINT" `
  --expected-dataset-content-fingerprint "$env:EXPECTED_DATASET_CONTENT_FINGERPRINT" `
  --runtime-executable "$env:LLAMA_SERVER_EXE" `
  --output "$env:PHASE5_REPORT"
if ($LASTEXITCODE -ne 0) { throw "Phase 5 benchmark failed" }
```

Inspect dataset coverage and reproducibility/resource metrics:

```powershell
$Report = Get-Content "$env:PHASE5_REPORT" -Raw | ConvertFrom-Json
$Report | Select-Object schema_version,dataset_id,dataset_version,manifest_fingerprint,dataset_content_fingerprint,iterations
$Report.visual_label_coverage | ConvertTo-Json -Depth 8
$Report.candidates | Select-Object candidate,hf_model,provider,provider_version,model_sha256,sample_count,failed_sample_count,precision,recall,f1,unsupported_claim_rate_pre_policy,unsupported_claim_rate_post_validation,p50_latency_ms,p95_latency_ms,model_acquisition_time_ms,model_load_time_ms,runtime_peak_ram_mb,vram_peak_mb,deterministic_rerun | Format-Table -AutoSize
$Report.candidates | ForEach-Object { $_.per_label_type | ConvertTo-Json -Depth 6 }
```

Verify model fingerprints are present whenever `/props.model_path` points to a readable local GGUF:

```powershell
$Report.candidates | ForEach-Object {
    if ($_.model_path -and (Test-Path $_.model_path -PathType Leaf)) {
        $LocalHash = (Get-FileHash -Algorithm SHA256 $_.model_path).Hash.ToLowerInvariant()
        if ($_.model_sha256 -and $LocalHash -ne $_.model_sha256) {
            throw "Model SHA-256 mismatch for $($_.candidate)"
        }
        [pscustomobject]@{ Candidate=$_.candidate; ModelPath=$_.model_path; SHA256=$LocalHash }
    }
}
```

## Expected result

- Compile and full unit/integration/architecture/evaluation regression are green.
- DI discovery resolves `IVisionProvider` to `LlamaCppVisionProvider` and resolves `EvidenceExtractionService` without manual registration.
- The benchmark validates the exact Phase 2 dataset fingerprints before inference.
- Visual-label coverage is reported rather than assumed.
- Each candidate report contains actual Precision/Recall/F1, pre-policy and post-validation unsupported-claim rates, rejected/failed sample count, p50/p95 latency, acquisition/warm-up time, offline model-load time, reproducibility status, model/runtime traceability, RAM, and VRAM when measurable.
- No candidate is promoted automatically.
- If required visual label types are missing/insufficient or candidate runs are not reproducible, Phase 5 remains open and the missing authorized human annotations/runtime evidence must be supplied.

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

Phase 5 must not be marked `COMPLETE / READY_FOR_NEXT_PHASE` until the target Windows environment provides the real visual-label coverage, multi-candidate benchmark, performance/resource evidence, deterministic rerun evidence, and full regression result.
