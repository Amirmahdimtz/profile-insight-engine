# Phase 1 — Requirements & Analysis Contract

This document fixes only the product vocabulary and data-contract invariants required before ML work. It does not select an OCR/Vision/Embedding provider, threshold, persistence model, API lifecycle, worker, or deployment runtime.

## Terminology

- **Analysis**: one request identified by `analysis_id` over one or more uniquely identified images.
- **Image**: one analysis member identified by `image_id`. An `ImageAnalysisResult` belongs to exactly one `analysis_id`.
- **Evidence**: one normalized, observable support unit tied to exactly one image. Provider-native/raw output is not the Evidence contract.
- **Theme**: an observable recurring/content theme backed by explicit evidence IDs and the exact images containing that evidence.
- **Insight**: an evidence-backed content/profile observation. It must not encode unsupported sensitive or internal personal traits.
- **Evidence count**: exactly the number of unique supporting Evidence IDs.
- **Image coverage**: `unique supporting images / total images in the analysis`.
- **Supporting image IDs**: exactly the image IDs reached by the supporting Evidence IDs, not an independently supplied approximation.

## Analysis boundary

Supported contract categories are observable content such as object, scene, OCR/text, activity, environment, non-sensitive visible interest, brand/team reference, religious-themed content, recurring group/social content, and recurring observable themes.

The contract rejects structured claims that assert sensitive or internal traits of the person, including religion, political orientation/ideology, ethnicity, mental health, sexual orientation, intelligence, honesty, family relationship, or inner/personality traits. Observable content and personal-trait attribution are intentionally distinct:

- Allowed: `religious-themed content detected`
- Rejected: `person religion = ...`

OCR/raw textual Evidence may contain arbitrary observed text; the policy applies to normalized claim labels/keys and Insight explanations, not to quoted text merely present in an image.

## Validation invariants

- IDs must be non-empty and trimmed.
- Request image IDs must be unique and non-empty.
- The Phase 1 contract does not hard-code a maximum image count. A configured caller-supplied `max_images` can be validated through `ProfileAnalysisRequest.validate_image_count()`.
- Confidence and image coverage are finite numbers in `[0, 1]`; NaN and Infinity are rejected.
- Strict `from_dict` / `from_json` boundaries reject unknown fields.
- Evidence IDs are unique across an analysis graph.
- Every Evidence image reference must resolve to its containing image.
- Every Theme/Insight evidence reference and supporting image reference must resolve inside the same analysis.
- `evidence_count` must match the number of supporting Evidence IDs.
- `supporting_image_ids` must exactly equal the images reached by supporting Evidence.
- `image_coverage` must equal supporting-image count divided by total-image count.
- Serialization uses sorted object keys, compact separators, UTF-8 text semantics, and disallows NaN/Infinity.

## Phase boundary

No Phase 2+ feature is implemented here: no evaluation dataset/benchmark runner, image preprocessing, OCR, VLM/Vision provider, embedding, aggregation runtime, insight-generation runtime, API, persistence, queue/worker, migration, model loading, GPU runtime, or deployment configuration.
