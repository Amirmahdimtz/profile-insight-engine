# Personality Analysis Engine Extension

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

## Goal

Add a deterministic personality-analysis engine for explicit self-report questionnaire data while preserving the product rule that personality must not be inferred from images.

The engine scores the Big Five dimensions:

- openness
- conscientiousness
- extraversion
- agreeableness
- neuroticism

## Non-goals

This extension does not:

- infer personality from face, appearance, clothing, OCR, Vision, or image content;
- ship or invent questionnaire wording;
- claim that a caller-supplied questionnaire is scientifically validated;
- apply population norms or percentile interpretation;
- create clinical or mental-health diagnoses;
- fabricate confidence values or high/medium/low labels.

## Input contract

`request_json` may contain an optional `personality_questionnaire`:

```json
{
  "instrument_id": "authorized-big-five-form",
  "instrument_version": "1.0",
  "scale_min": 1,
  "scale_max": 5,
  "responses": [
    {
      "item_id": "o-1",
      "trait": "openness",
      "response_value": 5,
      "reverse_scored": false
    },
    {
      "item_id": "c-1",
      "trait": "conscientiousness",
      "response_value": 4
    },
    {
      "item_id": "e-1",
      "trait": "extraversion",
      "response_value": 3
    },
    {
      "item_id": "a-1",
      "trait": "agreeableness",
      "response_value": 4
    },
    {
      "item_id": "n-1",
      "trait": "neuroticism",
      "response_value": 2,
      "reverse_scored": true
    }
  ],
  "consent_confirmed": true,
  "basis": "explicit_self_report_questionnaire"
}
```

Contract rules:

- consent is mandatory;
- `basis` is fixed to `explicit_self_report_questionnaire`;
- instrument ID/version and item IDs must be non-empty trimmed strings;
- scale values must be finite and `scale_min < scale_max`;
- every response must be finite and inside the declared scale;
- item IDs must be unique;
- at least one item must be supplied for every Big Five trait;
- configured maximum questionnaire size is enforced before repository persistence.

## Scoring contract

For each item:

```text
adjusted = response_value
if reverse_scored:
    adjusted = scale_min + scale_max - response_value
```

For each trait:

```text
scored_mean = mean(adjusted responses)
normalized_score = (scored_mean - scale_min) / (scale_max - scale_min)
```

Scores are rounded deterministically to six decimal places.

The engine does not convert normalized scores into qualitative personality labels because such thresholds require an instrument-specific validated interpretation contract.

## Output contract

The completed result adds an optional top-level `personality_profile`:

```json
{
  "instrument_id": "authorized-big-five-form",
  "instrument_version": "1.0",
  "model": "big_five_self_report_scoring_v1",
  "source": "explicit_self_report_questionnaire",
  "traits": [
    {
      "trait": "openness",
      "normalized_score": 0.75,
      "scored_mean": 4.0,
      "item_count": 2,
      "supporting_item_ids": ["o-1", "o-2"],
      "explanation": "..."
    }
  ],
  "summary": "Self-reported Big Five scores: ...",
  "disclaimer": "..."
}
```

The profile is separate from image-derived `Evidence`, `Theme`, `ProfileInsight`, and the existing automatic image-insight `summary`.

## Persistence and privacy

Raw questionnaire response values are not persisted in the completed result.

Persisted data contains:

- instrument ID/version;
- derived trait scores;
- item counts;
- supporting item IDs;
- deterministic explanations;
- summary and disclaimer.

The persisted result schema is now `profile-analysis-result-v3`.

Backward compatibility:

- `phase9-analysis-result-v1` remains readable;
- `profile-analysis-result-v2` remains readable;
- v1/v2 rows deserialize with `personality_profile=null`.

No Alembic migration is required because completed results already use the existing JSONB payload.

## Architecture

```text
ProfileAnalysisController
        |
        v
ProfileAnalysisService
        |
        +--> PersonalityAnalysisService
        |
        +--> existing image/OCR/aggregation/insight workflow
```

`PersonalityAnalysisService` lives in Core/services, uses constructor DI, and contains only deterministic scoring/business rules.

Controller responsibility remains DTO validation/mapping only.

## Configuration

```yaml
profile_analysis:
  max_personality_questionnaire_items: 100
```

Existing `max_self_reported_claim_chars` also limits instrument IDs, versions, and questionnaire item IDs.

## Local verification

Run from repository root:

```powershell
python -m compileall src evaluation scripts tests

if ($LASTEXITCODE -ne 0) {
    throw "Compile/import check failed"
}

python -m unittest `
  tests.test_personality_analysis_service `
  tests.test_profile_analysis_controller `
  tests.test_profile_analysis_service `
  tests.test_phase9_contracts `
  tests.test_phase9_architecture `
  tests.test_phase9_di_discovery `
  -v

if ($LASTEXITCODE -ne 0) {
    throw "Personality analysis targeted regression failed"
}

python scripts/verify_phase10.py

if ($LASTEXITCODE -ne 0) {
    throw "Full project verification failed"
}
```

The extension becomes verified only after local output is returned and all required checks are green.
