# Self-Reported Profile Claims Extension

## Status

`IMPLEMENTED_AWAITING_LOCAL_VERIFICATION`

This additive post-acceptance extension introduces an explicit-consent self-report channel. It does not change the image-inference safety boundary and it does not enable production Vision.

## Goal

Allow callers to attach structured claims that the person has explicitly stated about themselves, including categories that must never be inferred from appearance.

Supported categories:

- `religion`
- `political_orientation`
- `ethnicity`
- `mental_health`
- `sexual_orientation`
- `personality`
- `intelligence`
- `other_self_description`

Supported provenance types:

- `questionnaire`
- `bio`
- `post`
- `direct_statement`
- `other`

Every claim must use `basis=explicit_self_report` and `consent_confirmed=true`.

Image, OCR and Vision sources are intentionally unrepresentable as self-report provenance.

## API contract

`POST /api/v1/profile_analysis/` keeps the existing multipart shape. The `request_json` object may now include:

```json
{
  "analysis_id": "analysis-1",
  "image_ids": ["img-1"],
  "self_reported_claims": [
    {
      "claim_id": "claim-1",
      "category": "personality",
      "value": "introverted",
      "source_id": "questionnaire-q2",
      "source_type": "questionnaire",
      "consent_confirmed": true,
      "basis": "explicit_self_report"
    }
  ]
}
```

The response returns `self_reported_claims` as a separate top-level collection. Claims are not converted into image-derived `Evidence`, `Theme`, or `ProfileInsight` objects and are not included in the automatic insight summary.

## Privacy and persistence

Raw questionnaire, bio or post text is not accepted by this contract.

Only the structured claim fields and provenance are persisted. This is sensitive user-provided data and must be protected by the same database access controls as the rest of the persisted analysis result.

The persisted result schema is now `profile-analysis-result-v2`. Existing `phase9-analysis-result-v1` rows remain readable and deserialize with an empty self-report collection.

No database migration is required because the completed result is already stored in the existing JSONB payload.

## Configuration

```yaml
profile_analysis:
  max_self_reported_claims: 50
  max_self_reported_claim_chars: 256
```

These limits are enforced in Core before a repository row is created.

## Local verification

Run from repository root:

```powershell
python -m unittest `
  tests.test_self_report_contracts `
  tests.test_profile_analysis_controller `
  tests.test_profile_analysis_service `
  tests.test_phase9_contracts `
  tests.test_phase9_architecture `
  -v

if ($LASTEXITCODE -ne 0) {
    throw "Self-report extension targeted regression failed"
}

python -m unittest discover -s tests -p "test_*.py" -q

if ($LASTEXITCODE -ne 0) {
    throw "Full repository regression failed"
}
```

The extension becomes verified only after the local output is returned and all tests are green.
