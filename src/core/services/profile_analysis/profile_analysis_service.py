from __future__ import annotations

import json
from typing import Mapping, Sequence

from src.core.profile_analysis.contracts import (
    Evidence,
    EvidenceType,
    ImageAnalysisResult,
    ProfileAnalysisRequest,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
)
from src.core.profile_analysis.image_contracts import ImageValidationError, ProcessedImageBatch, RawImageInput
from src.core.profile_analysis.lifecycle_contracts import CompletedProfileAnalysis, ProfileAnalysisLifecycle
from src.core.profile_analysis.self_report_contracts import SelfReportedClaim
from src.core.services.profile_analysis.evidence_aggregation_service import EvidenceAggregationService
from src.core.services.profile_analysis.evidence_extraction_service import EvidenceExtractionService
from src.core.services.profile_analysis.image_processing_service import ImageProcessingService
from src.core.services.profile_analysis.insight_generation_service import InsightGenerationService
from src.core.services.profile_analysis.ocr_evidence_service import OcrEvidenceService
from src.infrastructure.di.inject import inject
from src.infrastructure.models.profile_analysis import ProfileAnalysisModel
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import (
    ProfileAnalysisAlreadyExistsError as RepositoryAlreadyExistsError,
    ProfileAnalysisRepository,
    ProfileAnalysisTransitionConflictError as RepositoryTransitionConflictError,
)
from src.infrastructure.utils.config_reader import ConfigReader


_PERSISTED_RESULT_SCHEMA_V1 = "phase9-analysis-result-v1"
_PERSISTED_RESULT_SCHEMA_V2 = "profile-analysis-result-v2"
_ALLOWED_TRANSITIONS = {
    ProfileAnalysisStatus.PENDING: frozenset({ProfileAnalysisStatus.IN_PROGRESS}),
    ProfileAnalysisStatus.IN_PROGRESS: frozenset(
        {ProfileAnalysisStatus.COMPLETED, ProfileAnalysisStatus.FAILED}
    ),
    ProfileAnalysisStatus.COMPLETED: frozenset(),
    ProfileAnalysisStatus.FAILED: frozenset(),
}


class ProfileAnalysisServiceError(RuntimeError):
    """Base Phase 9 workflow error with transport-neutral semantics."""


class ProfileAnalysisAlreadyExistsError(ProfileAnalysisServiceError):
    pass


class ProfileAnalysisNotFoundError(ProfileAnalysisServiceError):
    pass


class ProfileAnalysisNotReadyError(ProfileAnalysisServiceError):
    pass


class ProfileAnalysisInputError(ProfileAnalysisServiceError):
    pass


class ProfileAnalysisExecutionError(ProfileAnalysisServiceError):
    pass


class ProfileAnalysisTransitionError(ProfileAnalysisServiceError):
    pass


@inject
class ProfileAnalysisService:
    def __init__(
        self,
        image_processing_service: ImageProcessingService,
        ocr_evidence_service: OcrEvidenceService,
        evidence_extraction_service: EvidenceExtractionService,
        evidence_aggregation_service: EvidenceAggregationService,
        insight_generation_service: InsightGenerationService,
        profile_analysis_repository: ProfileAnalysisRepository,
        config_reader: ConfigReader,
    ):
        self._image_processing_service = image_processing_service
        self._ocr_evidence_service = ocr_evidence_service
        self._evidence_extraction_service = evidence_extraction_service
        self._evidence_aggregation_service = evidence_aggregation_service
        self._insight_generation_service = insight_generation_service
        self._profile_analysis_repository = profile_analysis_repository
        vision_enabled = config_reader.get("vision.enabled")
        if not isinstance(vision_enabled, bool):
            raise ValueError("vision.enabled must be boolean")
        self._vision_enabled = vision_enabled
        self._max_self_reported_claims = config_reader.get_positive_int(
            "profile_analysis.max_self_reported_claims"
        )
        self._max_self_reported_claim_chars = config_reader.get_positive_int(
            "profile_analysis.max_self_reported_claim_chars"
        )

    async def create_async(
        self,
        request: ProfileAnalysisRequest,
        images: Sequence[RawImageInput],
        self_reported_claims: Sequence[SelfReportedClaim] = (),
    ) -> CompletedProfileAnalysis:
        claims = self._validate_self_reported_claims(self_reported_claims)
        try:
            await self._profile_analysis_repository.create_pending_async(
                request.analysis_id,
                request.image_ids,
            )
        except RepositoryAlreadyExistsError as exc:
            raise ProfileAnalysisAlreadyExistsError(str(exc)) from exc

        await self._transition_async(
            request.analysis_id,
            ProfileAnalysisStatus.PENDING,
            ProfileAnalysisStatus.IN_PROGRESS,
        )

        batch: ProcessedImageBatch | None = None
        try:
            try:
                batch = await self._image_processing_service.process_async(request, images)
            except ImageValidationError as exc:
                raise ProfileAnalysisInputError(str(exc)) from exc

            image_results: list[ImageAnalysisResult] = []
            evidence: list[Evidence] = []
            for image in batch.images:
                ocr_result = await self._ocr_evidence_service.extract_async(image)
                vision_evidence: tuple[Evidence, ...] = ()
                if self._vision_enabled:
                    vision_result = await self._evidence_extraction_service.extract_async(image)
                    vision_evidence = vision_result.evidence
                image_evidence = tuple(
                    sorted(
                        (*ocr_result.evidence, *vision_evidence),
                        key=lambda item: item.id,
                    )
                )
                evidence.extend(image_evidence)
                image_results.append(
                    ImageAnalysisResult(
                        analysis_id=request.analysis_id,
                        image_id=image.image_id,
                        evidence=image_evidence,
                    )
                )

            evidence_tuple = tuple(sorted(evidence, key=lambda item: item.id))
            aggregates = self._evidence_aggregation_service.aggregate(batch, evidence_tuple)
            insight_result = self._insight_generation_service.generate(
                aggregates,
                evidence_tuple,
            )
            result = ProfileAnalysisResult(
                analysis_id=request.analysis_id,
                status=ProfileAnalysisStatus.COMPLETED,
                images=tuple(image_results),
                themes=(),
                insights=insight_result.insights,
            )
            completed = CompletedProfileAnalysis.from_insight_result(
                result,
                insight_result,
                self_reported_claims=claims,
            )
            persisted = self._sanitize_for_persistence(completed)
            payload = self._serialize_completed(persisted)
            await self._image_processing_service.release_async(batch)
            batch = None
            await self._transition_async(
                request.analysis_id,
                ProfileAnalysisStatus.IN_PROGRESS,
                ProfileAnalysisStatus.COMPLETED,
                result_payload=payload,
            )
            return persisted
        except (ProfileAnalysisInputError, ProfileAnalysisTransitionError):
            await self._mark_failed_async(request.analysis_id)
            raise
        except Exception as exc:
            await self._mark_failed_async(request.analysis_id)
            raise ProfileAnalysisExecutionError("profile analysis execution failed") from exc
        finally:
            if batch is not None:
                try:
                    await self._image_processing_service.release_async(batch)
                except Exception:
                    pass

    def _validate_self_reported_claims(
        self,
        claims: Sequence[SelfReportedClaim],
    ) -> tuple[SelfReportedClaim, ...]:
        if (
            isinstance(claims, (str, bytes, bytearray))
            or not isinstance(claims, Sequence)
        ):
            raise ProfileAnalysisInputError(
                "self_reported_claims must be a sequence"
            )
        items = tuple(claims)
        if len(items) > self._max_self_reported_claims:
            raise ProfileAnalysisInputError(
                "self-reported claim count exceeds configured limit"
            )
        if not all(isinstance(item, SelfReportedClaim) for item in items):
            raise ProfileAnalysisInputError(
                "self_reported_claims must contain SelfReportedClaim values"
            )
        claim_ids = [item.claim_id for item in items]
        if len(claim_ids) != len(set(claim_ids)):
            raise ProfileAnalysisInputError(
                "self_reported_claims must have unique claim_id values"
            )
        for item in items:
            if len(item.value) > self._max_self_reported_claim_chars:
                raise ProfileAnalysisInputError(
                    f"self-reported claim '{item.claim_id}' exceeds configured text limit"
                )
        return tuple(sorted(items, key=lambda item: item.claim_id))

    async def recover_interrupted_async(self) -> int:
        return await self._profile_analysis_repository.fail_interrupted_async()

    async def get_status_async(self, analysis_id: str) -> ProfileAnalysisLifecycle:
        entity = await self._profile_analysis_repository.get_by_id_async(analysis_id)
        if entity is None:
            raise ProfileAnalysisNotFoundError(f"analysis '{analysis_id}' not found")
        return self._to_lifecycle(entity)

    async def get_result_async(self, analysis_id: str) -> CompletedProfileAnalysis:
        entity = await self._profile_analysis_repository.get_by_id_async(analysis_id)
        if entity is None:
            raise ProfileAnalysisNotFoundError(f"analysis '{analysis_id}' not found")
        status = self._parse_status(entity.status)
        if status is not ProfileAnalysisStatus.COMPLETED:
            raise ProfileAnalysisNotReadyError(
                f"analysis '{analysis_id}' has status '{status.value}'"
            )
        if entity.result_payload is None:
            raise ProfileAnalysisExecutionError("completed analysis is missing its persisted result")
        completed = self._deserialize_completed(entity.result_payload)
        if completed.result.analysis_id != analysis_id:
            raise ProfileAnalysisExecutionError("persisted result analysis_id does not match its row")
        return completed

    async def _transition_async(
        self,
        analysis_id: str,
        current_status: ProfileAnalysisStatus,
        new_status: ProfileAnalysisStatus,
        *,
        result_payload: Mapping[str, object] | None = None,
    ) -> None:
        if new_status not in _ALLOWED_TRANSITIONS[current_status]:
            raise ProfileAnalysisTransitionError(
                f"invalid analysis transition: {current_status.value} -> {new_status.value}"
            )
        if new_status is ProfileAnalysisStatus.COMPLETED and result_payload is None:
            raise ProfileAnalysisTransitionError("completed transition requires an atomic result payload")
        if new_status is not ProfileAnalysisStatus.COMPLETED and result_payload is not None:
            raise ProfileAnalysisTransitionError("non-completed transition cannot persist a result")
        try:
            await self._profile_analysis_repository.transition_async(
                analysis_id,
                expected_status=current_status,
                new_status=new_status,
                result_payload=result_payload,
            )
        except RepositoryTransitionConflictError as exc:
            raise ProfileAnalysisTransitionError(str(exc)) from exc

    async def _mark_failed_async(self, analysis_id: str) -> None:
        try:
            await self._transition_async(
                analysis_id,
                ProfileAnalysisStatus.IN_PROGRESS,
                ProfileAnalysisStatus.FAILED,
            )
        except ProfileAnalysisTransitionError as exc:
            raise ProfileAnalysisExecutionError(
                "analysis failure state could not be persisted"
            ) from exc

    @staticmethod
    def _to_lifecycle(entity: ProfileAnalysisModel) -> ProfileAnalysisLifecycle:
        return ProfileAnalysisLifecycle(
            analysis_id=entity.id,
            status=ProfileAnalysisService._parse_status(entity.status),
            image_ids=tuple(entity.image_ids),
        )

    @staticmethod
    def _parse_status(value: str) -> ProfileAnalysisStatus:
        try:
            return ProfileAnalysisStatus(value)
        except ValueError as exc:
            raise ProfileAnalysisExecutionError("persisted analysis has an invalid status") from exc

    @staticmethod
    def _sanitize_for_persistence(
        completed: CompletedProfileAnalysis,
    ) -> CompletedProfileAnalysis:
        images: list[ImageAnalysisResult] = []
        for image in completed.result.images:
            sanitized_evidence: list[Evidence] = []
            for evidence in image.evidence:
                if evidence.type is not EvidenceType.OCR_TEXT:
                    sanitized_evidence.append(evidence)
                    continue
                metadata = dict(evidence.metadata)
                metadata.pop("raw_text", None)
                sanitized_evidence.append(
                    Evidence(
                        id=evidence.id,
                        image_id=evidence.image_id,
                        type=evidence.type,
                        label=evidence.label,
                        value=None,
                        confidence=evidence.confidence,
                        source=evidence.source,
                        metadata=metadata,
                    )
                )
            images.append(
                ImageAnalysisResult(
                    analysis_id=image.analysis_id,
                    image_id=image.image_id,
                    evidence=tuple(sanitized_evidence),
                )
            )
        sanitized_result = ProfileAnalysisResult(
            analysis_id=completed.result.analysis_id,
            status=completed.result.status,
            images=tuple(images),
            themes=completed.result.themes,
            insights=completed.result.insights,
        )
        return CompletedProfileAnalysis(
            result=sanitized_result,
            insight_policy_version=completed.insight_policy_version,
            summary=completed.summary,
            self_reported_claims=completed.self_reported_claims,
        )

    @staticmethod
    def _serialize_completed(completed: CompletedProfileAnalysis) -> dict[str, object]:
        return {
            "schema_version": _PERSISTED_RESULT_SCHEMA_V2,
            "result": json.loads(completed.result.to_json()),
            "insight_policy_version": completed.insight_policy_version,
            "summary": completed.summary,
            "self_reported_claims": [
                item.to_dict() for item in completed.self_reported_claims
            ],
        }

    @staticmethod
    def _deserialize_completed(payload: object) -> CompletedProfileAnalysis:
        if not isinstance(payload, Mapping):
            raise ProfileAnalysisExecutionError("persisted analysis result must be an object")
        schema_version = payload.get("schema_version")
        required_v1 = {
            "schema_version",
            "result",
            "insight_policy_version",
            "summary",
        }
        required_v2 = required_v1 | {"self_reported_claims"}
        try:
            if schema_version == _PERSISTED_RESULT_SCHEMA_V1:
                if set(payload) != required_v1:
                    raise ProfileAnalysisExecutionError(
                        "persisted analysis result has an invalid schema"
                    )
                claims: tuple[SelfReportedClaim, ...] = ()
            elif schema_version == _PERSISTED_RESULT_SCHEMA_V2:
                if set(payload) != required_v2:
                    raise ProfileAnalysisExecutionError(
                        "persisted analysis result has an invalid schema"
                    )
                claims_payload = payload["self_reported_claims"]
                if not isinstance(claims_payload, list):
                    raise ProfileAnalysisExecutionError(
                        "persisted self_reported_claims must be a list"
                    )
                claims = tuple(
                    SelfReportedClaim.from_dict(item)
                    for item in claims_payload
                )
            else:
                raise ProfileAnalysisExecutionError(
                    "persisted analysis result schema is unsupported"
                )
            result = ProfileAnalysisResult.from_dict(payload["result"])
            return CompletedProfileAnalysis(
                result=result,
                insight_policy_version=payload["insight_policy_version"],
                summary=payload["summary"],
                self_reported_claims=claims,
            )
        except (TypeError, ValueError) as exc:
            raise ProfileAnalysisExecutionError("persisted analysis result is invalid") from exc
