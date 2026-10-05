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


_PERSISTED_RESULT_SCHEMA = "phase9-analysis-result-v1"
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
    ):
        self._image_processing_service = image_processing_service
        self._ocr_evidence_service = ocr_evidence_service
        self._evidence_extraction_service = evidence_extraction_service
        self._evidence_aggregation_service = evidence_aggregation_service
        self._insight_generation_service = insight_generation_service
        self._profile_analysis_repository = profile_analysis_repository

    async def create_async(
        self,
        request: ProfileAnalysisRequest,
        images: Sequence[RawImageInput],
    ) -> CompletedProfileAnalysis:
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
                vision_result = await self._evidence_extraction_service.extract_async(image)
                image_evidence = tuple(
                    sorted(
                        (*ocr_result.evidence, *vision_result.evidence),
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
            completed = CompletedProfileAnalysis.from_insight_result(result, insight_result)
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
        )

    @staticmethod
    def _serialize_completed(completed: CompletedProfileAnalysis) -> dict[str, object]:
        return {
            "schema_version": _PERSISTED_RESULT_SCHEMA,
            "result": json.loads(completed.result.to_json()),
            "insight_policy_version": completed.insight_policy_version,
            "summary": completed.summary,
        }

    @staticmethod
    def _deserialize_completed(payload: object) -> CompletedProfileAnalysis:
        if not isinstance(payload, Mapping):
            raise ProfileAnalysisExecutionError("persisted analysis result must be an object")
        required = {"schema_version", "result", "insight_policy_version", "summary"}
        if set(payload) != required:
            raise ProfileAnalysisExecutionError("persisted analysis result has an invalid schema")
        if payload["schema_version"] != _PERSISTED_RESULT_SCHEMA:
            raise ProfileAnalysisExecutionError("persisted analysis result schema is unsupported")
        try:
            result = ProfileAnalysisResult.from_dict(payload["result"])
            return CompletedProfileAnalysis(
                result=result,
                insight_policy_version=payload["insight_policy_version"],
                summary=payload["summary"],
            )
        except (TypeError, ValueError) as exc:
            raise ProfileAnalysisExecutionError("persisted analysis result is invalid") from exc
