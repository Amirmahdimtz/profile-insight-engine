from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from src.application.profile_analysis.dtos.profile_analysis_dto import (
    CreateProfileAnalysisDto,
    ProfileAnalysisResultDto,
    ProfileAnalysisStatusDto,
)
from src.core.profile_analysis.image_contracts import ImageValidationError, RawImageInput
from src.core.services.profile_analysis.profile_analysis_service import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisExecutionError,
    ProfileAnalysisInputError,
    ProfileAnalysisNotFoundError,
    ProfileAnalysisNotReadyError,
    ProfileAnalysisService,
    ProfileAnalysisTransitionError,
)
from src.infrastructure.di.inject import inject
from src.infrastructure.utils.config_reader import ConfigReader


_UPLOAD_CHUNK_SIZE = 1024 * 1024


@inject
class ProfileAnalysisController:
    def __init__(
        self,
        profile_analysis_service: ProfileAnalysisService,
        config_reader: ConfigReader,
    ):
        self._profile_analysis_service = profile_analysis_service
        self._max_images = config_reader.get_positive_int("profile_analysis.max_images")
        self._max_upload_bytes = (
            config_reader.get_positive_int("profile_analysis.max_image_size_mb") * 1024 * 1024
        )

    def api(self) -> APIRouter:
        router = APIRouter(prefix="", tags=["Profile Analysis"])

        @router.post(
            "/",
            response_model=ProfileAnalysisResultDto,
            status_code=status.HTTP_201_CREATED,
            summary="Create and run a profile image analysis",
        )
        async def create_analysis(
            request_json: str = Form(...),
            images: list[UploadFile] = File(...),
        ) -> ProfileAnalysisResultDto:
            try:
                payload = CreateProfileAnalysisDto.model_validate_json(request_json)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=exc.errors(include_input=False, include_context=False),
                ) from exc

            if len(payload.image_ids) > self._max_images:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="image count exceeds configured max_images",
                )
            if len(images) != len(payload.image_ids):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="uploaded image count must match image_ids",
                )

            raw_images: list[RawImageInput] = []
            for image_id, upload in zip(payload.image_ids, images):
                if not upload.filename or not upload.content_type:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="each uploaded image requires filename and content type",
                    )
                content = await self._read_upload_async(upload)
                try:
                    raw_images.append(
                        RawImageInput(
                            image_id=image_id,
                            filename=upload.filename,
                            declared_mime_type=upload.content_type,
                            content=content,
                        )
                    )
                except ImageValidationError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=str(exc),
                    ) from exc

            try:
                result = await self._profile_analysis_service.create_async(
                    payload.to_core(),
                    tuple(raw_images),
                )
            except ProfileAnalysisAlreadyExistsError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except ProfileAnalysisInputError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc
            except ProfileAnalysisTransitionError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except ProfileAnalysisExecutionError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="profile analysis execution failed",
                ) from exc
            return ProfileAnalysisResultDto.from_core(result)

        @router.get(
            "/{analysis_id}",
            response_model=ProfileAnalysisStatusDto,
            summary="Get profile analysis status",
        )
        async def get_analysis_status(analysis_id: str) -> ProfileAnalysisStatusDto:
            try:
                result = await self._profile_analysis_service.get_status_async(analysis_id)
            except ProfileAnalysisNotFoundError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except ProfileAnalysisExecutionError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="profile analysis status could not be loaded",
                ) from exc
            return ProfileAnalysisStatusDto.from_core(result)

        @router.get(
            "/{analysis_id}/result",
            response_model=ProfileAnalysisResultDto,
            summary="Get completed profile analysis result",
        )
        async def get_analysis_result(analysis_id: str) -> ProfileAnalysisResultDto:
            try:
                result = await self._profile_analysis_service.get_result_async(analysis_id)
            except ProfileAnalysisNotFoundError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except ProfileAnalysisNotReadyError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except ProfileAnalysisExecutionError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="persisted profile analysis result is invalid",
                ) from exc
            return ProfileAnalysisResultDto.from_core(result)

        return router

    async def _read_upload_async(self, upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > self._max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="uploaded image exceeds configured file-size limit",
                )
            chunks.append(chunk)
        return b"".join(chunks)
