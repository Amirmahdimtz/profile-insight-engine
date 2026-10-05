import json
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.profile_analysis.profile_analysis_controller import ProfileAnalysisController
from src.core.profile_analysis.contracts import (
    Evidence,
    EvidenceType,
    ImageAnalysisResult,
    InsightType,
    ProfileAnalysisResult,
    ProfileAnalysisStatus,
    ProfileInsight,
)
from src.core.profile_analysis.lifecycle_contracts import CompletedProfileAnalysis, ProfileAnalysisLifecycle
from src.core.services.profile_analysis.profile_analysis_service import (
    ProfileAnalysisAlreadyExistsError,
    ProfileAnalysisNotFoundError,
    ProfileAnalysisNotReadyError,
)


class _Config:
    def get_positive_int(self, key):
        if key == "profile_analysis.max_images":
            return 2
        if key == "profile_analysis.max_image_size_mb":
            return 1
        raise KeyError(key)


class _Service:
    def __init__(self):
        self.create_error = None
        self.status_error = None
        self.result_error = None
        self.received = None

    async def create_async(self, request, images):
        if self.create_error:
            raise self.create_error
        self.received = (request, images)
        return _completed()

    async def get_status_async(self, analysis_id):
        if self.status_error:
            raise self.status_error
        return ProfileAnalysisLifecycle(
            analysis_id=analysis_id,
            status=ProfileAnalysisStatus.IN_PROGRESS,
            image_ids=("img-1",),
        )

    async def get_result_async(self, analysis_id):
        if self.result_error:
            raise self.result_error
        return _completed()


def _completed():
    evidence = Evidence(
        id="ev-1",
        image_id="img-1",
        type=EvidenceType.TOPIC,
        label="football",
        value="football",
        confidence=0.9,
        source="vision",
    )
    insight = ProfileInsight(
        key="visible_interest_1",
        type=InsightType.VISIBLE_INTEREST,
        label="Recurring topic-related content: football",
        explanation="Supported by one evidence signal across one image.",
        confidence=0.9,
        evidence_count=1,
        image_coverage=1.0,
        supporting_evidence_ids=("ev-1",),
        supporting_image_ids=("img-1",),
    )
    return CompletedProfileAnalysis(
        result=ProfileAnalysisResult(
            analysis_id="analysis-1",
            status=ProfileAnalysisStatus.COMPLETED,
            images=(
                ImageAnalysisResult(
                    analysis_id="analysis-1",
                    image_id="img-1",
                    evidence=(evidence,),
                ),
            ),
            insights=(insight,),
        ),
        insight_policy_version="phase8-candidate-v1",
        summary="Supported insights: Recurring topic-related content: football",
    )


class ProfileAnalysisControllerTests(unittest.TestCase):
    def setUp(self):
        self.service = _Service()
        controller = ProfileAnalysisController(self.service, _Config())
        app = FastAPI()
        app.include_router(controller.api(), prefix="/api/v1/profile_analysis")
        self.client = TestClient(app)

    def test_post_contract_maps_multipart_to_core_request(self):
        response = self.client.post(
            "/api/v1/profile_analysis/",
            data={
                "request_json": json.dumps(
                    {"analysis_id": "analysis-1", "image_ids": ["img-1"]}
                )
            },
            files={"images": ("image.png", b"not-decoded-by-fake-service", "image/png")},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["analysis_id"], "analysis-1")
        request, images = self.service.received
        self.assertEqual(request.image_ids, ("img-1",))
        self.assertEqual(images[0].declared_mime_type, "image/png")

    def test_dto_validation_and_image_count_errors_map_to_422(self):
        duplicate = self.client.post(
            "/api/v1/profile_analysis/",
            data={
                "request_json": json.dumps(
                    {"analysis_id": "analysis-1", "image_ids": ["img-1", "img-1"]}
                )
            },
            files=[("images", ("image.png", b"x", "image/png"))],
        )
        self.assertEqual(duplicate.status_code, 422)

        mismatch = self.client.post(
            "/api/v1/profile_analysis/",
            data={
                "request_json": json.dumps(
                    {"analysis_id": "analysis-1", "image_ids": ["img-1", "img-2"]}
                )
            },
            files=[("images", ("image.png", b"x", "image/png"))],
        )
        self.assertEqual(mismatch.status_code, 422)

    def test_duplicate_not_found_and_not_ready_mapping(self):
        self.service.create_error = ProfileAnalysisAlreadyExistsError("exists")
        response = self.client.post(
            "/api/v1/profile_analysis/",
            data={"request_json": json.dumps({"analysis_id": "analysis-1", "image_ids": ["img-1"]})},
            files={"images": ("image.png", b"x", "image/png")},
        )
        self.assertEqual(response.status_code, 409)

        self.service.status_error = ProfileAnalysisNotFoundError("missing")
        self.assertEqual(
            self.client.get("/api/v1/profile_analysis/missing").status_code,
            404,
        )

        self.service.result_error = ProfileAnalysisNotReadyError("pending")
        self.assertEqual(
            self.client.get("/api/v1/profile_analysis/analysis-1/result").status_code,
            409,
        )

    def test_upload_limit_maps_to_413(self):
        response = self.client.post(
            "/api/v1/profile_analysis/",
            data={"request_json": json.dumps({"analysis_id": "analysis-1", "image_ids": ["img-1"]})},
            files={"images": ("image.png", b"x" * (1024 * 1024 + 1), "image/png")},
        )
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
