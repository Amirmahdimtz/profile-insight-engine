from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from uuid import uuid4

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
from src.core.services.profile_analysis.profile_analysis_service import ProfileAnalysisService
from src.infrastructure.context.sql_db.psql_dbcontext import PsqlDbContext
from src.infrastructure.repositories.profile_analysis.profile_analysis_repository import ProfileAnalysisRepository


_SCHEMA_VERSION = "phase9-api-persistence-benchmark-v1"
_TEST_DATABASE_ENV = "PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"


class _DatabaseConfig:
    def get_non_empty_string(self, key):
        if key != "database.url_env":
            raise KeyError(key)
        return _TEST_DATABASE_ENV


class _ControllerConfig:
    def get_positive_int(self, key):
        if key == "profile_analysis.max_images":
            return 50
        if key == "profile_analysis.max_image_size_mb":
            return 12
        raise KeyError(key)


class _ReadOnlyService:
    def __init__(self, completed):
        self._completed = completed

    async def get_status_async(self, analysis_id):
        return ProfileAnalysisLifecycle(
            analysis_id=analysis_id,
            status=ProfileAnalysisStatus.COMPLETED,
            image_ids=("img-1", "img-2"),
        )

    async def get_result_async(self, analysis_id):
        return self._completed

    async def create_async(self, request, images):
        return self._completed


def _completed_fixture(analysis_id: str) -> CompletedProfileAnalysis:
    evidence = (
        Evidence(
            id="ev-1",
            image_id="img-1",
            type=EvidenceType.TOPIC,
            label="football",
            value="football",
            confidence=0.9,
            source="vision",
        ),
        Evidence(
            id="ev-2",
            image_id="img-2",
            type=EvidenceType.TOPIC,
            label="football",
            value="football",
            confidence=0.8,
            source="vision",
        ),
    )
    insight = ProfileInsight(
        key="visible_interest_1",
        type=InsightType.VISIBLE_INTEREST,
        label="Recurring topic-related content: football",
        explanation="Supported by two evidence signals across two images.",
        confidence=0.85,
        evidence_count=2,
        image_coverage=1.0,
        supporting_evidence_ids=("ev-1", "ev-2"),
        supporting_image_ids=("img-1", "img-2"),
    )
    return CompletedProfileAnalysis(
        result=ProfileAnalysisResult(
            analysis_id=analysis_id,
            status=ProfileAnalysisStatus.COMPLETED,
            images=(
                ImageAnalysisResult(analysis_id, "img-1", (evidence[0],)),
                ImageAnalysisResult(analysis_id, "img-2", (evidence[1],)),
            ),
            insights=(insight,),
        ),
        insight_policy_version="phase8-candidate-v1",
        summary="Supported insights: Recurring topic-related content: football",
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _measure_api(completed: CompletedProfileAnalysis, iterations: int) -> dict[str, float]:
    controller = ProfileAnalysisController(_ReadOnlyService(completed), _ControllerConfig())
    app = FastAPI()
    app.include_router(controller.api(), prefix="/api/v1/profile_analysis")
    client = TestClient(app)
    status_latencies: list[float] = []
    result_latencies: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        response = client.get(f"/api/v1/profile_analysis/{completed.result.analysis_id}")
        elapsed = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError("status API benchmark request failed")
        status_latencies.append(elapsed)

        started = time.perf_counter()
        response = client.get(
            f"/api/v1/profile_analysis/{completed.result.analysis_id}/result"
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError("result API benchmark request failed")
        result_latencies.append(elapsed)
    return {
        "status_p50_ms": statistics.median(status_latencies),
        "status_p95_ms": _percentile(status_latencies, 0.95),
        "result_p50_ms": statistics.median(result_latencies),
        "result_p95_ms": _percentile(result_latencies, 0.95),
    }


async def _measure_database(
    completed: CompletedProfileAnalysis,
    iterations: int,
    concurrency: int,
) -> dict[str, object]:
    if not os.environ.get(_TEST_DATABASE_ENV):
        raise RuntimeError(
            f"set {_TEST_DATABASE_ENV} to a disposable PostgreSQL database with the Phase 9 migration applied"
        )
    db = PsqlDbContext(_DatabaseConfig())
    repository = ProfileAnalysisRepository(db)
    payload = ProfileAnalysisService._serialize_completed(completed)
    write_latencies: list[float] = []
    read_latencies: list[float] = []
    created_ids: list[str] = []
    try:
        for _ in range(iterations):
            analysis_id = f"phase9-bench-{uuid4()}"
            created_ids.append(analysis_id)
            await repository.create_pending_async(analysis_id, ("img-1", "img-2"))
            await repository.transition_async(
                analysis_id,
                expected_status=ProfileAnalysisStatus.PENDING,
                new_status=ProfileAnalysisStatus.IN_PROGRESS,
            )
            row_payload = dict(payload)
            row_payload["result"] = dict(payload["result"])
            row_payload["result"]["analysis_id"] = analysis_id
            for image in row_payload["result"]["images"]:
                image["analysis_id"] = analysis_id

            started = time.perf_counter()
            await repository.transition_async(
                analysis_id,
                expected_status=ProfileAnalysisStatus.IN_PROGRESS,
                new_status=ProfileAnalysisStatus.COMPLETED,
                result_payload=row_payload,
            )
            write_latencies.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            loaded = await repository.get_by_id_async(analysis_id)
            read_latencies.append((time.perf_counter() - started) * 1000.0)
            if loaded is None or loaded.result_payload is None:
                raise RuntimeError("database benchmark read-after-write failed")

        target = created_ids[-1]
        started = time.perf_counter()
        concurrent_reads = await asyncio.gather(
            *(repository.get_by_id_async(target) for _ in range(concurrency))
        )
        concurrent_elapsed = (time.perf_counter() - started) * 1000.0
        if not all(item is not None and item.status == "completed" for item in concurrent_reads):
            raise RuntimeError("concurrent read benchmark returned inconsistent state")

        return {
            "write_p50_ms": statistics.median(write_latencies),
            "write_p95_ms": _percentile(write_latencies, 0.95),
            "read_p50_ms": statistics.median(read_latencies),
            "read_p95_ms": _percentile(read_latencies, 0.95),
            "concurrent_read_count": concurrency,
            "concurrent_read_wall_ms": concurrent_elapsed,
        }
    finally:
        for analysis_id in created_ids:
            await repository.delete_async(analysis_id)
        await db.engine.dispose()


async def _run(args) -> dict[str, object]:
    fixture = _completed_fixture("phase9-benchmark-fixture")
    payload = ProfileAnalysisService._serialize_completed(fixture)
    graph_size = len(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    api_metrics = _measure_api(fixture, args.iterations)
    db_metrics = await _measure_database(fixture, args.iterations, args.concurrency)
    return {
        "schema_version": _SCHEMA_VERSION,
        "metric_scope": "phase9_contract_and_persistence_not_end_to_end_ml",
        "iterations": args.iterations,
        "analysis_graph_size_bytes": graph_size,
        "api": api_metrics,
        "database": db_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.concurrency <= 0:
        parser.error("iterations and concurrency must be positive")
    report = asyncio.run(_run(args))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
