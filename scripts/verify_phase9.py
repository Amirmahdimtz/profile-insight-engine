from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_NAME = f"profile-insight-phase9-verify-{os.getpid()}"
DATABASE_NAME = "profile_insight_phase9_test"
DATABASE_USER = "postgres"
POSTGRES_IMAGE = "postgres:17-alpine"
BENCHMARK_OUTPUT = ROOT / "phase9_api_persistence_benchmark.json"


class VerificationError(RuntimeError):
    pass


def _run(
    label: str,
    args: list[str],
    *,
    env: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {label} ===", flush=True)
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=capture_output,
            check=False,
        )
    except OSError as exc:
        raise VerificationError(f"{label} could not start: {exc}") from exc
    if capture_output:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise VerificationError(
            f"{label} failed with exit code {completed.returncode}"
        )
    return completed


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _docker_logs(env: dict[str, str]) -> None:
    try:
        subprocess.run(
            ["docker", "logs", CONTAINER_NAME],
            cwd=ROOT,
            env=env,
            text=True,
            check=False,
        )
    except OSError:
        return


def _wait_for_postgres(env: dict[str, str]) -> None:
    print("\n=== PostgreSQL readiness ===", flush=True)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "exec",
                    CONTAINER_NAME,
                    "pg_isready",
                    "-U",
                    DATABASE_USER,
                    "-d",
                    DATABASE_NAME,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError as exc:
            raise VerificationError(
                f"PostgreSQL readiness check could not start: {exc}"
            ) from exc
        if completed.returncode == 0:
            print("PostgreSQL is ready.", flush=True)
            return
        time.sleep(1.0)
    _docker_logs(env)
    raise VerificationError("PostgreSQL did not become ready within 30 seconds")


def _verify_repository_state(env: dict[str, str]) -> None:
    branch = _run(
        "Git branch",
        ["git", "branch", "--show-current"],
        env=env,
        capture_output=True,
    ).stdout.strip()
    if branch != "main":
        raise VerificationError(f"expected branch 'main', got {branch!r}")

    tracked_status = _run(
        "Tracked repository status",
        ["git", "status", "--short", "--untracked-files=no"],
        env=env,
        capture_output=True,
    ).stdout.strip()
    if tracked_status:
        raise VerificationError(
            "tracked repository changes are present; commit/stash them before verification"
        )

    _run(
        "Git HEAD",
        ["git", "rev-parse", "HEAD"],
        env=env,
        capture_output=True,
    )


def _verify_benchmark_report() -> None:
    try:
        report = json.loads(BENCHMARK_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError("Phase 9 benchmark report is missing or invalid") from exc
    if report.get("schema_version") != "phase9-api-persistence-benchmark-v1":
        raise VerificationError("unexpected Phase 9 benchmark schema_version")
    if (
        report.get("metric_scope")
        != "phase9_contract_and_persistence_not_end_to_end_ml"
    ):
        raise VerificationError("unexpected Phase 9 benchmark metric_scope")
    print(
        "Benchmark contract:",
        report["schema_version"],
        report["metric_scope"],
        flush=True,
    )


def main() -> int:
    env = dict(os.environ)
    port = _free_loopback_port()
    database_url = (
        f"postgresql+asyncpg://{DATABASE_USER}@127.0.0.1:{port}/{DATABASE_NAME}"
    )
    env["PROFILE_INSIGHT_DATABASE_URL"] = database_url
    env["PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"] = database_url

    container_started = False
    try:
        _verify_repository_state(env)

        _run(
            "Docker availability",
            ["docker", "version", "--format", "{{.Server.Version}}"],
            env=env,
            capture_output=True,
        )

        _run(
            "Start disposable PostgreSQL",
            [
                "docker",
                "run",
                "--rm",
                "--name",
                CONTAINER_NAME,
                "--env",
                f"POSTGRES_USER={DATABASE_USER}",
                "--env",
                "POSTGRES_HOST_AUTH_METHOD=trust",
                "--env",
                f"POSTGRES_DB={DATABASE_NAME}",
                "--publish",
                f"127.0.0.1:{port}:5432",
                "--detach",
                POSTGRES_IMAGE,
            ],
            env=env,
            capture_output=True,
        )
        container_started = True
        _wait_for_postgres(env)

        _run(
            "Database host-connection preflight",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "current",
            ],
            env=env,
        )

        _run(
            "Compile/import check",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "evaluation",
                "tests",
                "scripts",
            ],
            env=env,
        )

        _run(
            "Focused Phase 9 tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_phase9_contracts",
                "tests.test_profile_analysis_service",
                "tests.test_profile_analysis_controller",
                "tests.test_phase9_architecture",
                "tests.test_phase9_di_discovery",
                "tests.test_phase9_documentation",
                "tests.test_phase9_local_verification",
                "-v",
            ],
            env=env,
        )

        _run(
            "Alembic upgrade",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "upgrade",
                "head",
            ],
            env=env,
        )
        _run(
            "Alembic current",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "current",
            ],
            env=env,
        )
        _run(
            "Alembic metadata check",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "check",
            ],
            env=env,
        )

        _run(
            "PostgreSQL integration tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_profile_analysis_repository_integration",
                "-v",
            ],
            env=env,
        )

        _run(
            "Alembic downgrade",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "downgrade",
                "base",
            ],
            env=env,
        )
        _run(
            "Alembic re-upgrade",
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "src/infrastructure/alembic.ini",
                "upgrade",
                "head",
            ],
            env=env,
        )

        route_script = (
            "from src.infrastructure.di.bootstrap import bootstrap_di; "
            "bootstrap_di(); "
            "from src.application.web import WebService; "
            "from src.infrastructure.di.inject import resolve; "
            "app=resolve(WebService).create_app(); "
            "spec=app.openapi(); "
            "methods={'get','post','put','patch','delete'}; "
            "routes={(method.upper(), path) for path, operations in "
            "spec['paths'].items() for method in operations if method in methods}; "
            "expected={"
            "('POST','/api/v1/profile_analysis/'),"
            "('GET','/api/v1/profile_analysis/{analysis_id}'),"
            "('GET','/api/v1/profile_analysis/{analysis_id}/result')}; "
            "missing=expected-routes; "
            "assert not missing, f'missing routes: {sorted(missing)}'; "
            "assert not any(method=='DELETE' and path.startswith('/api/v1/profile_analysis') "
            "for method, path in routes), 'unexpected DELETE route'; "
            "print(sorted(item for item in routes if item[1].startswith('/api/')))"
        )
        _run(
            "DI/discovery route smoke check",
            [sys.executable, "-c", route_script],
            env=env,
        )

        if BENCHMARK_OUTPUT.exists():
            BENCHMARK_OUTPUT.unlink()
        _run(
            "Phase 9 API/persistence benchmark",
            [
                sys.executable,
                "-m",
                "evaluation.phase9_api_persistence_benchmark",
                "--iterations",
                "20",
                "--concurrency",
                "8",
                "--output",
                str(BENCHMARK_OUTPUT),
            ],
            env=env,
        )
        _verify_benchmark_report()

        _run(
            "Full repository regression",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
                "-q",
            ],
            env=env,
        )

        _verify_repository_state(env)
        print("\nPHASE 9 LOCAL VERIFICATION PASSED", flush=True)
        print(f"Benchmark: {BENCHMARK_OUTPUT}", flush=True)
        return 0
    except VerificationError as exc:
        print(
            f"\nPHASE 9 LOCAL VERIFICATION FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if container_started:
            print(
                "\n=== PostgreSQL container logs ===",
                file=sys.stderr,
                flush=True,
            )
            _docker_logs(env)
        return 1
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER_NAME],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
