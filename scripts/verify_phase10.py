from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_NAME = f"profile-insight-phase10-verify-{os.getpid()}"
DATABASE_NAME = "profile_insight_phase10_test"
DATABASE_USER = "postgres"
POSTGRES_IMAGE = "postgres:17-alpine"
HARDWARE_OUTPUT = ROOT / "phase10_hardware_profile.json"
BENCHMARK_OUTPUT = ROOT / "phase10_final_benchmark.json"


class VerificationError(RuntimeError):
    pass


def _run(label: str, args: list[str], *, env: dict[str, str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
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
        raise VerificationError(f"{label} failed with exit code {completed.returncode}")
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
        completed = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "pg_isready", "-U", DATABASE_USER, "-d", DATABASE_NAME],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            print("PostgreSQL is ready.", flush=True)
            return
        time.sleep(1.0)
    _docker_logs(env)
    raise VerificationError("PostgreSQL did not become ready within 30 seconds")


def _verify_repository_state(env: dict[str, str]) -> None:
    branch = _run("Git branch", ["git", "branch", "--show-current"], env=env, capture_output=True).stdout.strip()
    if branch != "main":
        raise VerificationError(f"expected branch 'main', got {branch!r}")
    tracked = _run(
        "Tracked repository status",
        ["git", "status", "--short", "--untracked-files=no"],
        env=env,
        capture_output=True,
    ).stdout.strip()
    if tracked:
        raise VerificationError("tracked repository changes are present; commit/stash them before verification")
    _run("Git HEAD", ["git", "rev-parse", "HEAD"], env=env, capture_output=True)


def _verify_phase10_report() -> None:
    try:
        report = json.loads(BENCHMARK_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError("Phase 10 benchmark report is missing or invalid") from exc
    if report.get("schema_version") != "phase10-final-benchmark-v1":
        raise VerificationError("unexpected Phase 10 benchmark schema_version")
    if report.get("acceptance_ready") is not False:
        raise VerificationError("implementation-only verification must not claim Phase 10 acceptance")
    print("Phase 10 report correctly remains awaiting real benchmark evidence.", flush=True)


def main() -> int:
    env = dict(os.environ)
    port = _free_loopback_port()
    database_url = f"postgresql+asyncpg://{DATABASE_USER}@127.0.0.1:{port}/{DATABASE_NAME}"
    env["PROFILE_INSIGHT_DATABASE_URL"] = database_url
    env["PROFILE_INSIGHT_PHASE9_TEST_DATABASE_URL"] = database_url
    container_started = False
    try:
        _verify_repository_state(env)
        _run("Docker availability", ["docker", "version", "--format", "{{.Server.Version}}"], env=env, capture_output=True)
        _run(
            "Start disposable PostgreSQL",
            [
                "docker", "run", "--rm", "--name", CONTAINER_NAME,
                "--env", f"POSTGRES_USER={DATABASE_USER}",
                "--env", "POSTGRES_HOST_AUTH_METHOD=trust",
                "--env", f"POSTGRES_DB={DATABASE_NAME}",
                "--publish", f"127.0.0.1:{port}:5432",
                "--detach", POSTGRES_IMAGE,
            ],
            env=env,
            capture_output=True,
        )
        container_started = True
        _wait_for_postgres(env)
        _run(
            "Compile/import check",
            [sys.executable, "-m", "compileall", "-q", "src", "evaluation", "tests", "scripts"],
            env=env,
        )
        _run(
            "Focused Phase 10 tests",
            [
                sys.executable, "-m", "unittest",
                "tests.test_phase10_hardening",
                "tests.test_phase10_benchmark",
                "tests.test_phase10_local_verification",
                "tests.test_phase10_documentation",
                "tests.test_phase9_architecture",
                "tests.test_phase9_di_discovery",
                "-v",
            ],
            env=env,
        )
        _run(
            "Alembic upgrade",
            [sys.executable, "-m", "alembic", "-c", "src/infrastructure/alembic.ini", "upgrade", "head"],
            env=env,
        )
        _run(
            "Alembic metadata check",
            [sys.executable, "-m", "alembic", "-c", "src/infrastructure/alembic.ini", "check"],
            env=env,
        )
        _run(
            "PostgreSQL recovery regression",
            [
                sys.executable, "-m", "unittest",
                "tests.test_profile_analysis_repository_integration",
                "tests.test_phase10_hardening.Phase10RecoveryIntegrationTests",
                "-v",
            ],
            env=env,
        )
        if HARDWARE_OUTPUT.exists():
            HARDWARE_OUTPUT.unlink()
        _run(
            "Hardware/runtime evidence collection",
            [sys.executable, "scripts/collect_phase10_hardware.py", "--output", str(HARDWARE_OUTPUT)],
            env=env,
        )
        if BENCHMARK_OUTPUT.exists():
            BENCHMARK_OUTPUT.unlink()
        _run(
            "Phase 10 acceptance-report contract",
            [
                sys.executable, "-m", "evaluation.phase10_benchmark",
                "--hardware-report", str(HARDWARE_OUTPUT),
                "--output", str(BENCHMARK_OUTPUT),
            ],
            env=env,
        )
        _verify_phase10_report()
        _run(
            "Full repository regression",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"],
            env=env,
        )
        _verify_repository_state(env)
        print("\nPHASE 10 IMPLEMENTATION VERIFICATION PASSED", flush=True)
        print(f"Hardware profile: {HARDWARE_OUTPUT}", flush=True)
        print(f"Acceptance report scaffold: {BENCHMARK_OUTPUT}", flush=True)
        print("Phase 10 remains IMPLEMENTED_AWAITING_LOCAL_VERIFICATION until real model/end-to-end evidence is supplied.", flush=True)
        return 0
    except VerificationError as exc:
        print(f"\nPHASE 10 IMPLEMENTATION VERIFICATION FAILED: {exc}", file=sys.stderr, flush=True)
        if container_started:
            print("\n=== PostgreSQL container logs ===", file=sys.stderr, flush=True)
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
