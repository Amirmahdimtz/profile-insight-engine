from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "phase10-hardware-profile-v1"


def _run_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _total_ram_mb() -> float | None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullTotalPhys / (1024.0 * 1024.0)
        except (OSError, AttributeError, ValueError):
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024.0 * 1024.0)
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total_mb = float(parts[2])
        except ValueError:
            total_mb = None
        rows.append({"name": parts[0], "driver_version": parts[1], "memory_total_mb": total_mb})
    return rows


def collect(llama_server_executable: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "processor": platform.processor() or None,
        },
        "memory": {"total_ram_mb": _total_ram_mb()},
        "gpu": {"nvidia": _nvidia_gpus()},
        "runtime_versions": {
            "tesseract": _run_version(["tesseract", "--version"]),
            "llama_server": _run_version([llama_server_executable, "--version"]),
            "docker": _run_version(["docker", "--version"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-server-executable", default="llama-server")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.llama_server_executable)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
