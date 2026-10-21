# Requires Windows PowerShell 5.1+, Docker Desktop Linux containers, and the
# already-built profile-insight-engine:local image. Run from any directory.
# No public proxy, TLS, external port exposure, or production DB is involved.
[CmdletBinding()]
param(
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $composeArgs = @(
        "-p", "profile-insight-local",
        "--env-file", ".env.local",
        "-f", "compose.yaml",
        "-f", "compose.local.yaml"
    )

    function Invoke-Compose {
        param([string[]]$Arguments)
        & docker compose @composeArgs @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose command failed (exit code $LASTEXITCODE): $($Arguments -join ' ')"
        }
    }

    if ($Stop) {
        if (-not (Test-Path ".env.local")) {
            throw "Missing .env.local; this local Compose stack has not been initialized."
        }
        Invoke-Compose -Arguments @("stop")
        Write-Host "Local containers stopped. Database volume and .env.local are retained."
        return
    }

    $dockerOS = & docker info --format "{{.OSType}}"
    if ($LASTEXITCODE -ne 0 -or $dockerOS.Trim() -ne "linux") {
        throw "Start Docker Desktop with Linux containers before running this script."
    }

    & docker compose version
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose v2 is required."
    }

    & docker image inspect "profile-insight-engine:local" --format "{{.Id}}"
    if ($LASTEXITCODE -ne 0) {
        throw "Missing built image profile-insight-engine:local. Build it first with: docker build -t profile-insight-engine:local ."
    }

    if (-not (Test-Path ".env.local")) {
        $template = [System.IO.File]::ReadAllText((Join-Path $repoRoot ".env.example"))
        if (-not $template.Contains("REPLACE_WITH_ADMIN_64_HEX_CHAR_SECRET") -or -not $template.Contains("REPLACE_WITH_APP_64_HEX_CHAR_SECRET")) {
            throw "Unexpected .env.example secret placeholders; refusing to guess."
        }

        function New-SecretHex {
            $bytes = New-Object byte[] 32
            $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
            try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
            return [System.BitConverter]::ToString($bytes).Replace("-", "").ToLowerInvariant()
        }

        $config = $template.Replace("REPLACE_WITH_ADMIN_64_HEX_CHAR_SECRET", (New-SecretHex))
        $config = $config.Replace("REPLACE_WITH_APP_64_HEX_CHAR_SECRET", (New-SecretHex))
        $privateEnvPath = Join-Path $repoRoot ".env.local"
        [System.IO.File]::WriteAllText($privateEnvPath, $config, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Created ignored .env.local with separate cryptographically random DB passwords."
    }

    $currentEnv = [System.IO.File]::ReadAllText((Join-Path $repoRoot ".env.local"))
    if ($currentEnv.Contains("REPLACE_WITH_")) {
        throw ".env.local still has placeholder values."
    }

    Write-Host "Starting isolated PostgreSQL, Alembic migration and API (no Nginx proxy)."
    try {
        Invoke-Compose -Arguments @("up", "-d", "--no-build", "--wait", "--wait-timeout", "300", "db", "migrate", "app")
    }
    catch {
        & docker compose @composeArgs ps -a
        & docker compose @composeArgs logs --tail 80 db migrate app
        throw
    }

    Write-Host "Checking database schema drift."
    Invoke-Compose -Arguments @("run", "--rm", "--no-deps", "app", "python", "-m", "alembic", "-c", "src/infrastructure/alembic.ini", "check")

    Write-Host "Checking loopback HTTP API."
    $openApi = Invoke-RestMethod -Uri "http://127.0.0.1:5000/openapi.json" -TimeoutSec 30
    if ($openApi.info.title -ne "Profile Insight Engine") {
        throw "Unexpected OpenAPI title; the expected application is not running."
    }

    Write-Host "Running HTTP POST, status and persisted-result smoke test with synthetic images."
    $smokePython = @'
import io
import json
import uuid

import httpx
from PIL import Image, ImageDraw, ImageFont

def synthetic_png(y_position):
    image = Image.new("RGB", (960, 260), "white")
    drawing = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=58)
    drawing.text((25, y_position), "HELLO PROFILE TEST 123", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

analysis_id = "docker-local-smoke-" + uuid.uuid4().hex
request = {"analysis_id": analysis_id, "image_ids": ["smoke-image-1", "smoke-image-2"]}
files = [
    ("images", ("smoke1.png", synthetic_png(35), "image/png")),
    ("images", ("smoke2.png", synthetic_png(95), "image/png")),
]
base_url = "http://127.0.0.1:8000"
with httpx.Client(base_url=base_url, timeout=240.0) as client:
    create = client.post(
        "/api/v1/profile_analysis/",
        data={"request_json": json.dumps(request)},
        files=files,
    )
    if create.status_code != 201:
        raise RuntimeError(f"POST failed: HTTP {create.status_code}")
    created = create.json()
    if created.get("analysis_id") != analysis_id or created.get("status") != "completed":
        raise RuntimeError("POST response has incorrect analysis_id or incomplete lifecycle state")
    if len(created.get("images", [])) != 2:
        raise RuntimeError("POST response did not include both uploaded images")

    status = client.get(f"/api/v1/profile_analysis/{analysis_id}")
    if status.status_code != 200:
        raise RuntimeError(f"GET status failed: HTTP {status.status_code}")
    state = status.json()
    if state.get("analysis_id") != analysis_id or state.get("status") != "completed":
        raise RuntimeError("Persisted status does not match completed analysis")

    result = client.get(f"/api/v1/profile_analysis/{analysis_id}/result")
    if result.status_code != 200:
        raise RuntimeError(f"GET result failed: HTTP {result.status_code}")
    body = result.json()
    if body.get("analysis_id") != analysis_id or body.get("status") != "completed":
        raise RuntimeError("Persisted result does not match completed analysis")
    if len(body.get("images", [])) != 2:
        raise RuntimeError("Persisted result did not include both images")

    print("PASS: POST returned HTTP 201 and completed result")
    print("PASS: GET status returned HTTP 200 and completed lifecycle")
    print("PASS: GET result returned HTTP 200 with two images")
    print(f"Smoke insights count: {len(body.get('insights', []))} (zero is valid)")
'@

    $smokePython | & docker compose @composeArgs exec -T app python -
    if ($LASTEXITCODE -ne 0) {
        throw "Containerized HTTP smoke test failed (exit code $LASTEXITCODE)."
    }

    Write-Host "PASS: Docker Desktop local API + PostgreSQL + migration + OCR analysis + persistence smoke test."
    Write-Host "API docs: http://127.0.0.1:5000/docs"
    Write-Host "To stop later: powershell -ExecutionPolicy Bypass -File scripts/verify_docker_local.ps1 -Stop"
}
finally {
    Pop-Location
}
