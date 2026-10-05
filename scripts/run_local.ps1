param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$env:env_type = "production"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $env:PROFILE_INSIGHT_DATABASE_URL) {
    throw "PROFILE_INSIGHT_DATABASE_URL must be set to a postgresql+asyncpg:// URL before startup."
}

python -m alembic -c src/infrastructure/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { throw "Alembic upgrade failed." }
python -m alembic -c src/infrastructure/alembic.ini check
if ($LASTEXITCODE -ne 0) { throw "Alembic metadata check failed." }

$Tesseract = Get-Command tesseract -ErrorAction SilentlyContinue
if (-not $Tesseract) { throw "tesseract is not available on PATH." }
$tessLanguages = & $Tesseract.Source --list-langs 2>&1
foreach ($language in @("fas", "eng")) {
    if ($tessLanguages -notcontains $language) {
        throw "Tesseract language '$language' is unavailable."
    }
}

$VisionExecutable = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_non_empty_string('vision.runtime_executable'))"
$VisionModel = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_non_empty_string('vision.model_id'))"
$VisionBaseUrl = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_non_empty_string('vision.base_url'))"
$VisionContext = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_positive_int('vision.runtime_context_size'))"
$VisionParallel = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_positive_int('vision.runtime_parallel'))"
$VisionStartupTimeout = python -c "from src.infrastructure.utils.config_reader import ConfigReader; print(ConfigReader().get_positive_int('vision.startup_timeout_seconds'))"

$VisionUri = [Uri]$VisionBaseUrl
if ($VisionUri.Host -notin @("127.0.0.1", "localhost")) {
    throw "Local deployment requires the configured vision runtime to use loopback."
}
$VisionCommand = Get-Command $VisionExecutable -ErrorAction SilentlyContinue
if (-not $VisionCommand) { throw "Configured llama.cpp runtime executable was not found: $VisionExecutable" }

$VisionProcess = $null
try {
    try {
        $Health = Invoke-RestMethod -Uri "$($VisionBaseUrl.TrimEnd('/'))/health" -TimeoutSec 2
    } catch {
        $Health = $null
    }

    if (-not $Health -or $Health.status -ne "ok") {
        $VisionArgs = @(
            "-hf", $VisionModel,
            "--host", $VisionUri.Host,
            "--port", [string]$VisionUri.Port,
            "--ctx-size", [string]$VisionContext,
            "--parallel", [string]$VisionParallel,
            "--offline",
            "--log-disable"
        )
        $VisionProcess = Start-Process -FilePath $VisionCommand.Source -ArgumentList $VisionArgs -PassThru -NoNewWindow
        $Deadline = [DateTime]::UtcNow.AddSeconds([int]$VisionStartupTimeout)
        do {
            if ($VisionProcess.HasExited) { throw "llama.cpp runtime exited before becoming healthy." }
            Start-Sleep -Milliseconds 500
            try {
                $Health = Invoke-RestMethod -Uri "$($VisionBaseUrl.TrimEnd('/'))/health" -TimeoutSec 2
            } catch {
                $Health = $null
            }
        } while ((!$Health -or $Health.status -ne "ok") -and [DateTime]::UtcNow -lt $Deadline)
        if (-not $Health -or $Health.status -ne "ok") { throw "llama.cpp runtime did not become healthy." }
    }

    python -m uvicorn src.host.app:app --host $HostAddress --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Uvicorn exited with a failure code." }
}
finally {
    if ($VisionProcess -and -not $VisionProcess.HasExited) {
        Stop-Process -Id $VisionProcess.Id -Force
    }
}
