# Windows Docker Desktop local verification and DevOps handoff

This is a **loopback-only local acceptance smoke test** for the existing
`profile-insight-engine:local` image. It reuses the Linux-targeted application
image, PostgreSQL, Alembic and the selected `degraded_ocr_only` runtime, but
**does not start public Nginx, require a domain or create TLS certificates**.
It does not modify `compose.yaml` or the public deployment edge.

## Prerequisites

- Windows PowerShell 5.1 or later;
- Docker Desktop running in **Linux container** mode, with Docker Compose v2;
- the previously built `profile-insight-engine:local` image (must include
  `fas`, `eng`, `osd` under `/usr/local/share/tessdata_best`);
- enough available Docker storage and RAM for PostgreSQL and the app;
- localhost TCP port 5000 unused.

Run from repository root (PowerShell):

```powershell
git switch main
git pull --ff-only
powershell -ExecutionPolicy Bypass -File .\scripts\verify_docker_local.ps1
```

The script generates two distinct random 64-character hex database passwords
locally on first run and writes them to **`.env.local`**, which is already
ignored by `.gitignore`. It reuses that file on subsequent runs, so the
database credentials stay aligned with the retained PostgreSQL volume. Do not
publish this file, print its contents, or commit it. The script never uses the
user's earlier example password.

The Compose invocation uses the isolated project name
`profile-insight-local`, the production `compose.yaml` plus
`compose.local.yaml`, and starts **only** `db`, `migrate`, and `app`.
The local override adds a loopback-only mapping
`127.0.0.1:5000:8000` and a local frontend network attachment to the app.
The database remains private. The public proxy service is not launched.
Never use this override for a public-facing deployment: it bypasses the
production TLS/Basic Auth edge.

The script performs all of these runtime checks:

1. Linux-mode Docker Engine and an existing application image;
2. PostgreSQL readiness, successful Alembic migration, and healthy API;
3. Alembic metadata drift check against the started database;
4. GET `http://127.0.0.1:5000/openapi.json` and expected application title;
5. HTTP POST of **two generated, non-sensitive PNG images** through the real
   profile-analysis API, expecting HTTP 201 and completed lifecycle;
6. GET of the persisted status and completed result by fresh unique
   `analysis_id`, expecting HTTP 200 and two returned images.

The smoke uses fresh unique IDs because the service does not expose DELETE
and analysis IDs may not be reused. **Zero insights is valid** for synthetic
images and the selected OCR-only evidence policy; this is not a product
accuracy benchmark.

A successful run ends with:

```text
PASS: POST returned HTTP 201 and completed result
PASS: GET status returned HTTP 200 and completed lifecycle
PASS: GET result returned HTTP 200 with two images
PASS: Docker Desktop local API + PostgreSQL + migration + OCR analysis + persistence smoke test.
```

The local API docs are then at `http://127.0.0.1:5000/docs`.
Do not expose port 5000 beyond localhost because this local test has **no
authentication or TLS**.

## Stop and diagnostics

Stop the local containers while retaining the database:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_docker_local.ps1 -Stop
```

On failure, inspect service state and recent logs (do not paste secret files or
sensitive payloads):

```powershell
docker compose -p profile-insight-local --env-file .env.local -f compose.yaml -f compose.local.yaml ps -a
docker compose -p profile-insight-local --env-file .env.local -f compose.yaml -f compose.local.yaml logs --tail 100 db migrate app
```

Do not use `docker compose down -v` to troubleshoot; it would delete the
PostgreSQL test volume. Do not regenerate `.env.local` while keeping the
existing volume, because PostgreSQL role credentials are created during first
database initialization. Existing local evidence and benchmark files must not
be deleted for convenience.

## Handoff to Linux DevOps

Pass the reviewed source commit, Dockerfile, `compose.yaml`,
`docs/docker_compose_public_deployment.md`, the deployment tests, and the
redacted Windows smoke results. **Do not pass `.env.local`**, local DB contents
or any local test images.

The DevOps owner is responsible for the actual Linux-host release:

- approved DNS hostname, real trusted TLS certificate and renewal, server
  firewall/security group, Nginx Basic Auth or stronger identity boundary;
- fresh production credentials, image digest pinning, least-privilege database
  role and authorized backup/restore strategy;
- `deploy/preflight.sh`, Linux-host build, migrations, healthchecks, external
  HTTPS auth/redirect checks, approved load/abuse limits and log/privacy policy;
- a **new real selected-profile API/E2E benchmark on the exact release
  revision** and hardware.

Passing this Windows smoke alone does **not** establish production readiness,
Vision quality, insight precision, Linux-host behavior or TLS security. The
active API remains `degraded_ocr_only`; Vision and Embedding are not active.
