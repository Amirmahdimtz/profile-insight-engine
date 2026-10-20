# Docker Compose Deployment: Public Internet Edge

This package provides a **single-host, authenticated deployment baseline** for the currently selected `degraded_ocr_only` runtime. It does not enable Vision or Embedding and does not change the application/API contract.

> **Validation status:** the repository's accepted benchmark target was Windows 11 / CPU-only. A Linux container image and public-edge configuration are new deployment targets and must pass the local verification steps below before production use. A successful image build is not a substitute for a real API/E2E benchmark on the exact release revision.

## 1. Topology and constraints

```text
Public clients
    |
    | HTTPS (443); HTTP (80) redirects to the configured HTTPS hostname
    v
Nginx reverse proxy
  - externally supplied TLS certificate
  - HTTP Basic Auth gate
  - per-source-IP request/connection limits
  - 100 MiB maximum request body by default
    |
    | private Compose network
    v
FastAPI / Uvicorn (one process, one worker)
  - Tesseract 5 + official tessdata_best (fas, eng, osd)
  - synchronous analysis; no queue/retry/resume
    |
    v
PostgreSQL 17
  - persistent named volume
  - bootstrap/admin role is used only for initialization and migrations
  - least-privilege application role is used by FastAPI
  - migration service runs before the API starts
```

Only the proxy publishes host ports (`80` and `443`). The application and PostgreSQL have no host-published ports. The application and database share an internal Compose network; the proxy joins that network and a separate frontend network.

Operational constraints inherited from the application:

- Keep exactly **one application instance and one Uvicorn worker**. Startup recovery marks interrupted `pending`/`in_progress` analyses as `failed`; it does not resume work.
- The POST analysis request is synchronous. The historical target-host p95 was approximately 90 seconds, not a universal latency guarantee.
- `vision.enabled` remains `false`. No llama.cpp server, GPU, or Embedding service is part of this image/runtime.
- `/openapi.json` is only a basic liveness probe. The application has no dedicated health/readiness endpoint, and this probe does not prove PostgreSQL/OCR readiness.
- The API has no application-level authentication/authorization or CORS policy. Basic Auth at Nginx is a shared access gate, **not** a full user identity, per-user authorization, or quota system.

## 2. Files in this package

- `Dockerfile`: Python runtime, Tesseract executable, pinned upstream `tessdata_best` revision, non-root application user.
- `compose.yaml`: PostgreSQL, one-shot Alembic migration, API, and Nginx proxy.
- `.env.example`: non-secret Compose settings; copy it to the ignored `.env` file.
- `deploy/nginx/default.conf`: HTTPS redirect, TLS termination, Basic Auth, request limits, conservative logging, and proxy timeouts.
- `deploy/postgres/init/01-create-app-role.sh`: creates the separate application DB role and grants DML/default-sequence permissions for objects created by the migration/admin role on first database initialization.
- `deploy/preflight.sh`: fails closed on placeholder credentials/hostname, missing TLS/Auth files, mismatched certificate/key, weak local file permissions, or invalid Compose configuration.
- `.dockerignore`: excludes local credentials, TLS files, datasets, and development-only content from the image build context.

The following files are intentionally **not** supplied because they are deployment secrets/materials and depend on the operator's actual domain and identity policy:

- `deploy/tls/fullchain.pem`
- `deploy/tls/privkey.pem`
- `deploy/secrets/.htpasswd`
- `.env`

These paths are ignored by Git. Compose uses long bind-mount syntax with `create_host_path: false`; missing required files should stop startup rather than silently create empty directories.

## 3. Prerequisites

- Docker Engine and Docker Compose v2.
- A DNS name pointing to this host.
- A valid TLS certificate and private key for that DNS name, provisioned by the organization's approved certificate process. Certificate issuance and renewal are intentionally not coupled to Compose because the domain and ACME/provider policy are not known here.
- A protected operator account for creating and rotating the Basic Auth credentials.
- Host firewall/security-group rules that permit inbound TCP 80/443 only as intended. Do not publish the app or database ports. Compose binds these ports to IPv4 `0.0.0.0`; do not publish an AAAA record unless IPv6 binding/routing/firewall is separately configured and verified.
- Enough disk and memory for Python/Tesseract and temporary canonical images. The 100 MiB edge request cap is a conservative initial public-edge policy, not the application's theoretical maximum request size.

The examples below use a Linux shell with OpenSSL and GNU sed on a Docker host. Adapt the file/permission commands for Windows PowerShell or a non-GNU host.

## 4. Configure secrets and TLS

Run from the repository root.

### 4.1 Create the private environment file

```sh
cp .env.example .env
POSTGRES_PASSWORD="$(openssl rand -hex 32)"
APP_DB_PASSWORD="$(openssl rand -hex 32)"
sed -i \
  -e "s/REPLACE_WITH_ADMIN_64_HEX_CHAR_SECRET/${POSTGRES_PASSWORD}/" \
  -e "s/REPLACE_WITH_APP_64_HEX_CHAR_SECRET/${APP_DB_PASSWORD}/" .env
unset POSTGRES_PASSWORD APP_DB_PASSWORD
chmod 600 .env
```

Both generated database passwords are hexadecimal so they can be embedded in the SQLAlchemy URL without URL-escaping reserved characters. `POSTGRES_USER`/`POSTGRES_PASSWORD` are the bootstrap/admin credentials used by the migration service; FastAPI uses the separate `APP_DB_USER`/`APP_DB_PASSWORD` role. The init script runs only when PostgreSQL initializes an empty data directory. For an existing database volume, create/review the app role and grants explicitly before switching this stack; do not delete the volume to force initialization. Changing values in `.env` later does not rotate an existing database role; perform role/password rotation through an approved database procedure. Do not commit `.env`, print it in logs, or paste it into issues. Keep the role names simple (letters, digits, and underscores).

Review `.env` and set image references appropriate for the release. The provided image tags are convenient defaults for initial verification, not immutable production pins. Before production rollout, resolve and record approved image digests for the Python base, PostgreSQL, and Nginx images.

### 4.2 Configure the real hostname

Edit `deploy/nginx/default.conf` and replace **every** `example.com` occurrence with the actual public DNS name. The HTTP redirect is deliberately fixed to the configured hostname rather than reflecting an untrusted `Host` header.

### 4.3 Install the TLS certificate

Place the valid certificate chain and matching private key at:

```text
deploy/tls/fullchain.pem
deploy/tls/privkey.pem
```

Nginx expects a non-passphrase-protected PEM private key at startup; protect it with mode `600` and the host's approved secret-management controls. Confirm the certificate covers the exact configured hostname and is not expired. Compose does not issue or renew certificates; after an approved renewal updates the mounted files, validate and reload Nginx:

```sh
docker compose exec proxy nginx -t
docker compose exec proxy nginx -s reload
```

### 4.4 Create the Basic Auth file

The following command runs `htpasswd` in a disposable Apache utility container and prompts for the password without putting the plaintext password in the command line:

```sh
docker run --rm -it \
  -v "$PWD/deploy/secrets:/out" \
  --user "$(id -u):$(id -g)" \
  httpd:2.4-alpine \
  htpasswd -B -c /out/.htpasswd 'replace-with-username'
```

Replace `replace-with-username` with the intended operator/API username. For each additional user, run the same command without `-c`; `-c` creates/replaces the file and must only be used for the first account. Restrict access to the resulting file:

```sh
chmod 600 .env deploy/secrets/.htpasswd deploy/tls/privkey.pem
```

The credentials are shared at the reverse-proxy layer and do not provide per-customer authorization or usage accounting.

## 5. Validate and start

Before startup, confirm that the hostname, certificate, private key, `.htpasswd`, and `.env` are real and correct. Then run:

```sh
sh deploy/preflight.sh
docker compose build --pull
docker compose up -d
```

The preflight verifies that no placeholder hostname/password remains, the certificate is current and matches the configured hostname/key, sensitive files are not group/world-readable, the Basic Auth file contains a bcrypt entry, and Compose interpolation/configuration succeeds. It does not prove that the external DNS/firewall/certificate chain is correct from the public internet; run the external HTTPS smoke test as well.

The `migrate` service waits for PostgreSQL readiness and runs:

```text
python -m alembic -c src/infrastructure/alembic.ini upgrade head
```

The API starts only after the migration service exits successfully. If migration fails, stop the rollout and investigate; do not manually patch the schema or blindly downgrade a production database.

Inspect service state and startup logs:

```sh
docker compose ps -a
docker compose logs --tail=100 db migrate app proxy
```

Expected state: `db` healthy, `migrate` exited with code 0, `app` healthy, and `proxy` running/healthy. The proxy's container healthcheck validates Nginx configuration; it is not an external TLS or application-readiness check.

Run the Alembic metadata drift check against the configured database:

```sh
docker compose run --rm --no-deps app \
  python -m alembic -c src/infrastructure/alembic.ini check
```

Check the externally visible API. `curl` prompts for the Basic Auth password when only the username is supplied:

```sh
curl --fail --show-error --user 'replace-with-username' \
  https://your-configured-hostname.example/openapi.json
```

Replace the hostname and username placeholders with your actual values. Confirm that unauthenticated requests return `401`, HTTP requests redirect to HTTPS, and the HTTPS certificate validates without `-k`/`--insecure`. Always send Basic Auth credentials to the `https://` URL directly; do not run a credentialed request against `http://`, because some clients send the Authorization header before following a redirect.

Finally, run a controlled POST/status/result smoke test using a **non-sensitive test image** and a fresh unique `analysis_id`. The API has no DELETE endpoint and analysis IDs cannot be reused. Use the API contract documented in the root README; the edge gate applies Basic Auth to `/api/v1/`, `/docs`, and `/openapi.json` alike.

## 6. Edge policy and tuning

Initial Nginx values are explicit starting defaults, not benchmark-derived production guarantees:

- TLS protocols: TLS 1.2 and TLS 1.3.
- Maximum body size: `100m` for the whole multipart request. The application allows up to 50 images and 12 MiB per image, so the edge default is intentionally narrower than the theoretical application maximum. Raise it only after memory/disk capacity and abuse testing; do not assume 50 maximum-sized images will fit through this edge policy.
- The app's `/tmp` tmpfs is capped at 2 GiB. Adjust only after testing decompressed-image working sets and confirming sufficient host memory.
- Request rate: `1r/s` per source IP with a burst of 2; at most 2 concurrent connections per source IP.
- Upstream connect timeout: 10 seconds; send/read timeouts: 180 seconds.
- `Cache-Control: no-store` is added to prevent intermediary/browser caching of profile results.
- Access logs intentionally omit the request URI, query string, headers, and body. Source IP, timestamp, method, status, and response size remain operational metadata.

Tune limits only after representative tests. If a cloud load balancer or another reverse proxy is placed in front of Nginx, the current per-IP policy will see that proxy's source address. Configure `real_ip` only with the exact trusted proxy CIDRs and the actual forwarding header; never trust arbitrary client-supplied `X-Forwarded-For` values.

Nginx does not add a CORS allowlist. Cross-origin browser access is not enabled by this package; add only a reviewed origin allowlist if a browser client needs it.

## 7. Operations

### Logs and status

```sh
docker compose ps -a
docker compose logs --tail=200 app proxy db migrate
```

Do not add request bodies, raw image bytes, canonical image bytes, raw OCR text, provider payloads, or database credentials to operational logs. Application and database logs may still contain operational metadata; handle them accordingly.

### Restart/recovery

Avoid restarting the app while analyses are in progress. On application startup, stale temporary scopes are removed and interrupted `pending`/`in_progress` rows are marked `failed`; work is not resumed. Keep one app container and one Uvicorn worker. Do not scale the service or set multiple workers without redesigning and revalidating lifecycle/recovery ownership.

### Database backup

The named volume is persistent but is **not** a backup. Provision a reviewed backup schedule and test restoration outside the live database. Example manual custom-format dump to a protected host backup directory:

```sh
install -d -m 700 /var/backups/profile-insight-engine
umask 077
docker compose exec -T db sh -c \
  'pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > "/var/backups/profile-insight-engine/profile-insight-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

Verify each dump with `pg_restore --list <backup-file>` and periodically perform a restore drill to a separate database/host. Do not test destructive restore commands against the live database. The repository does not provide a production backup scheduler or restore automation.

### Release/update sequence

1. Review the exact source commit and approved container image digests.
2. Take a verified backup and confirm the restore/rollback plan.
3. Build the reviewed release image and run all local verification below.
4. Apply migrations through the Compose `migrate` service; stop on any failure.
5. Start/update the API and proxy, then repeat the authenticated OpenAPI and controlled end-to-end smoke tests.
6. Archive the source commit, image digests, migration result, and benchmark evidence in the organization's approved release/evidence store.

Do not call a release benchmark-accepted solely because the Docker image builds or `scripts/verify_phase10.py` succeeds. Run the real selected-profile API/E2E benchmark and `python -m evaluation.phase10_benchmark` with the authorized dataset and exact fingerprints as described in `docs/phase_10_benchmark_hardening_local_deployment.md`.

### Shutdown

```sh
docker compose stop
```

`docker compose down` removes containers and networks but preserves the named database volume. Do not use `docker compose down -v` unless database deletion is explicitly intended and separately approved.

## 8. Public-production acceptance gates

Do not treat the stack as production-ready until all applicable gates are satisfied:

- [ ] A real hostname is configured; the TLS certificate validates externally and renewal is tested.
- [ ] Secrets are generated, stored, rotated, and backed up according to policy; `.env`, `.htpasswd`, and private key remain outside source control. FastAPI uses the separate application DB role, not the PostgreSQL bootstrap/admin role.
- [ ] Approved immutable image digests are recorded for the application base, PostgreSQL, and Nginx.
- [ ] Firewall/security group exposes only intended public ports; app and DB remain private.
- [ ] Basic Auth is acceptable for the intended audience, or is replaced by an approved identity/API gateway with per-user authorization and quotas. Do not offer an anonymous public API with the current app.
- [ ] Request limits and timeouts are load/abuse-tested against representative non-sensitive data.
- [ ] Backup and restore are tested; monitoring/alerting and disk-space monitoring are operational.
- [ ] Container build, full tests, migration checks, authenticated POST/status/result smoke tests, and the real current-revision API/E2E benchmark pass on the target host.
- [ ] Privacy/retention rules cover persisted completed results and explicitly self-reported fields; there is no DELETE endpoint in the current API.

The Compose files provide a hardened starting boundary, not high availability, anonymous public access, a complete identity system, automated certificate management, production observability, or proof that Linux/container performance matches the historical Windows benchmark.
