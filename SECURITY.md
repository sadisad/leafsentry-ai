# Security policy

Report a suspected vulnerability privately to sangmalinghay@gmail.com. Include reproduction steps and affected version. Do not send credentials or private images in public issues. No response-time guarantee is implied.

## Supported scope

Version 0.1.x is an educational reference, not a hardened multi-tenant service. Do not expose it to the Internet without the controls below.

## Application controls

- JPEG, PNG, and WebP decoding only; declared MIME must match detected format.
- Default file limit 5 MiB; pixel and dimension limits are checked before decode.
- Complete image verification; decompression warnings/errors become safe errors.
- No user-configurable model IDs, remote inference URLs, or uploaded checkpoints.
- Immutable model revision and `trust_remote_code=False`; modern PyTorch uses restricted weight loading for the upstream `.bin` checkpoint. Revision pinning is not a substitute for trust in upstream artifacts.
- API errors do not expose traces. Request IDs are restricted to 64 safe characters and are never metric labels.
- CSP and anti-framing headers; the API reference has no third-party CDN scripts.
- Non-root Docker runtime and read-only Compose filesystem. Model cache and temporary uploads are writable, isolated locations.
- SHA-pinned workflow actions, locked dependencies, CodeQL, and dependency auditing.

## Deployment requirements

The app's file bound is enforced **after multipart parsing**. A reverse proxy must enforce a total HTTP-body bound before requests reach ASGI, along with upload/header timeouts, rate limits, and a concurrency budget. For a 5 MiB file, allow a small multipart overhead (for example, a 6 MiB total request cap).

The application does not provide authentication, tenant isolation, Internet-safe abuse prevention, persistent audit storage, or encryption at rest. Terminate TLS at a trusted proxy, protect `/metrics`, restrict network exposure, and monitor model-cache disk consumption. Do not place secrets in the runtime environment unless required by your deployment.

Uploads are not retained by application code, but the multipart parser may spool bytes to temporary disk; uploaded files are closed when the prediction request finishes. Proxy, platform, and backup policies must be reviewed separately. The application does not detect all non-leaf, unsupported-crop, or malicious-but-valid image inputs.

## Local checks

```bash
uv sync --frozen --extra dev --no-install-project
uv run --no-sync ruff check .
uv export --frozen --no-dev --extra ml --no-emit-project --output-file /tmp/leafsentry-runtime.txt
uv run --no-sync pip-audit --no-deps --disable-pip -r /tmp/leafsentry-runtime.txt
```

The runtime export includes optional ML packages even when the audit environment has not installed weights or ML dependencies. Known issues must be investigated or upgraded, not suppressed to produce a green badge. Audit results are time-scoped; a passing scan is not proof of absence of vulnerabilities.
