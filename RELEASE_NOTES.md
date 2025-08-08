# Release Notes: LCM Image Generator v1.2.0

## Added
- OCI image metadata in Dockerfile for improved GHCR visibility.
- `LOG_LEVEL` environment variable support for configurable logging.
- FastAPI web UI with `/healthz` endpoint for monitoring.
- Non-root user execution in Docker images for enhanced security.
- CLI mode support for direct image generation.

## Changed
- Updated `transformers` to version 4.53.0 in `requirements.txt`.
- Refactored Dockerfile for simplified model weight loading and improved readability.
- GitHub Actions workflows now use specific commit SHAs and versioned actions for reliability.
- Docker image publishing workflow derives lowercase repository name for GHCR consistency.

## Fixed
- Enhanced error handling in `security.yml` for `pip-audit` and Trivy scans.
- Improved release creation logic using official GitHub Script in `create-release.yml`.

## Security
- Removed deprecated Snyk container and security workflows (`snyk-container.yml`, `snyk-security.yml`).
- Docker images hardened with non-root user execution and OCI labels.

## CI/CD
- Updated workflows to use version tags for actions.
- Improved release handling and error management in `docker-publish.yml` and `create-release.yml`.

## Docker
- Images available at:
  - `ghcr.io/lazarev-cloud/lcm:latest`
  - `ghcr.io/lazarev-cloud/lcm:gpu`
- Optimized build process with Python 3.10 base image.
- Added support for environment variable-driven configuration.

## Docs
- Updated README with Docker usage instructions, CLI mode details, and GitHub Actions workflow explanations.

---

### Quick start

**CPU:**
```bash
docker run --rm -p 8000:8000 ghcr.io/lazarev-cloud/lcm:latest
```

**GPU:**
```bash
docker run --rm -p 8000:8000 --gpus all ghcr.io/lazarev-cloud/lcm:gpu
```
