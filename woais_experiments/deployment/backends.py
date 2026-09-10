"""Backend command helpers. Never provisions cloud infrastructure."""

from __future__ import annotations

from typing import Any

COMMANDS: dict[str, str] = {}


def local_commands(*, python: str = "python3.13") -> str:
    return f"""# Local HTTP (dry-run; no paid APIs)
{python} -m woais_experiments.deployment.app --dry-run --backend local --host 127.0.0.1 --port 8765
{python} -m woais_experiments.deployment.benchmark --dry-run --backend local

# Local paid run (requires TOGETHER_API_KEY / OPENAI_API_KEY)
{python} -m woais_experiments.deployment.benchmark --allow-api --backend local
"""


def docker_commands(*, python: str = "python3.13") -> str:
    return f"""# Docker image (does not push or deploy)
docker build -t ecologic-deploy -f woais_experiments/deployment/Dockerfile .
docker run --rm -p 8080:8080 --name ecologic-deploy ecologic-deploy

# Benchmark against the container (dry-run inside the image unless you pass keys)
{python} -m woais_experiments.deployment.benchmark --dry-run --backend docker --base-url http://127.0.0.1:8080

# Paid generation in the container (keys via env; still requires client --allow-api)
docker run --rm -p 8080:8080 \\
  -e ECOLOGIC_DEPLOY_ALLOW_API=1 \\
  -e TOGETHER_API_KEY -e OPENAI_API_KEY \\
  ecologic-deploy python -m woais_experiments.deployment.app --allow-api --backend docker --host 0.0.0.0 --port 8080
{python} -m woais_experiments.deployment.benchmark --allow-api --backend docker --base-url http://127.0.0.1:8080
"""


def cloudrun_commands(*, python: str = "python3.13") -> str:
    return f"""# Optional Cloud Run — print-only. This package never runs gcloud deploy.
# Replace PROJECT, REGION, IMAGE.
docker build -t IMAGE -f woais_experiments/deployment/Dockerfile .
docker push IMAGE
gcloud run deploy ecologic-router --image IMAGE --region REGION --port 8080 --no-allow-unauthenticated
gcloud run services describe ecologic-router --region REGION --format='value(status.url)'

# After YOU deploy, point the client at the URL (dry-run still uses the stub if the service was started without --allow-api)
{python} -m woais_experiments.deployment.benchmark --dry-run --backend cloudrun --base-url https://SERVICE_URL
{python} -m woais_experiments.deployment.benchmark --allow-api --backend cloudrun --base-url https://SERVICE_URL
"""


def lambda_commands(*, python: str = "python3.13") -> str:
    return f"""# Optional AWS Lambda-compatible handler — print-only. This package never creates a function.
# Handler: woais_experiments.deployment.app.lambda_handler
# Local in-process (no AWS):
{python} -m woais_experiments.deployment.benchmark --dry-run --backend lambda

# sam local (you must provide a SAM/template; we do not write one automatically):
# sam local start-api
# {python} -m woais_experiments.deployment.benchmark --dry-run --backend lambda --base-url http://127.0.0.1:3000

# If you already have a Function URL:
# {python} -m woais_experiments.deployment.benchmark --allow-api --backend lambda --base-url https://FUNCTION_URL
"""


def print_all_commands(python: str = "python3.13") -> str:
    return "\n\n".join([
        local_commands(python=python).rstrip(),
        docker_commands(python=python).rstrip(),
        cloudrun_commands(python=python).rstrip(),
        lambda_commands(python=python).rstrip(),
    ])


def require_base_url(backend: str, base_url: str | None) -> None:
    if backend in {"docker", "cloudrun"} and not base_url:
        raise SystemExit(
            f"--base-url is required for backend={backend} "
            "(this tool does not provision containers or cloud services)"
        )
    if backend == "cloudrun":
        # never call gcloud
        pass


def describe_backend(backend: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "measurement_type": "MEASURED",
        "backend": backend,
        "provisions_infrastructure": False,
        "listen": cfg.get("listen"),
    }
