FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Server is stdlib HTTP + production classifier source. PyYAML loads deployment_config.yaml.
RUN pip install --no-cache-dir pyyaml

COPY woais_experiments /app/woais_experiments
COPY backend/main.py /app/backend/main.py

EXPOSE 8080
# Default: dry-run stub. Pass --allow-api (and API keys) for paid generation.
# This image does not provision Cloud Run or Lambda.
CMD ["python", "-m", "woais_experiments.deployment.app", "--dry-run", "--backend", "docker", "--host", "0.0.0.0", "--port", "8080"]
