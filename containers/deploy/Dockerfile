FROM python:3.10-slim

WORKDIR /app

COPY deploy_models.py /app/deploy_models.py

RUN pip install --no-cache-dir google-cloud-aiplatform

# 👇 This ensures correct permissions at runtime
ENV GOOGLE_AUTH_SCOPES=https://www.googleapis.com/auth/cloud-platform

ENTRYPOINT ["python", "/app/deploy_models.py"]
