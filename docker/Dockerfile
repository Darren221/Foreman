# The application image, shared by the api, worker, and both UIs — compose runs each
# with a different command. The worker reaches the host Docker daemon through the
# mounted socket (the Python `docker` SDK talks to it directly, so no Docker CLI is
# needed in the image) to launch throwaway sandbox containers.
FROM python:3.12-slim

WORKDIR /app

# Install deps first (cached unless pyproject changes), then the source.
COPY pyproject.toml README.md ./
COPY foreman ./foreman
RUN pip install --no-cache-dir ".[postgres,ui]"

EXPOSE 8000

# Default to the API; compose overrides command for the worker and the UIs.
CMD ["uvicorn", "--factory", "foreman.api.app:build_app", "--host", "0.0.0.0", "--port", "8000"]
