# insjob agent image — built on each server by `job agent install`.
# The build context is the installed `insjob` package directory, copied to ./insjob.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "nvidia-ml-py>=12" \
      "json5>=0.9" "typer>=0.12" "rich>=13"

COPY insjob /app/insjob
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 DOCKER_CONFIG=/tmp/.docker
RUN printf '#!/bin/sh\nexec python -m insjob.agent.remote_ops "$@"\n' > /usr/local/bin/insjob-remote \
 && chmod +x /usr/local/bin/insjob-remote

CMD ["python", "-m", "insjob.agent.main"]
