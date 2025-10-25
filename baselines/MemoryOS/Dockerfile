FROM python:3.10-slim
WORKDIR /app
COPY memoryos-pypi/ ./memoryos-pypi/
RUN pip install --no-cache-dir -r memoryos-pypi/requirements.txt && \
    apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*