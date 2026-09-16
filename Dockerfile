FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GOOGLE_CLOUD_PROJECT=mb-poc-352009

WORKDIR /app

# Install package dependencies and application source
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8080

CMD ["glive", "serve", "--host", "0.0.0.0"]
