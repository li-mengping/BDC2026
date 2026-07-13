ARG PYTHON_IMAGE=python:3.10-slim-bookworm@sha256:ff7161e2b8e2a56fc6a62a6099ff8feb72f1a6dbae9860cdcb9a6c65cf4c6be9
FROM ${PYTHON_IMAGE}

RUN pip install --no-cache-dir uv==0.10.6

# Set working directory
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the application code
COPY . .
ENV PATH="/app/.venv/bin:$PATH"

CMD ["sleep", "infinity"]
