ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
# Source: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# # Install dependency files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy the application code
COPY . .
ENV PATH="/app/.venv/bin:$PATH"

# Set environment to use the virtual environment
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Keep container running idle; execute train/predict manually via docker exec.
CMD ["sleep", "infinity"]
