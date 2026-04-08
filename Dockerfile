
# ---------------------------------------------------------------------------
# Support Ticket OpenEnv — Production Dockerfile
# ---------------------------------------------------------------------------
# Build:  docker build -t support-ticket-env .
# Run:    docker run -p 7860:7860 support-ticket-env
# Test:   docker run --rm support-ticket-env pytest tests/ -v
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Metadata
LABEL name="support-ticket-env" \
      version="1.0.0" \
      description="OpenEnv: Customer Support Ticket Resolution" \
      maintainer="openenv-community"

# Hugging Face Spaces requirements
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
FROM python:3.13.5-slim

WORKDIR /app 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

