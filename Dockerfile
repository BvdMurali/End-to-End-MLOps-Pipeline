FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    MLFLOW_TRACKING_URI=mlruns

# Set working directory
WORKDIR /app

# Copy and install requirements first to leverage Docker build cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY app/ /app/app/
COPY config/ /app/config/
COPY artifacts/ /app/artifacts/
COPY logs/ /app/logs/

# Create a non-root user and group
RUN groupadd -r mlops-group && \
    useradd -r -g mlops-group -d /app -s /sbin/nologin mlops-user

# Ensure correct file permissions for logs and artifacts
RUN mkdir -p /app/logs /app/artifacts && \
    chown -R mlops-user:mlops-group /app

# Switch to the non-root user
USER mlops-user

# Expose API port
EXPOSE 8000

# Docker Healthcheck utilizing python's built-in urllib
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1

# Start the FastAPI application
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
