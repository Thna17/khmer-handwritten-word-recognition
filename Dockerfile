# Khmer Handwritten Text Recognition (KHTR) — Production Container
FROM python:3.11-slim

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install system dependencies (OpenCV headless runtime, font packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    fonts-khmeros \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Expose Gradio port
EXPOSE 7860

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Launch Gradio server
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
