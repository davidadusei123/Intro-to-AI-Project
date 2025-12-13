FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer logs correctly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ultralytics needs a writable config directory
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# System dependencies required for PIL / OpenCV / Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source (main.py, models/, etc.)
COPY . .

# Cloud Run listens on $PORT (default 8080)
EXPOSE 8080

# Start FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
