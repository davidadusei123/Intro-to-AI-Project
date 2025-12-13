FROM python:3.11-slim

# ------------------------
# Environment
# ------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Cloud Run always sets PORT
ENV PORT=8080

WORKDIR /app

# ------------------------
# System dependencies (required by Ultralytics / OpenCV / PIL)
# ------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

# ------------------------
# Python dependencies
# ------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------
# App source
# ------------------------
COPY . .

# ------------------------
# Cloud Run port
# ------------------------
EXPOSE 8080

# ------------------------
# Start FastAPI
# ------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
