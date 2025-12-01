# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
from PIL import Image
import io
import torch

app = FastAPI(
    title="YOLOv8 FastAPI Service",
    description="Simple object detection service powered by YOLOv8",
    version="1.0.0",
)

# ---- Load model once at startup ----
# Use "yolov8n.pt" for a pretrained model, or your own path: "runs/detect/train/weights/best.pt"
MODEL_PATH = "yolov8n.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)


# ---- Response schema (optional but nice with OpenAPI docs) ----
class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionResponse(BaseModel):
    detections: List[Detection]


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Basic validation
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Read file bytes
    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or unsupported image file")

    # Run inference
    # `model` can take a PIL image directly
    results = model(img, verbose=False)

    # YOLOv8 returns a Results list; take first item
    result = results[0]

    detections: List[Detection] = []

    boxes = result.boxes  # Boxes object
    if boxes is not None and len(boxes) > 0:
        # boxes.xyxy: (N, 4), boxes.conf: (N,), boxes.cls: (N,)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        names = model.names  # class id -> name

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append(
                Detection(
                    class_name=str(names[int(cls[i])]),
                    confidence=float(conf[i]),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

    return PredictionResponse(detections=detections)
