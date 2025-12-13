# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import torch
from pathlib import Path

app = FastAPI(
    title="ChipSight YOLOv8 Service",
    description="PCB defect detection service with severity scoring",
    version="1.0.0",
)

# ---- Model config (lazy-load to avoid Cloud Run startup timeouts) ----
MODEL_PATH = "models/best.pt" if Path("models/best.pt").exists() else "yolov8n.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None  # will be loaded on first request


def get_model() -> YOLO:
    global _model
    if _model is None:
        m = YOLO(MODEL_PATH)
        # Cloud Run usually has CPU only; this is safe either way
        try:
            m.to(device)
        except Exception:
            pass
        _model = m
    return _model


# Defect type severity weights (higher = more critical)
SEVERITY_WEIGHTS = {
    "missing_hole": 1.5,
    "mouse_bite": 1.2,
    "open_circuit": 2.0,  # Most critical
    "short": 2.0,         # Most critical
    "spur": 1.0,
    "spurious_copper": 1.1,
}

DEFAULT_WEIGHT = 1.0


# ---- Response schema ----
class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    severity_score: float
    severity_level: str  # "Critical", "Moderate", "Minor"


class PredictionResponse(BaseModel):
    detections: List[Detection]
    annotated_image_base64: Optional[str] = None


def calculate_severity(confidence: float, bbox_area: float, class_name: str) -> Tuple[float, str]:
    """
    Calculate severity score: confidence × bounding box area × defect-type-weight
    Returns: (severity_score, severity_level)
    """
    weight = SEVERITY_WEIGHTS.get(class_name, DEFAULT_WEIGHT)
    severity_score = confidence * bbox_area * weight

    if severity_score > 0.5:
        level = "Critical"
    elif severity_score > 0.2:
        level = "Moderate"
    else:
        level = "Minor"

    return severity_score, level


def get_severity_color(severity_level: str) -> str:
    color_map = {
        "Critical": "#FF0000",
        "Moderate": "#FFA500",
        "Minor": "#00FF00",
    }
    return color_map.get(severity_level, "#FFFFFF")


def draw_annotations(img: Image.Image, detections: List[Detection]) -> Image.Image:
    draw = ImageDraw.Draw(img)

    font = None
    font_small = None

    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",                 # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Linux
        "C:/Windows/Fonts/arial.ttf",                          # Windows
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 12)
            break
        except Exception:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except Exception:
            font = None
            font_small = None

    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        color = get_severity_color(det.severity_level)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        class_name = det.class_name.replace("_", " ").title()
        conf_percent = int(det.confidence * 100)
        label = f"{class_name} {conf_percent}%"

        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(label) * 8
            text_height = 16

        label_y = max(0, y1 - text_height - 5)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 10, label_y + text_height + 5],
            fill=color,
            outline=color,
        )

        draw.text(
            (x1 + 5, label_y + 2),
            label,
            fill="white",
            font=font,
        )

        severity_text = det.severity_level
        if font_small:
            sb = draw.textbbox((0, 0), severity_text, font=font_small)
            severity_width = sb[2] - sb[0]
        else:
            severity_width = len(severity_text) * 6

        severity_y = y2 + 5
        draw.rectangle(
            [x1, severity_y, x1 + severity_width + 10, severity_y + 18],
            fill=color,
            outline=color,
        )
        draw.text(
            (x1 + 5, severity_y + 2),
            severity_text,
            fill="white",
            font=font_small,
        )

    return img


@app.get("/health")
async def healthcheck():
    # Don't force-load the model here; just report config quickly
    return {"status": "ok", "model": MODEL_PATH, "device": device}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), return_image: bool = True):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_width, img_height = img.size
        total_area = img_width * img_height
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or unsupported image file")

    # Lazy-load model (first request may take longer)
    m = get_model()

    results = m(img, verbose=False)
    result = results[0]

    detections: List[Detection] = []
    annotated_img = img.copy()

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        names = m.names

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_name = str(names[int(cls[i])])
            confidence = float(conf[i])

            bbox_area = ((x2 - x1) * (y2 - y1)) / total_area
            severity_score, severity_level = calculate_severity(confidence, bbox_area, class_name)

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    severity_score=severity_score,
                    severity_level=severity_level,
                )
            )

        if return_image:
            annotated_img = draw_annotations(annotated_img, detections)

    annotated_image_base64 = None
    if return_image:
        import base64

        buffered = io.BytesIO()
        annotated_img.save(buffered, format="PNG")
        annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return PredictionResponse(detections=detections, annotated_image_base64=annotated_image_base64)
