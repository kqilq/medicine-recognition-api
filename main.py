import os
import io
import gc
import requests
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Limit PyTorch to 1 CPU thread to avoid Render memory limits
torch.set_num_threads(1)

app = FastAPI(title="YOLOv8 Medicine Object Detection API")

# Enable CORS for Base44 frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (best.pt in root folder, fallback to base weights)
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8m.pt"

print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


class PredictRequest(BaseModel):
    file_url: str
    count: int = 0  # Optional user-selected count (0 means return all detected)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Medicine Detection API is active."}


@app.post("/predict")
async def predict_medicine(payload: PredictRequest):
    file_url = payload.file_url
    max_count = payload.count

    if not file_url or not isinstance(file_url, str):
        raise HTTPException(status_code=400, detail="file_url is required and must be a valid string.")

    try:
        # 1. Download image from Base44 URL
        response = requests.get(file_url, timeout=15)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_w, img_h = image.size

        # 2. Run Inference with low NMS overlap threshold to suppress duplicates
        results = model.predict(
            source=image,
            conf=0.25,  # 25% minimum confidence threshold
            iou=0.45,   # NMS threshold: drops overlapping duplicate boxes
            imgsz=416,
            verbose=False
        )

        detections = []
        result = results[0]

        # 3. Extract bounding boxes and category labels
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                detections.append({
                    "medicine": class_name,
                    "confidence": round(confidence * 100, 1),
                    # Absolute pixel coordinates: [ymin, xmin, ymax, xmax]
                    "box": [round(y1, 1), round(x1, 1), round(y2, 1), round(x2, 1)],
                    # Normalized coordinates (0.0 - 1.0) for frontend overlay
                    "box_normalized": [
                        round(y1 / img_h, 4),
                        round(x1 / img_w, 4),
                        round(y2 / img_h, 4),
                        round(x2 / img_w, 4)
                    ]
                })

        # 4. Sort detections by highest confidence score first
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        # 5. Cap results to user-selected count if provided
        if max_count > 0 and len(detections) > max_count:
            detections = detections[:max_count]

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Image download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        # Force garbage collection to keep Render memory low
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
