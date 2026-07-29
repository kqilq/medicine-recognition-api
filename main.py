import os
import io
import gc
import requests
import torch
from PIL import Image, ImageEnhance
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Stabilize CPU execution for cloud hosts (e.g. Render / Railway)
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

# Load trained YOLO model (best.pt in root folder, fallback to base weights)
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8s.pt"
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


class PredictRequest(BaseModel):
    file_url: str
    count: int = 0      # 0 means return all detected
    conf: float = 0.10  # Low cutoff prevents missing faint/pale items
    iou: float = 0.45   # Non-maximum suppression threshold


@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Medicine Detection API is active.",
        "supported_classes": list(model.names.values()) if hasattr(model, 'names') else []
    }


@app.post("/predict")
async def predict_medicine(payload: PredictRequest):
    file_url = payload.file_url
    max_count = payload.count
    target_conf = max(0.05, min(payload.conf, 0.90))

    if not file_url or not isinstance(file_url, str):
        raise HTTPException(status_code=400, detail="file_url is required and must be a valid string.")

    try:
        # 1. Download target image from payload URL
        response = requests.get(file_url, timeout=15)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_w, img_h = image.size

        def extract_from_result(result, img_w, img_h):
            detections_local = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0].item())
                        class_id = int(box.cls[0].item())
                        
                        # Dynamically extract class name from model state
                        class_name = model.names.get(class_id, f"Unknown_{class_id}")

                        detections_local.append({
                            "medicine": class_name,
                            "confidence": round(confidence * 100, 1),
                            "box": [round(y1, 1), round(x1, 1), round(y2, 1), round(x2, 1)],
                            "box_normalized": [
                                round(y1 / img_h, 4),
                                round(x1 / img_w, 4),
                                round(y2 / img_h, 4),
                                round(x2 / img_w, 4)
                            ]
                        })
                    except Exception:
                        continue
            return detections_local

        # 2. Primary inference pass
        with torch.no_grad():
            results = model.predict(
                source=image,
                conf=target_conf,
                iou=payload.iou,
                imgsz=640,
                verbose=False
            )

        detections = extract_from_result(results[0], img_w, img_h)

        # 3. Enhanced Fallback Pass (if initial pass yields 0 items)
        if len(detections) == 0:
            try:
                print("No detections from primary pass — running contrast fallback pass")
                enhanced = ImageEnhance.Contrast(image).enhance(1.5)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)

                with torch.no_grad():
                    fallback_results = model.predict(
                        source=enhanced,
                        conf=0.05,
                        iou=0.35,
                        imgsz=640,
                        verbose=False
                    )
                fallback_detections = extract_from_result(fallback_results[0], img_w, img_h)

                if len(fallback_detections) > 0:
                    detections = fallback_detections

            except Exception as fe:
                print(f"Fallback inference error: {fe}")

        # 4. Sort results by highest confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        # 5. Crop count limit if requested
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
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
