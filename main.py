import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(
    title="Medicine Recognition API",
    description="YOLOv8 powered API for identifying Chinese medicines",
    version="1.1.0"
)

# Enable CORS so Base44 / frontend can connect seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "best.pt"
    
    # Fallback to base model if best.pt is missing
    if not os.path.exists(model_path):
        print("Warning: best.pt not found. Falling back to yolov8n.pt")
        model_path = "yolov8n.pt"
        
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Medicine Recognition API is running!"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    expected_count: Optional[str] = Form(None)  # Options: "1", "2", "3", "4", "5", "5+"
):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Validate file format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Save incoming image to a temporary file for YOLO processing
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        detected_items = []

        # Run inference with a low confidence threshold (10%) to catch all potential items
        results = model(temp_path, conf=0.10)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                medicine_name = model.names[class_id]
                bbox = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

                detected_items.append({
                    "medicine": medicine_name,
                    "confidence": round(conf, 1),
                    "box": bbox
                })

        # Sort all detected candidates by confidence score (highest first)
        detected_items.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply expected count cap if user selected a specific numeric count ("1" to "5")
        if expected_count and expected_count.isdigit():
            limit = int(expected_count)
            detected_items = detected_items[:limit]

        return {
            "success": True,
            "total_detected": len(detected_items),
            "expected_count_applied": expected_count,
            "detected_items": detected_items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        # Clean up temporary image file from disk
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
