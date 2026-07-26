import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(
    title="Medicine Recognition API",
    description="YOLOv8 powered API for identifying Chinese medicines",
    version="1.0.0"
)

# Enable CORS for Base44 / frontend requests
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
    
    # Fallback to base model if best.pt doesn't exist yet
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
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    # Save uploaded file temporarily to disk
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        detected_items = []

        # PASS 1: Standard confidence threshold (15%)
        results = model(temp_path, conf=0.15)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                medicine_name = model.names[class_id]
                bbox = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

                detected_items.append({
                    "medicine": medicine_name,
                    "confidence": round(conf, 1),
                    "box": bbox,
                    "is_best_guess": False
                })

        # PASS 2: "Best Guess" Fallback if no detection met 15% confidence
        if len(detected_items) == 0:
            low_conf_results = model(temp_path, conf=0.01)
            all_candidates = []

            for r in low_conf_results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    medicine_name = model.names[class_id]
                    bbox = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]

                    all_candidates.append({
                        "medicine": medicine_name,
                        "confidence": round(conf, 1),
                        "box": bbox,
                        "is_best_guess": True
                    })

            # Pick the single highest confidence prediction from low-conf candidates
            if all_candidates:
                best_match = max(all_candidates, key=lambda x: x["confidence"])
                detected_items.append(best_match)

        return {
            "success": True,
            "total_detected": len(detected_items),
            "detected_items": detected_items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        # Clean up temporary image file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
