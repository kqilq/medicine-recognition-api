from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# Load best.pt on startup
MODEL_PATH = "best.pt"
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    print("Warning: best.pt not found! Fallback to standard yolov8n.pt")
    model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Run object detection
    results = model(image, conf=0.35)
    
    detected_items = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            medicine_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            detected_items.append({
                "medicine": medicine_name,
                "confidence": round(confidence * 100, 1),
                "box": [round(coord, 2) for coord in box.xyxy[0].tolist()]
            })

    return {
        "success": True,
        "total_detected": len(detected_items),
        "detected_items": detected_items
    }
