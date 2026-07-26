from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Run detection
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
