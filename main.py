import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained YOLO model
MODEL_FILE = "best.pt"
model = YOLO(MODEL_FILE) if torch.cuda.is_available() or True else None

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-Medicine Recognition API"}

@app.get("/medicines")
def get_medicines():
    if not model or not hasattr(model, 'names'):
        return {"medicines": []}
    
    names = list(model.names.values())
    return {"medicines": [{"name": name} for name in names]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLO object detection
    results = model(image, conf=0.25)
    
    detected_items = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            medicine_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]

            detected_items.append({
                "medicine_name": medicine_name,
                "confidence_score": round(confidence * 100, 1),
                "bounding_box": xyxy
            })

    return {
        "total_detected": len(detected_items),
        "detected_medicines": detected_items
    }
