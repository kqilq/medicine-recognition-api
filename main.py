import io
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

class MedicineFeatureExtractor(nn.Module):
    def __init__(self, embedding_size=128):
        super(MedicineFeatureExtractor, self).__init__()
        # Download standard ResNet18 backbone directly
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# Initialize model architecture
model = MedicineFeatureExtractor(embedding_size=128)

# Load custom linear layer weights safely with strict=False
if os.path.exists("medicine_model_small.pt"):
    small_weights = torch.load("medicine_model_small.pt", map_location=device)
    model.load_state_dict(small_weights, strict=False)

model.to(device).eval()

EMBEDDINGS_FILE = "medicine_embeddings.pt"

def load_index():
    if os.path.exists(EMBEDDINGS_FILE):
        data = torch.load(EMBEDDINGS_FILE, map_location=device)
        return data["embeddings"], data["labels"], data["class_names"]
    return torch.empty((0, 128)), torch.tensor([]), []

db_embeddings, db_labels, class_names = load_index()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_index():
    torch.save({
        "embeddings": db_embeddings,
        "labels": db_labels,
        "class_names": class_names
    }, EMBEDDINGS_FILE)

# -------------------------------------------------------------------
# 1. Get List of All Existing Medicines (For Base44 Display)
# -------------------------------------------------------------------
@app.get("/medicines")
async def get_medicines():
    global class_names, db_labels
    
    medicine_list = []
    for idx, name in enumerate(class_names):
        photo_count = int((db_labels == idx).sum().item())
        medicine_list.append({
            "id": idx,
            "name": name,
            "photo_count": photo_count
        })
        
    return {
        "status": "success",
        "medicines": medicine_list,
        "total": len(medicine_list)
    }

# -------------------------------------------------------------------
# 2. Prediction Endpoint
# -------------------------------------------------------------------
@app.post("/predict")
async def predict_medicine(file: UploadFile = File(...)):
    global db_embeddings, db_labels, class_names
    if db_embeddings.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Database is empty. Add medicines first.")

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(img_tensor)

    similarities = torch.mm(query_embedding, db_embeddings.T).squeeze(0)
    top_similarity, top_idx = torch.max(similarities, dim=0)
    
    predicted_name = class_names[db_labels[top_idx].item()]
    confidence = round(float((top_similarity.item() + 1) / 2) * 100, 2)

    return {
        "status": "success",
        "medicine_name": predicted_name,
        "confidence_score": confidence
    }

# -------------------------------------------------------------------
# 3. Add or Update Medicine Endpoint
# -------------------------------------------------------------------
@app.post("/add-medicine")
async def add_medicine(medicine_name: str = Form(...), files: list[UploadFile] = File(...)):
    global db_embeddings, db_labels, class_names

    if medicine_name not in class_names:
        class_names.append(medicine_name)
    
    class_idx = class_names.index(medicine_name)
    new_embeddings = []

    for file in files:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(img_tensor)
            new_embeddings.append(emb)

    if new_embeddings:
        new_emb_tensor = torch.cat(new_embeddings, dim=0)
        new_label_tensor = torch.tensor([class_idx] * len(new_embeddings), dtype=torch.long)

        db_embeddings = torch.cat([db_embeddings, new_emb_tensor], dim=0)
        db_labels = torch.cat([db_labels, new_label_tensor], dim=0)
        save_index()

    return {
        "status": "success",
        "message": f"Added {len(files)} photos for '{medicine_name}'",
        "total_classes": len(class_names)
    }

# -------------------------------------------------------------------
# 4. Delete Medicine Endpoint
# -------------------------------------------------------------------
@app.delete("/remove-medicine")
async def remove_medicine(medicine_name: str = Form(...)):
    global db_embeddings, db_labels, class_names

    if medicine_name not in class_names:
        raise HTTPException(status_code=404, detail="Medicine name not found.")

    class_idx = class_names.index(medicine_name)
    
    keep_mask = (db_labels != class_idx)
    db_embeddings = db_embeddings[keep_mask]
    db_labels = db_labels[keep_mask]

    db_labels[db_labels > class_idx] -= 1
    class_names.remove(medicine_name)

    save_index()

    return {
        "status": "success",
        "message": f"Successfully removed '{medicine_name}'",
        "remaining_classes": class_names
    }
