import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File
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
    def __init__(self):
        super(MedicineFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return nn.functional.normalize(x, p=2, dim=1)

# Instantiate the model correctly without arguments
model = MedicineFeatureExtractor().to(device).eval()

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EMBEDDINGS_FILE = "medicine_embeddings.pt"

def load_index():
    try:
        data = torch.load(EMBEDDINGS_FILE, map_location=device)
        return data["embeddings"], data["labels"], data["class_names"]
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None, []

@app.get("/")
def root():
    return {"status": "ok", "message": "Medicine Recognition API is running"}

@app.get("/medicines")
def get_medicines():
    db_embeddings, db_labels, class_names = load_index()
    if db_embeddings is None:
        return {"medicines": []}

    result = []
    for idx, name in enumerate(class_names):
        count = (db_labels == idx).sum().item()
        result.append({"name": name, "photo_count": count})

    return {"medicines": result}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    db_embeddings, db_labels, class_names = load_index()
    if db_embeddings is None or len(db_embeddings) == 0:
        return {"error": "No trained medicines found in database"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model(img_tensor)

    # Cosine similarity matrix multiplication
    similarities = torch.mm(query_emb, db_embeddings.t()).squeeze(0)
    best_idx = torch.argmax(similarities).item()
    best_score = similarities[best_idx].item()
    predicted_class_idx = db_labels[best_idx].item()

    predicted_name = class_names[predicted_class_idx]
    confidence = round(max(0.0, min(100.0, float(best_score) * 100)), 1)

    return {
        "medicine_name": predicted_name,
        "confidence_score": confidence
    }
