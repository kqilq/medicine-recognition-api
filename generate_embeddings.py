import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

device = torch.device("cpu")

# --- FIXED EXTRACTOR: Uses raw ImageNet features directly ---
class MedicineFeatureExtractor(nn.Module):
    def __init__(self):
        super(MedicineFeatureExtractor, self).__init__()
        # Load standard ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep everything up to the final pooling layer (removes the final classification layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)         # Output shape: [batch, 512, 1, 1]
        x = torch.flatten(x, 1)       # Output shape: [batch, 512]
        return nn.functional.normalize(x, p=2, dim=1) # L2 normalize feature vector

model = MedicineFeatureExtractor().to(device).eval()

# Standard PyTorch Preprocessing matching ImageNet expectations
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DATASET_DIR = "dataset"
OUTPUT_FILE = "medicine_embeddings.pt"

def build_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Directory '{DATASET_DIR}' does not exist.")
        return

    class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    all_embeddings = []
    all_labels = []

    print(f"Found {len(class_names)} medicines: {class_names}")

    for idx, name in enumerate(class_names):
        folder_path = os.path.join(DATASET_DIR, name)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing '{name}' ({len(image_files)} photos)...")

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            try:
                pil_image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(img_tensor)

                all_embeddings.append(emb)
                all_labels.append(idx)
            except Exception as e:
                print(f"Skipping corrupt image {img_path}: {e}")

    if all_embeddings:
        db_embeddings = torch.cat(all_embeddings, dim=0)
        db_labels = torch.tensor(all_labels, dtype=torch.long)

        torch.save({
            "embeddings": db_embeddings,
            "labels": db_labels,
            "class_names": class_names
        }, OUTPUT_FILE)
        
        print(f"\nSuccess! Saved embeddings for {len(class_names)} medicines to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    build_dataset()
