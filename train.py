import os
import shutil
import random
from ultralytics import YOLO

def auto_split_dataset(source_dir="dataset", split_ratio=0.8):
    """
    Scans dataset/ subdirectories (e.g. dataset/南杏) and automatically
    splits photos into dataset_split/train and dataset_split/val.
    """
    split_dir = "dataset_split"
    
    # Clean previous split if it exists
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        # Skip internal train/val folders if present
        if cls in ["train", "val", "images", "labels"]:
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Create target directories
        train_cls_dir = os.path.join(split_dir, "train", cls)
        val_cls_dir = os.path.join(split_dir, "val", cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_cls_dir, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_cls_dir, img))

    print(f"Dataset split completed successfully into '{split_dir}'")
    return split_dir

def main():
    # 1. Automatically generate train/val directory structure
    split_dataset_path = auto_split_dataset("dataset", split_ratio=0.8)

    # 2. Load classification model
    model = YOLO("yolov8n-cls.pt")

    # 3. Train on the automatically prepared dataset path
    model.train(
        data=split_dataset_path,
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs",
        name="classify_run",
        exist_ok=True
    )

    # 4. Save trained weights to best.pt
    trained_weights = os.path.join("runs", "classify_run", "weights", "best.pt")
    if os.path.exists(trained_weights):
        shutil.copy(trained_weights, "best.pt")
        print("Success! Trained model saved to 'best.pt'")

if __name__ == "__main__":
    main()
