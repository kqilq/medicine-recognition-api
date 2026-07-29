import os
import shutil
import random
import glob
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

def detect_objects_and_get_yolo_boxes(img_path, class_id):
    """
    Scans an image using Adaptive Thresholding to capture dark, hollow,
    or sliced medicinal roots accurately without missing pieces.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding handles varied lighting and dark organic textures
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    # Morphological closing bridges hollow interior structures
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = (w * h) * 0.003  # Detects smaller slices (> 0.3% total area)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Normalize for YOLO format
            x_center = (x + bw / 2.0) / w
            y_center = (y + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h
            
            boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
    return boxes


def auto_annotate_and_split(source_dir="dataset", split_ratio=0.8):
    reserved_dirs = ["images", "labels", "train", "val", "runs", ".git", ".github"]
    
    # 1. Load existing class index mapping from data.yaml if it exists
    existing_classes = []
    if os.path.exists("data.yaml"):
        try:
            with open("data.yaml", "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
                if data_cfg and "names" in data_cfg:
                    if isinstance(data_cfg["names"], dict):
                        existing_classes = [data_cfg["names"][k] for k in sorted(data_cfg["names"].keys())]
                    elif isinstance(data_cfg["names"], list):
                        existing_classes = data_cfg["names"]
        except Exception as e:
            print(f"Warning reading data.yaml: {e}")
            existing_classes = []

    # 2. Find medicine folders currently in dataset/
    found_folders = [
        d for d in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, d)) 
        and d not in reserved_dirs 
        and not d.startswith(".")
    ]
    
    # Preserve original indices for existing classes; append new classes at the end
    new_folders = sorted([d for d in found_folders if d not in existing_classes])
    classes = existing_classes + new_folders

    if not classes:
        print("No medicine categories found in dataset/.")
        return

    print(f"Final class mapping ({len(classes)} total): {classes}")

    # 3. Write data.yaml with locked ID mapping
    yaml_content = "path: ./dataset\ntrain: images/train\nval: images/val\n\nnames:\n"
    for idx, cls_name in enumerate(classes):
        yaml_content += f"  {idx}: {cls_name}\n"
    
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    img_train_dir = os.path.join(source_dir, "images", "train")
    img_val_dir = os.path.join(source_dir, "images", "val")
    lbl_train_dir = os.path.join(source_dir, "labels", "train")
    lbl_val_dir = os.path.join(source_dir, "labels", "val")

    # Clean stale cache files only (Preserve processed images and labels)
    for cache_file in glob.glob(f"{source_dir}/**/*.cache", recursive=True):
        try:
            os.remove(cache_file)
        except OSError:
            pass

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    # 4. Copy raw image files & generate annotations for new items
    for class_id, cls_name in enumerate(classes):
        class_folder = os.path.join(source_dir, cls_name)
        if not os.path.exists(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not images:
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * split_ratio))
        
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        def process_image_list(img_list, dest_img_dir, dest_lbl_dir):
            for img_name in img_list:
                src_img_path = os.path.join(class_folder, img_name)
                base_name = os.path.splitext(img_name)[0]
                
                dest_img_path = os.path.join(dest_img_dir, f"{cls_name}_{img_name}")
                if not os.path.exists(dest_img_path):
                    shutil.copy(src_img_path, dest_img_path)

                src_txt_path = os.path.join(class_folder, f"{base_name}.txt")
                dest_txt_path = os.path.join(dest_lbl_dir, f"{cls_name}_{base_name}.txt")

                if os.path.exists(src_txt_path):
                    shutil.copy(src_txt_path, dest_txt_path)
                elif not os.path.exists(dest_txt_path):
                    boxes = detect_objects_and_get_yolo_boxes(src_img_path, class_id)
                    with open(dest_txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.writelines(boxes)

        process_image_list(train_imgs, img_train_dir, lbl_train_dir)
        process_image_list(val_imgs, img_val_dir, lbl_val_dir)


def main():
    print("--- STEP 1: Auto-Detecting Objects & Updating Labels ---")
    auto_annotate_and_split("dataset")

    print("\n--- STEP 2: Incremental Training / Fine-Tuning YOLO ---")
    
    # Load previously trained best.pt if present; otherwise fall back to base weights
    base_model = "best.pt" if os.path.exists("best.pt") else "yolov8s.pt"
    print(f"Loading base model weights from: {base_model}")
    
    model = YOLO(base_model)

    results = model.train(
        data="data.yaml",
        epochs=40,         # Fast convergence when fine-tuning
        imgsz=640,
        batch=4,
        workers=2,
        lr0=0.001,         # Gentle learning rate prevents catastrophic unlearning
        project="runs",
        name="detect_run",
        exist_ok=True
    )

    print("\n--- STEP 3: Exporting Model Weights ---")
    target_weight = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(target_weight):
        shutil.copy(target_weight, "best.pt")
        print("Successfully updated best.pt in root directory!")

if __name__ == "__main__":
    main()
