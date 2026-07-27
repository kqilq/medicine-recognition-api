import os
import shutil
import random
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects_and_get_yolo_boxes(img_path, class_id):
    """
    Scans an image using OpenCV, isolates objects against the background,
    and returns precise individual YOLO bounding boxes for each piece.
    If the image is a blank background, it returns an empty list.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    h, w, _ = img.shape
    
    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to separate herbs from white/light background
    _, thresh = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (outlines) of distinct shapes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = (w * h) * 0.01  # Ignore tiny noise artifacts smaller than 1% of image size
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Normalize coordinates for YOLO (0.0 to 1.0)
            x_center = (x + bw / 2.0) / w
            y_center = (y + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h
            
            boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
    # Returns empty list [] for blank background photos so YOLO treats them as negative samples
    return boxes

def auto_annotate_and_split(source_dir="dataset", split_ratio=0.8):
    reserved_dirs = ["images", "labels", "train", "val", "runs", ".git", ".github"]
    classes = [
        d for d in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, d)) 
        and d not in reserved_dirs 
        and not d.startswith(".")
    ]
    classes.sort()
    
    if not classes:
        print("No raw medicine folders found in dataset/.")
        return

    print(f"Found {len(classes)} medicine categories: {classes}")

    # Generate data.yaml
    yaml_content = f"path: ./dataset\ntrain: images/train\nval: images/val\n\nnames:\n"
    for idx, cls_name in enumerate(classes):
        yaml_content += f"  {idx}: {cls_name}\n"
    
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    img_train_dir = os.path.join(source_dir, "images", "train")
    img_val_dir = os.path.join(source_dir, "images", "val")
    lbl_train_dir = os.path.join(source_dir, "labels", "train")
    lbl_val_dir = os.path.join(source_dir, "labels", "val")

    # Clean previous splits & caches
    for generated_dir in [os.path.join(source_dir, "images"), os.path.join(source_dir, "labels")]:
        if os.path.exists(generated_dir):
            shutil.rmtree(generated_dir)

    for cache_file in glob.glob(f"{source_dir}/**/*.cache", recursive=True):
        try:
            os.remove(cache_file)
        except OSError:
            pass

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    for class_id, cls_name in enumerate(classes):
        class_folder = os.path.join(source_dir, cls_name)
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
                shutil.copy(src_img_path, dest_img_path)

                src_txt_path = os.path.join(class_folder, f"{base_name}.txt")
                dest_txt_path = os.path.join(dest_lbl_dir, f"{cls_name}_{base_name}.txt")

                if os.path.exists(src_txt_path):
                    shutil.copy(src_txt_path, dest_txt_path)
                else:
                    # Dynamically calculate tight boxes or write an empty txt for blank photos
                    boxes = detect_objects_and_get_yolo_boxes(src_img_path, class_id)
                    with open(dest_txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.writelines(boxes)

        process_image_list(train_imgs, img_train_dir, lbl_train_dir)
        process_image_list(val_imgs, img_val_dir, lbl_val_dir)

def main():
    print("--- STEP 1: Auto-Detecting Objects & Generating Labels ---")
    auto_annotate_and_split("dataset")

    print("\n--- STEP 2: Training YOLO Model ---")
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="data.yaml",
        epochs=60,
        imgsz=416,
        batch=4,
        workers=2,
        lr0=0.005,
        project="runs",
        name="detect_run",
        exist_ok=True
    )

    print("\n--- STEP 3: Saving Model ---")
    target_weight = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(target_weight):
        shutil.copy(target_weight, "best.pt")
        print("Successfully saved best.pt to root!")

if __name__ == "__main__":
    main()
