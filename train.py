import os
import shutil
import random
import glob
from ultralytics import YOLO

def auto_annotate_and_split(source_dir="dataset", split_ratio=0.8):
    reserved_dirs = ["images", "labels", "train", "val", "runs"]
    classes = [
        d for d in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, d)) and d not in reserved_dirs
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

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    for class_id, cls_name in enumerate(classes):
        class_folder = os.path.join(source_dir, cls_name)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if not images:
            print(f"Warning: No images found in {class_folder}. Skipping...")
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
                    with open(dest_txt_path, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

        process_image_list(train_imgs, img_train_dir, lbl_train_dir)
        process_image_list(val_imgs, img_val_dir, lbl_val_dir)

def main():
    print("--- STEP 1: Preparing Dataset & Auto-Generating Labels ---")
    auto_annotate_and_split("dataset")

    print("\n--- STEP 2: Training YOLO Detection Model ---")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        project="runs",
        name="detect_run",
        exist_ok=True
    )

    print("\n--- STEP 3: Locating and Copying best.pt to Root ---")
    
    # Check directly inside YOLO's output directory
    target_weight = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    
    if os.path.exists(target_weight):
        shutil.copy(target_weight, "best.pt")
        size_mb = os.path.getsize("best.pt") / (1024 * 1024)
        print(f"SUCCESS: Copied '{target_weight}' to root 'best.pt' ({size_mb:.2f} MB)")
    else:
        # Fallback recursive search
        found_weights = glob.glob("runs/**/weights/best.pt", recursive=True)
        if found_weights:
            shutil.copy(found_weights[-1], "best.pt")
            size_mb = os.path.getsize("best.pt") / (1024 * 1024)
            print(f"SUCCESS (Fallback): Copied '{found_weights[-1]}' to root 'best.pt' ({size_mb:.2f} MB)")
        else:
            print("ERROR: Failed to find best.pt in runs/ folder!")

if __name__ == "__main__":
    main()
