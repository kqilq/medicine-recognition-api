import os
import shutil
import random
import glob
from PIL import Image
from ultralytics import YOLO

def auto_annotate_raw_images(source_dir="dataset"):
    """
    Scans dataset/<medicine_name> folders containing raw images.
    Auto-generates YOLO bounding box text files based on image dimensions
    so admins don't need to manually draw bounding boxes or use Roboflow.
    """
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and d not in ["images", "labels", "train", "val"]]
    
    # Sort class names deterministically to maintain consistent class IDs
    classes.sort()
    
    # Create data.yaml mapping dynamically
    yaml_content = f"path: ./dataset\ntrain: images/train\nval: images/val\n\nnames:\n"
    for idx, cls_name in enumerate(classes):
        yaml_content += f"  {idx}: {cls_name}\n"
    
    with open("data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Updated data.yaml with {len(classes)} classes: {classes}")

    # Create target YOLO directory structure
    img_train_dir = os.path.join(source_dir, "images", "train")
    img_val_dir = os.path.join(source_dir, "images", "val")
    lbl_train_dir = os.path.join(source_dir, "labels", "train")
    lbl_val_dir = os.path.join(source_dir, "labels", "val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    # Process each medicine class folder
    for class_id, cls_name in enumerate(classes):
        class_folder = os.path.join(source_dir, cls_name)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * 0.8))  # 80% train, 20% val
        
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        def process_and_move(image_list, dest_img_dir, dest_lbl_dir):
            for img_name in image_list:
                src_img_path = os.path.join(class_folder, img_name)
                base_name = os.path.splitext(img_name)[0]
                
                # Copy Image
                dest_img_path = os.path.join(dest_img_dir, f"{cls_name}_{img_name}")
                shutil.copy(src_img_path, dest_img_path)

                # Check if a manually created .txt file already exists
                src_txt_path = os.path.join(class_folder, f"{base_name}.txt")
                dest_txt_path = os.path.join(dest_lbl_dir, f"{cls_name}_{base_name}.txt")

                if os.path.exists(src_txt_path):
                    shutil.copy(src_txt_path, dest_txt_path)
                else:
                    # AUTO-GENERATE Bounding Box: Default center object bounding box
                    # (Class ID, x_center, y_center, width, height) normalized [0, 1]
                    # Suitable for sample photos centered on herbs
                    with open(dest_txt_path, "w") as txt_file:
                        txt_file.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

        process_and_move(train_imgs, img_train_dir, lbl_train_dir)
        process_and_move(val_imgs, img_val_dir, lbl_val_dir)

def main():
    print("Preparing dataset and auto-generating labels...")
    auto_annotate_raw_images("dataset")

    print("Starting YOLO Object Detection Training...")
    model = YOLO("yolov8n.pt")  # Multi-object detection weights

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs",
        name="detect_run",
        exist_ok=True
    )

    trained_weights = os.path.join("runs", "detect_run", "weights", "best.pt")
    if os.path.exists(trained_weights):
        shutil.copy(trained_weights, "best.pt")
        print("Model successfully trained and saved to best.pt!")

if __name__ == "__main__":
    main()
