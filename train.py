import os
import shutil
import random
from ultralytics import YOLO

def organize_and_split():
    dataset_dir = "dataset"
    img_train = os.path.join(dataset_dir, "images", "train")
    img_val = os.path.join(dataset_dir, "images", "val")
    lbl_train = os.path.join(dataset_dir, "labels", "train")
    lbl_val = os.path.join(dataset_dir, "labels", "val")

    # Create destination directories
    for d in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(d, exist_ok=True)

    # Walk through root folder or subfolder images
    image_extensions = ('.jpg', '.jpeg', '.png')
    all_images = []
    for root, _, files in os.walk(dataset_dir):
        if "images" in root or "labels" in root:
            continue
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))

    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)

    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    def move_pairs(file_list, dest_img, dest_lbl):
        for img_path in file_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            txt_path = os.path.splitext(img_path)[0] + ".txt"

            shutil.copy(img_path, os.path.join(dest_img, base_name + ext))
            if os.path.exists(txt_path):
                shutil.copy(txt_path, os.path.join(dest_lbl, base_name + ".txt"))

    move_pairs(train_files, img_train, lbl_train)
    move_pairs(val_files, img_val, lbl_val)

def main():
    organize_and_split()
    
    model = YOLO("yolov8n.pt")
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

if __name__ == "__main__":
    main()
