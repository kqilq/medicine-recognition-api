import os
import yaml
from ultralytics import YOLO

DATASET_DIR = "dataset"
OUTPUT_MODEL = "best.pt"

def prepare_and_train():
    if not os.path.exists(DATASET_DIR):
        print(f"Directory '{DATASET_DIR}' not found.")
        return

    class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"Found {len(class_names)} classes: {class_names}")

    # Build dataset configuration YAML for YOLO
    yaml_data = {
        'path': os.path.abspath(DATASET_DIR),
        'train': '.',
        'val': '.',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_data, f)

    # Load pre-trained nano YOLO model for quick transfer learning
    model = YOLO('yolov8n.pt')

    # Train on dataset
    model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        project='runs',
        name='train_result',
        exist_ok=True
    )

    # Save final model weights
    model.export(format='engine') if False else None
    best_weights = os.path.join('runs', 'train_result', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        os.system(f"cp {best_weights} {OUTPUT_MODEL}")
        print(f"Successfully trained and saved model to '{OUTPUT_MODEL}'")

if __name__ == "__main__":
    prepare_and_train()
