import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def measure_latency_torch(model, input_size=(224, 224), device="cpu", warmup=20, iterations=100):
    """Requirement: 20 warm-up predictions, then average over 100 timed predictions."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)
            
    # Timed
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end = time.time()
    
    return ((end - start) / iterations) * 1000  # ms per image

def get_metrics(y_true, y_pred):
    """Requirement: accuracy alone is never accepted."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }

def save_results_json(path, schema_data):
    """Requirement: Save results.json with the phase schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=4)
    print(f"Saved results to {path}")

def copy_error_images(paths, y_true, y_pred, dest_dir):
    """Requirement: Copy every misclassified test image into errors/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, true, pred in zip(paths, y_true, y_pred):
        if true != pred:
            import shutil
            shutil.copy2(img_path, dest_dir / Path(img_path).name)
            count += 1
    print(f"Copied {count} misclassified images to {dest_dir}")