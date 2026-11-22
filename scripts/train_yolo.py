import time
import json
import torch
from pathlib import Path
from ultralytics import YOLO
import sys

sys.path.append(str(Path(__file__).parent))
from utils import save_results_json

SEED = 2470425
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

def train_variant(model_name, variant_name, imgsz=640):
    print(f"\n--- Training {variant_name} ---")
    model = YOLO(model_name)
    
    start_time = time.time()
    results = model.train(
        data=str(DATASET_DIR), epochs=50, imgsz=imgsz, batch=32, 
        patience=5, seed=SEED, project=str(RESULTS_DIR), name=variant_name,
        exist_ok=True
    )
    train_time = time.time() - start_time
    epochs_run = model.trainer.epoch + 1
    
    # 1. Latency protocol (20 warm-ups + 100 timed)
    import numpy as np
    from PIL import Image
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import torch
    
    # Get a real image for latency testing
    test_folder = DATASET_DIR / "test" / "good"
    sample_img_path = next(test_folder.glob("*.jpg"))
    dummy = np.array(Image.open(sample_img_path).convert("RGB"))
    
    # Warm-up
    for _ in range(20): 
        model.predict(dummy, verbose=False)
        
    # Timed
    start = time.time()
    for _ in range(100): 
        model.predict(dummy, verbose=False)
    latency = ((time.time() - start) / 100) * 1000
    
    # 2. Manual evaluation for Precision/Recall/F1 on TEST split
    print(f"Evaluating {variant_name} on Test Split manually...")
    best_weights = RESULTS_DIR / variant_name / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))
    
    y_true, y_pred = [], []
    test_dir = DATASET_DIR / "test"
    class_names = ["good", "bad"] # Alphabetic order YOLO uses
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_folder = test_dir / cls_name
        for img_path in cls_folder.glob("*.jpg"):
            res = best_model.predict(str(img_path), verbose=False)[0]
            pred_cls_idx = res.probs.top1
            y_true.append(cls_idx)
            y_pred.append(pred_cls_idx)
            
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    size_mb = best_weights.stat().st_size / (1024 * 1024)
    
    schema = {
        "experiment": f"YOLOv8_{variant_name}", "week": 7, "model": "YOLOv8-cls",
        "regime": "full_finetune", "input_size": f"{imgsz}x{imgsz}", "seed": SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs_planned": 50, "epochs_run": epochs_run, "train_time_s": train_time,
        "latency_ms_per_image": latency, "model_size_mb": size_mb, "params": 1437442, 
        "test_samples": len(y_true), "metrics": {
            "accuracy": acc, "f1_macro": f1,
            "precision_macro": prec, "recall_macro": rec
        }
    }
    save_results_json(RESULTS_DIR / variant_name / "results.json", schema)
    print(f"Finished {variant_name} | F1: {f1:.4f} | Latency: {latency:.2f}ms")
    
# Train two mandatory variants
train_variant("yolov8n-cls.pt", "yolov8n_cls")
train_variant("yolov8s-cls.pt", "yolov8s_cls")