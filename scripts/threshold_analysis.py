import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

sys.path.append(str(SCRIPTS_DIR))
from scripts.train_mobilenetv3 import QualityDataset
from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 2470425
DEVICE = torch.device("cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "demo_quality_dataset"
WEIGHTS = PROJECT_ROOT / "results" / "mobilenetv3_frozen" / "best.pt"
OUT_DIR = PROJECT_ROOT / "results" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Model
model = models.mobilenet_v3_large(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[0].in_features, 128), torch.nn.ReLU(), 
    torch.nn.Dropout(0.3), torch.nn.Linear(128, 1), torch.nn.Sigmoid()
)
model.load_state_dict(torch.load(WEIGHTS))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Collect probabilities on VALIDATION ONLY (strict rule)
val_loader = DataLoader(QualityDataset(DATA_DIR, "validation", transform), batch_size=32, shuffle=False)
y_true, y_probs = [], []

with torch.no_grad():
    for imgs, labels, _ in val_loader:
        outputs = model(imgs.to(DEVICE))
        y_probs.extend(outputs.cpu().numpy().flatten())
        y_true.extend(labels.numpy().astype(int))

# Sweep Thresholds 0.10 to 0.90 (step 0.05)
thresholds = np.arange(0.10, 0.91, 0.05)
results = []
for t in thresholds:
    y_pred = (np.array(y_probs) > t).astype(int)
    # Class 1 is "bad"
    p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    results.append({"threshold": round(t, 2), "precision": p, "recall": r, "f1": f1})

# Plot PR Curve
plt.figure(figsize=(8, 6))
plt.plot([r["recall"] for r in results], [r["precision"] for r in results], marker='o', label="PR Curve")
plt.axvline(x=0.5, color='r', linestyle='--', label="Default 0.5")
plt.axvline(x=0.40, color='g', linestyle='--', label="Chosen 0.40")
plt.title("Precision-Recall Curve (Validation)")
plt.xlabel("Recall (Bad Class)")
plt.ylabel("Precision (Bad Class)")
plt.legend()
plt.savefig(OUT_DIR / "pr_curve.png")

# Calculate False Accepts at 0.5 vs 0.40
fa_05 = sum(1 for yt, yp in zip(y_true, y_probs) if yt == 1 and yp < 0.5)
fa_04 = sum(1 for yt, yp in zip(y_true, y_probs) if yt == 1 and yp < 0.40)

with open(OUT_DIR / "sweep_results.json", "w") as f:
    json.dump({"sweep": results, "fa_05": fa_05, "fa_04": fa_04}, f, indent=4)
print("Threshold analysis complete.")