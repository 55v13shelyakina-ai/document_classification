import json
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Import utilities
sys.path.append(str(Path(__file__).parent))
from utils import measure_latency_torch, get_metrics, save_results_json, copy_error_images

# --- STRICT PARAMETERS ---
SEED = 2470425  # Student ID
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "demo_quality_dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "mobilenetv3_frozen"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

# --- DATASET ---
class QualityDataset(Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.records = []
        with open(self.dataset_dir / "metadata.jsonl", "r") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                if rec["split"] == split:
                    self.records.append(rec)

    def __len__(self): return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.dataset_dir / rec["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        label = 0 if rec["quality_label"] == "good" else 1
        return img, torch.tensor(label, dtype=torch.float32), str(img_path)

# Standard PyTorch normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(QualityDataset(DATA_DIR, "train", transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QualityDataset(DATA_DIR, "validation", transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(QualityDataset(DATA_DIR, "test", transform), batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL: MobileNetV3-Large ---
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False # Frozen base

# Custom Head: GAP -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)
# MobileNetV3's classifier[0] is Linear, so we replace the whole block.
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

criterion = nn.BCELoss() # Binary crossentropy
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # Only train the head

# --- TRAINING LOOP ---
history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
best_val_loss = float('inf')
patience_counter = 0
epochs_run = 0

start_time = time.time()
for epoch in range(EPOCHS):
    epochs_run += 1
    model.train()
    train_loss, train_correct = 0, 0
    
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_correct += ((outputs > 0.5) == labels).sum().item()
        
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += ((outputs > 0.5) == labels).sum().item()

    tr_loss = train_loss / len(train_loader.dataset)
    tr_acc = train_correct / len(train_loader.dataset)
    vl_loss = val_loss / len(val_loader.dataset)
    vl_acc = val_correct / len(val_loader.dataset)
    
    history["loss"].append(tr_loss)
    history["accuracy"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_accuracy"].append(vl_acc)
    
    print(f"Epoch {epoch+1}: val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f}")
    
    # Early Stopping & Model Checkpoint
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_counter = 0
        torch.save(model.state_dict(), RESULTS_DIR / "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
train_time = time.time() - start_time

# --- EVALUATION ON TEST ---
model.load_state_dict(torch.load(RESULTS_DIR / "best.pt"))
model.eval()
y_true, y_pred, paths = [], [], []

with torch.no_grad():
    for imgs, labels, batch_paths in test_loader:
        outputs = model(imgs.to(DEVICE))
        preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        y_true.extend(labels.numpy().astype(int))
        y_pred.extend(preds)
        paths.extend(batch_paths)

metrics = get_metrics(y_true, y_pred)
copy_error_images(paths, y_true, y_pred, RESULTS_DIR / "errors")

# --- LATENCY & RESULTS ---
latency = measure_latency_torch(model, input_size=(224, 224), device=DEVICE)
size_mb = (RESULTS_DIR / "best.pt").stat().st_size / (1024 * 1024)
params = sum(p.numel() for p in model.parameters())

schema = {
    "experiment": "MobileNetV3_Frozen", "week": 6, "model": "MobileNetV3-Large", "regime": "frozen_base",
    "input_size": "224x224", "seed": SEED, "device": str(DEVICE),
    "epochs_planned": EPOCHS, "epochs_run": epochs_run, "train_time_s": train_time,
    "latency_ms_per_image": latency, "model_size_mb": size_mb, "params": params,
    "test_samples": len(y_true), "metrics": metrics, "history": history
}
save_results_json(RESULTS_DIR / "results.json", schema)

# --- PLOTS ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(history["loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history["accuracy"], label="Train"); plt.plot(history["val_accuracy"], label="Val"); plt.title("Accuracy"); plt.legend()
plt.savefig(RESULTS_DIR / "curves.png")
print(f"Done. Results saved to {RESULTS_DIR}")