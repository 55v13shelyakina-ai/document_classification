import json
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

sys.path.append(str(SCRIPTS_DIR))
from scripts.utils import measure_latency_torch, get_metrics, save_results_json, copy_error_images
from scripts.train_mobilenetv3 import QualityDataset 

SEED = 2470425
EPOCHS, PATIENCE, BATCH_SIZE = 50, 5, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "demo_quality_dataset"
FROZEN_WEIGHTS = PROJECT_ROOT / "results" / "mobilenetv3_frozen" / "best.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "mobilenetv3_finetune_top30"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(QualityDataset(DATA_DIR, "train", transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QualityDataset(DATA_DIR, "validation", transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(QualityDataset(DATA_DIR, "test", transform), batch_size=BATCH_SIZE, shuffle=False)

# 1. Load Model and Frozen Weights
model = models.mobilenet_v3_large(weights=None) # We will load our own weights
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
)
model.load_state_dict(torch.load(FROZEN_WEIGHTS))

# 2. Unfreeze Top ~30 Layers
# MobileNetV3 has ~20 blocks in its features. We unfreeze the last 3 blocks + the head.
for param in model.parameters():
    param.requires_grad = False # Freeze everything first

# Unfreeze the last few convolutional blocks (approximates top 30 layers)
for i in range(len(model.features) - 3, len(model.features)):
    for param in model.features[i].parameters():
        param.requires_grad = True

# Unfreeze the classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
criterion = nn.BCELoss()
# 3. LR = 1e-5 (10x lower than 0.001)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

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

    tr_loss, tr_acc = train_loss/len(train_loader.dataset), train_correct/len(train_loader.dataset)
    vl_loss, vl_acc = val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)
    
    history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
    history["val_loss"].append(vl_loss); history["val_accuracy"].append(vl_acc)
    print(f"Epoch {epoch+1}: Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f}, Acc: {vl_acc:.4f}")
    
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_counter = 0
        torch.save(model.state_dict(), RESULTS_DIR / "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE: break
train_time = time.time() - start_time

# Test Eval
model.load_state_dict(torch.load(RESULTS_DIR / "best.pt"))
model.eval()
y_true, y_pred, paths = [], [], []
with torch.no_grad():
    for imgs, labels, batch_paths in test_loader:
        outputs = model(imgs.to(DEVICE))
        preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        y_true.extend(labels.numpy().astype(int)); y_pred.extend(preds); paths.extend(batch_paths)

metrics = get_metrics(y_true, y_pred)
copy_error_images(paths, y_true, y_pred, RESULTS_DIR / "errors")
latency = measure_latency_torch(model, device=DEVICE)
size_mb = (RESULTS_DIR / "best.pt").stat().st_size / (1024 * 1024)

save_results_json(RESULTS_DIR / "results.json", {
    "experiment": "MobileNetV3_Finetune_Top30", "week": 9, "model": "MobileNetV3-Large", "regime": "finetune_top30",
    "input_size": "224x224", "seed": SEED, "device": str(DEVICE), "epochs_planned": EPOCHS,
    "epochs_run": epochs_run, "train_time_s": train_time, "latency_ms_per_image": latency,
    "model_size_mb": size_mb, "params": 3095089, "test_samples": len(y_true),
    "metrics": metrics, "history": history
})

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(history["loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history["accuracy"], label="Train"); plt.plot(history["val_accuracy"], label="Val"); plt.title("Accuracy"); plt.legend()
plt.savefig(RESULTS_DIR / "curves.png")
print("Fine-tuning done.")