import json
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent))
from utils import measure_latency_torch, get_metrics, save_results_json, copy_error_images
from train_mobilenetv3 import QualityDataset # Reuse the dataset class

SEED = 2470425
EPOCHS, PATIENCE, BATCH_SIZE = 50, 5, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "demo_quality_dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "efficientnet_b0"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

# EfficientNet-B0 specific preprocessing (Torch mode)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(QualityDataset(DATA_DIR, "train", transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QualityDataset(DATA_DIR, "validation", transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(QualityDataset(DATA_DIR, "test", transform), batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False

# Custom Head
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
best_val_loss = float('inf')
patience_counter = 0
epochs_run = 0

for epoch in range(EPOCHS):
    epochs_run += 1
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_correct += ((model(imgs) > 0.5) == labels).sum().item()
        
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            loss = criterion(model(imgs), labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += ((model(imgs) > 0.5) == labels).sum().item()

    tr_loss, tr_acc = train_loss/len(train_loader.dataset), train_correct/len(train_loader.dataset)
    vl_loss, vl_acc = val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)
    
    history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
    history["val_loss"].append(vl_loss); history["val_accuracy"].append(vl_acc)
    print(f"Epoch {epoch+1}: val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f}")
    
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_counter = 0
        torch.save(model.state_dict(), RESULTS_DIR / "best.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE: break

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
params = sum(p.numel() for p in model.parameters())

save_results_json(RESULTS_DIR / "results.json", {
    "experiment": "EfficientNet_B0", "week": 8, "model": "EfficientNet-B0", "regime": "frozen_base",
    "input_size": "224x224", "seed": SEED, "device": str(DEVICE), "epochs_planned": EPOCHS,
    "epochs_run": epochs_run, "train_time_s": 0, "latency_ms_per_image": latency,
    "model_size_mb": size_mb, "params": params, "test_samples": len(y_true),
    "metrics": metrics, "history": history
})
print("EfficientNet-B0 done.")