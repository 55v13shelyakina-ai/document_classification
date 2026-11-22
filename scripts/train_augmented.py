import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
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

# --- STRICT PARAMETERS ---
SEED = 2470425
EPOCHS, PATIENCE, BATCH_SIZE = 50, 5, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJECT_ROOT / "demo_quality_dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "mobilenetv3_augmented"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

# --- CUSTOM AUGMENTATION: GLARE ---
class AddGlare:
    """Adds 1-3 semi-transparent white ellipses to simulate camera glare."""
    def __init__(self, p=0.5, alpha_min=150, alpha_max=230):
        self.p = p
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def __call__(self, img):
        # Only apply glare randomly based on probability
        if random.random() > self.p:
            return img
        
        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        w, h = img.size
        for _ in range(random.randint(1, 3)):
            x0 = random.randint(0, int(w * 0.7))
            y0 = random.randint(0, int(h * 0.7))
            x1 = min(w, x0 + random.randint(int(w * 0.2), int(w * 0.5)))
            y1 = min(h, y0 + random.randint(int(h * 0.2), int(h * 0.5)))
            alpha = random.randint(self.alpha_min, self.alpha_max)
            
            draw.ellipse([x0, y0, x1, y1], fill=(255, 255, 255, alpha))
            
        # Composite the glare onto the image and convert back to RGB
        out = Image.alpha_composite(img, overlay).convert("RGB")
        return out

# --- TRANSFORMS ---
# Train gets glare. Val/Test stay clean!
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([AddGlare(p=0.5)], p=0.5), # 50% chance to apply glare
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(QualityDataset(DATA_DIR, "train", train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QualityDataset(DATA_DIR, "validation", val_test_transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(QualityDataset(DATA_DIR, "test", val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False # Frozen base

in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

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

    tr_loss, tr_acc = train_loss/len(train_loader.dataset), train_correct/len(train_loader.dataset)
    vl_loss, vl_acc = val_loss/len(val_loader.dataset), val_correct/len(val_loader.dataset)
    
    history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
    history["val_loss"].append(vl_loss); history["val_accuracy"].append(vl_acc)
    print(f"Epoch {epoch+1}: Val Loss: {vl_loss:.4f}, Val Acc: {vl_acc:.4f}")
    
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

# --- TEST EVAL ---
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

# --- METRICS & JSON ---
latency = measure_latency_torch(model, input_size=(224, 224), device=DEVICE)
size_mb = (RESULTS_DIR / "best.pt").stat().st_size / (1024 * 1024)

schema = {
    "experiment": "MobileNetV3_Augmented", "week": 9, "model": "MobileNetV3-Large", 
    "regime": "frozen_base_augmented", "input_size": "224x224", "seed": SEED, 
    "device": str(DEVICE), "epochs_planned": EPOCHS, "epochs_run": epochs_run, 
    "train_time_s": train_time, "latency_ms_per_image": latency, "model_size_mb": size_mb, 
    "params": 3095089, "test_samples": len(y_true), "metrics": metrics, 
    "augmentation_config": {"glare_prob": 0.5, "alpha_range": "150-230", "applied_to": "train_only"},
    "history": history
}
save_results_json(RESULTS_DIR / "results.json", schema)

# --- PLOTS ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(history["loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history["accuracy"], label="Train"); plt.plot(history["val_accuracy"], label="Val"); plt.title("Accuracy"); plt.legend()
plt.savefig(RESULTS_DIR / "curves.png")
print(f"Done. Results saved to {RESULTS_DIR}")