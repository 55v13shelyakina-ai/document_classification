import torch
from pathlib import Path
from torchvision import models
import torch.nn as nn
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "results" / "mobilenetv3_frozen" / "best.pt"
OUT_DIR = PROJECT_ROOT / "results" / "deployment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = models.mobilenet_v3_large(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 128), nn.ReLU(), 
    nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
)
model.load_state_dict(torch.load(WEIGHTS))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = OUT_DIR / "mobilenetv3.onnx"
torch.onnx.export(model, dummy_input, str(onnx_path), opset_version=12, 
                  input_names=['input'], output_names=['output'])
print(f"Exported to {onnx_path}")