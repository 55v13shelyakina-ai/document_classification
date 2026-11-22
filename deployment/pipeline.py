import time
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError

class DocumentPipeline:
    def __init__(self, config_path):
        # 1. Load Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.threshold = self.config.get('threshold', 0.5)
        self.base_dir = Path(__file__).parent
        self.model_path = self.base_dir / self.config['paths']['quality_model']
        
        self.device = torch.device("cpu")
        
        # 2. Load Model ONCE at startup
        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 128), nn.ReLU(), 
            nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self, image_path: str, filename: str) -> dict:
        start_time = time.time()
        
        # Default error response (R1.5 - never crash)
        response = {
            "file": filename,
            "status": "error",
            "quality": None,
            "quality_confidence": None,
            "document_type": None,
            "document_type_confidence": None,
            "reason": "unsupported_file",
            "model_versions": {"quality": "MobileNetV3-Frozen-v1", "doctype": "None"},
            "latency_ms": 0,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        try:
            # 3. Load Image
            img = Image.open(image_path).convert("RGB")
            
            # 4. Quality Gate Inference
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob_bad = float(self.model(tensor).item())
            
            # If probability of "bad" is higher than threshold, reject it
            if prob_bad >= self.threshold:
                response.update({
                    "status": "rejected",
                    "quality": "bad",
                    "quality_confidence": round(prob_bad, 4),
                    "reason": "poor_quality"
                })
            else:
                # 5. Document Classifier (Mocked for now since we only trained Quality)
                # In a real setup, you would run your INN/SNILS classifier here.
                response.update({
                    "status": "accepted",
                    "quality": "good",
                    "quality_confidence": round(1 - prob_bad, 4),
                    "document_type": "PASSPORT", # Mocked
                    "document_type_confidence": 0.95, # Mocked
                    "reason": None
                })

        except UnidentifiedImageError:
            response["reason"] = "unsupported_file"
        except Exception as e:
            response["reason"] = "internal_error"
        
        # Calculate latency
        response["latency_ms"] = int((time.time() - start_time) * 1000)
        return response