import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "demo_quality_dataset"
YOLO_DIR = PROJECT_ROOT / "yolo_dataset"

for split in ["train", "val", "test"]:
    for cls in ["good", "bad"]:
        (YOLO_DIR / split / cls).mkdir(parents=True, exist_ok=True)

with open(SRC_DIR / "metadata.jsonl", "r") as f:
    for line in f:
        rec = json.loads(line)
        split = "val" if rec["split"] == "validation" else rec["split"]
        img_path = SRC_DIR / rec["image_path"]
        shutil.copy2(img_path, YOLO_DIR / split / rec["quality_label"] / img_path.name)
print("YOLO dataset built.")