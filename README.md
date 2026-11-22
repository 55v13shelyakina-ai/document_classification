# Document Quality Gate: Classification Pipeline

This repository contains the complete machine learning pipeline for classifying the quality of scanned documents (Good vs. Bad). The models are trained to detect defects such as glare, blur, skew, occlusion, and bad exposure.

## Project Structure

```text
6/
├── demo_quality_dataset/       # The 1800-image dataset (Good/Bad)
├── scripts/
│   ├── utils.py                # Shared helpers (metrics, JSON, latency)
│   ├── train_mobilenetv3.py    # Task 2: Baseline (Frozen MobileNetV3)
│   ├── build_yolo_dataset.py   # Prepares folders for YOLO
│   ├── train_yolo.py           # Task 3: YOLOv8 nano & small
│   ├── train_efficientnet.py   # Task 4: Third Model (EfficientNet-B0)
│   ├── train_finetune.py       # Task 5A: Fine-tuning top 30 layers
│   ├── threshold_analysis.py   # Task 5B: Sweep & PR curve
│   ├── train_augmented.py      # Task 5C: Glare augmentation
│   └── export_onnx.py          # Task 5D: ONNX Export
├── results/                    # JSONs, curves, and error galleries
├── reports/                    # Decision memos and markdown reports
└── README.md