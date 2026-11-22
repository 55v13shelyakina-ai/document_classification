import json
import os
from pathlib import Path
from pipeline import DocumentPipeline

# Initialize the pipeline
pipeline = DocumentPipeline("config.yaml")
samples_dir = Path("samples")
results = []

print("Running batch integration test...")

# Loop through all files in the samples folder
for file_path in samples_dir.iterdir():
    if file_path.is_file():
        print(f"Processing: {file_path.name}")
        result = pipeline.run(str(file_path), file_path.name)
        results.append(result)

# Save the results to predictions.json
with open(samples_dir / "predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nDone! Processed {len(results)} files.")
print(f"Results saved to samples/predictions.json")