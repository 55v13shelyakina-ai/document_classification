import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pipeline import DocumentPipeline

app = FastAPI(title="Document Quality Gate API")

# Load model ONCE at startup (R1.2)
pipeline = DocumentPipeline("config.yaml")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run pipeline
        result = pipeline.run(temp_path, file.filename)
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running. Go to /docs for Swagger UI."}