"""
Main application module for the Multimodal Tower Inspection API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
from backend.vision_model import analyze_tower_image
from backend.llm_agent import generate_report

# Initialize the FastAPI application
app = FastAPI(
    title="Multimodal Tower Inspection API",
    description="A basic FastAPI backend for the Multimodal Tower Inspection system.",
    version="0.1.0"
)

@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running.
    
    Returns:
        dict: A JSON response containing a welcome message and API status.
    """
    return {
        "message": "Welcome to the Multimodal Tower Inspection API",
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded image and run it through the vision model.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: The structured JSON output from the vision model.
    """
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Prevent path traversal vulnerabilities by using secure filename check (basic)
    file_name = os.path.basename(file.filename)
    file_path = os.path.join(temp_dir, file_name)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Analyze the saved image
        result = analyze_tower_image(file_path)
        
        # Generate report
        report = generate_report(result)
        
        return {"detections": result, "report": report}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
        
    finally:
        file.file.close()
        # Clean up the file after analysis to avoid filling the disk
        if os.path.exists(file_path):
            os.remove(file_path)

import base64
from io import BytesIO
from PIL import Image, ImageDraw

@app.post("/detect/")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Endpoint to process a batch of images and return base64 annotated images and detections.
    """
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    all_detections = []
    annotated_images_b64 = {}
    
    for file in files:
        file_name = os.path.basename(file.filename)
        file_path = os.path.join(temp_dir, file_name)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Analyze the saved image
            result = analyze_tower_image(file_path)
            
            for d in result:
                d["source_image"] = file_name
                
            all_detections.extend(result)
            
            # Generate annotated image
            base_image = Image.open(file_path).convert("RGB")
            draw = ImageDraw.Draw(base_image)
            
            for d in result:
                conf = d.get("confidence", 0)
                if conf >= 0.5:
                    box = d.get("box", [])
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        color = "lime" if conf >= 0.5 else "red"
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                        
            buffered = BytesIO()
            base_image.save(buffered, format="JPEG")
            b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            annotated_images_b64[file_name] = b64_str
            
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")
        finally:
            file.file.close()
            if os.path.exists(file_path):
                os.remove(file_path)
                
    return {
        "annotated_images": annotated_images_b64,
        "detections": all_detections
    }

class ReportRequest(BaseModel):
    detections: List[Dict[str, Any]]

@app.post("/report")
async def create_consolidated_report(request: ReportRequest):
    """
    Endpoint to receive a list of all raw detections from multiple images and generate a single LLM report.
    """
    try:
        report = generate_report(request.detections)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
