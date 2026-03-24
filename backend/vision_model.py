from ultralytics import YOLO
import logging
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading HuggingFace OOD zero-shot classification pipeline...")
ood_detector = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
logger.info("OOD pipeline loaded.")

def validate_is_tower(image_path: str) -> bool:
    """
    Validates if the image is a telecom tower or related to one using a zero-shot classifier.
    Returns True if it's a tower, False otherwise.
    """
    try:
        image = Image.open(image_path)
        candidate_labels = [
            'telecom tower', 'cell tower', 'drone shot of a tower', 
            'animal', 'person', 'indoor scene', 'vehicle', 'random everyday object'
        ]
        results = ood_detector(image, candidate_labels=candidate_labels)
        
        highest_label = results[0]['label']
        logger.info(f"OOD validation for {image_path}: predicted '{highest_label}' with score {results[0]['score']}")
        
        if highest_label in ['telecom tower', 'cell tower', 'drone shot of a tower']:
            return True
        return False
    except Exception as e:
        logger.error(f"Error in OOD validation for {image_path}: {e}")
        return False

def calculate_damage_percentage(roi_image) -> float:
    """
    Calculates the percentage of damaged area (rust/anomalies) in the given ROI.
    """
    if roi_image is None or roi_image.size == 0:
        return 0.0
    
    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    
    # Define an HSV range for 'Rust' or 'Anomalies'
    lower_rust = np.array([0, 50, 20])
    upper_rust = np.array([30, 255, 200])
    
    # Create mask for rust
    mask = cv2.inRange(hsv, lower_rust, upper_rust)
    
    # Calculate percentage
    damaged_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    if total_pixels == 0:
        return 0.0
        
    percentage = (damaged_pixels / total_pixels) * 100.0
    return round(percentage, 1)

def analyze_tower_image(image_path: str) -> list:
    """
    Analyzes an image of a telecom tower for defects like rust and missing bolts.
    
    This function uses a custom-trained YOLOv8 model for inference. It returns a 
    structured JSON-friendly list of dictionaries representing detected objects, 
    including class names, confidence scores, and bounding box coordinates.

    Args:
        image_path (str): The file path to the image to analyze.

    Returns:
        list: A list of dictionaries containing the analysis results.
    """
    logger.info(f"Starting inference on image: {image_path}")
    
    try:
        # Load the custom-trained YOLO model
        model = YOLO('runs/detect/train/weights/best.pt')
        
        # Run inference on the provided image
        results = model.predict(image_path)
        
        # Log that inference ran successfully
        logger.info(f"YOLOv8 inference completed for {image_path}.")
        
        extracted_results = []
        for result in results:
            orig_img = result.orig_img
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Crop ROI for damage calculation
                x1_i, y1_i, x2_i, y2_i = int(max(0, x1)), int(max(0, y1)), int(int(x2)), int(int(y2))
                roi_image = orig_img[y1_i:y2_i, x1_i:x2_i]
                damage_area_percentage = calculate_damage_percentage(roi_image)
                
                extracted_results.append({
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "damage_area_percentage": damage_area_percentage
                })
        
        return extracted_results
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {str(e)}")
        # Return a structured error dictionary in a list
        return [{"error": "Failed to analyze image", "details": str(e)}]

if __name__ == "__main__":
    # Example usage for testing
    # result = analyze_tower_image("path/to/test/image.jpg")
    # print(result)
    pass
