from ultralytics import YOLO
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                extracted_results.append({
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
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
