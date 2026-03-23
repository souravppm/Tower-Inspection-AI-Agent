from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize the model using the pre-trained 'yolov8n.pt' weights
    model = YOLO('yolov8n.pt')
    
    # Train the YOLOv8 model on the dataset
    print("Starting YOLOv8 training on the dataset...")
    model.train(
        data='datasets/data.yaml',
        epochs=25,
        imgsz=640,
        batch=8,
        device=0  # Use device=0 for the RTX 4060 GPU
    )
    print("Training complete.")
