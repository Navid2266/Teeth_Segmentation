from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model (replace with your model)
    model.train(
        data='data.yaml',  # Path to dataset YAML file
        epochs=100,  # Number of epochs
        batch=4,  # Batch size
        imgsz=1024,  # Image size
        workers=2,  # Number of workers
        name='teeth_model4'  # Experiment name
    )

if __name__ == "__main__":
    train_model()
