import argparse
from ultralytics import YOLO

def train_yolo(data, imgsz, epochs, batch, name, model_size):
    # Determine model path based on model size
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Training
    results = model.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, name=name)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train YOLO model')
    
    # Add arguments
    parser.add_argument('--data', type=str, help='Path to data YAML file', required=True)
    parser.add_argument('--imgsz', type=int, help='Input image size', required=True)
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('--batch', type=int, help='Batch size', required=True)
    parser.add_argument('--name', type=str, help='Name for the model', required=True)
    parser.add_argument('--model_size', type=str, help='Size of YOLO model (n, s, m, l, x)', required=True, choices=['n', 's', 'm', 'l', 'x'])
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    train_yolo(args.data, args.imgsz, args.epochs, args.batch, args.name, args.model_size)
