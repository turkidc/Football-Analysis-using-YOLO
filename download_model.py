from ultralytics import YOLO
import os

def download_model():
    # Kreiraj models direktorij ako ne postoji
    os.makedirs("models", exist_ok=True)
    
    # Preuzmi YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    # Spremi model
    model.save("models/best.pt")
    print("Model je uspješno preuzet i spremljen u models/best.pt")

if __name__ == "__main__":
    download_model() 