# Load libraries
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# Download YOLOv11n face detection model from Hugging Face
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")

# Load YOLOv11n model
model = YOLO(model_path)

# Function to detect faces
def detect_faces(image_path):
    # Run inference
    output = model(image_path)

    # Parse results
    results = Detections.from_ultralytics(output[0])
    print(results)
    return results

# Function to cover detected faces with a black rectangle
def cover_faces(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Run face detection
    detections = detect_faces(image_path)

    # Define rectangle color (black) and thickness (-1 for filled)
    color = (0, 0, 0)  # Black color
    thickness = -1  # Fill the rectangle completely

    # Loop through detected faces
    for i in range(len(detections.xyxy)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detections.xyxy[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw filled rectangle over the face
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Display image with covered faces
    try:
        cv2.imshow("Face Detection - Covered Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Displaying images is not supported in this environment: {e}")

    # Save the modified image as "edit_covered.jpg"
    cv2.imwrite("../sample-results/face-result-3.jpg", img)

# Example usage
image_path = "../data/faces&&lot/pic3.jpg"  # Replace with your actual image path
cover_faces(image_path)
