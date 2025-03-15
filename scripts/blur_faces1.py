# libraries
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# hugging face model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# this is comment
# model
model = YOLO(model_path)

# function to find how many detections there are 
def detect_faces(image_path):
    # load image
    image = Image.open(image_path).convert("RGB")

    # run inference
    output = model(image)

    # parse results
    results = Detections.from_ultralytics(output[0])

    #print(results)
    return results

# function to draw rectangles around each detection
def cover_faces(image_path):
    # open image using opencv
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # run moddel
    detections = detect_faces(image_path)

    # get image dimensions
    height, width, _ = img.shape

    # Define rectangle color (black) and thickness (-1 for filled)
    color = (0, 0, 0)  # Black color
    thickness = -1  # Fill the rectangle completely

    # Loop through detected faces
    for i in range(len(detections.xyxy)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detections.xyxy[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        snip = img[y1:y2, x1:x2] 
        blurred_snip = cv2.GaussianBlur(snip, (51, 51), 0)
        img[y1:y2, x1:x2] = blurred_snip

        # Draw filled rectangle over the face
        #cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Display image with covered faces
    try:
        cv2.imshow("Face Detection - Covered Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Displaying images is not supported in this environment: {e}")

    # Save the modified image
    cv2.imwrite("../sample-results/blurred_face_3.jpg", img)

image_path = "../data/faces&&lot/pic3.jpg" 
cover_faces(image_path)

