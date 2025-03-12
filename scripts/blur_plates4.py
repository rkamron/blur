from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

def model_run(path, conf_thresh=0.5, iou_thresh=0.4):
    # Step 1: Download the YOLO model weights from Hugging Face
    weights_path = hf_hub_download(repo_id="krishnamishra8848/Nepal-Vehicle-License-Plate-Detection", filename="last.pt")

    # Step 2: Load the YOLO model
    model = YOLO(weights_path)

    # Load and preprocess the image
    image = Image.open(path).convert('RGB')
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = img_cv.shape

    # Perform inference
    results = model.predict(img_cv, conf=conf_thresh, iou=iou_thresh)

    # Extract detections
    detections = []
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integers
                if conf >= conf_thresh:  # Apply confidence threshold
                    detections.append((x1, y1, x2, y2, conf))

    # Apply Non-Maximum Suppression (NMS)
    if len(detections) > 0:
        boxes = np.array([d[:4] for d in detections])
        scores = np.array([d[4] for d in detections])

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)

        color = (0, 255, 255)  # Yellow for rectangle
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2, conf = detections[i]

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                label = f'License Plate {conf:.2f}'
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Crop detected license plate for OCR
                roi = img_cv[y1:y2, x1:x2]

                # Perform OCR on the cropped license plate
                plate_text = pytesseract.image_to_string(roi, config='--psm 6')
                print(f"Detected text: {plate_text.strip()}")

    # Display image with detections
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Run the model with an image
model_run('../data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-25/camera7/2015-11-25_1619.jpg', conf_thresh=0.6, iou_thresh=0.4)
