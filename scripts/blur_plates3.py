from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np

def model_run(path, conf_thresh=0.5, iou_thresh=0.4):
    # Load the YOLOS model and feature extractor
    feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
    model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')

    # Load image
    image = Image.open(path)
    img_cv = cv2.imread(path)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found at {path}")

    img_h, img_w, _ = img_cv.shape

    # Convert image for model input
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract predictions
    logits = outputs.logits  # Predicted class scores
    bboxes = outputs.pred_boxes  # Predicted bounding boxes (relative format)

    # Convert predictions to a usable format
    detections = []
    
    for logit, bbox in zip(logits[0], bboxes[0]):
        probs = torch.softmax(logit, dim=0)  # Convert logits to probabilities
        conf = probs.max().item()  # Get max confidence score
        class_idx = torch.argmax(logit).item()  # Get class index

        if conf >= conf_thresh:  # Apply confidence threshold
            # Convert bbox from relative format to absolute pixel values
            x_center, y_center, width, height = bbox * torch.tensor([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)
            detections.append((x1, y1, x2, y2, conf))  # Store detection

    # Convert detections to numpy format for NMS
    if len(detections) > 0:
        boxes = np.array([d[:4] for d in detections])
        scores = np.array([d[4] for d in detections])

        # Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)

        color = (0, 255, 255)  # Yellow for rectangle
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2, conf = detections[i]
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                label = f'License Plate {conf:.2f}'
                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display image with detections
    try:
        cv2.imshow('License Plate Detection', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Displaying images is not supported in this environment: {e}")

    # Save the result
    cv2.imwrite('result_yolos.jpg', img_cv)

# Run the model with an image
model_run('../data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-25/camera7/2015-11-25_1619.jpg', conf_thresh=0.6, iou_thresh=0.4)
