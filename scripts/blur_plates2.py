import yolov5
import numpy as np
import cv2

def model_run(path):
    model = yolov5.load('keremberke/yolov5n-license-plate')

    model.conf = 0.001  # NMS confidence threshold
    model.iou = 0.01  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # Maximum number of detections per image

    img_path = path

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = np.ascontiguousarray(img)  # ensure the image array is writable

    results = model(img, size=640)

    # Parse results
    predictions = results.pred[0]
    print(f"\nLength: {len(predictions)} \npredictions: {predictions}\n")
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    color = (0, 255, 255)

    # render results
    for *box, conf, cls in predictions:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1) # ..., 2

        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #display
    try:
        cv2.imshow('License Plate Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Displaying images is not supported in this environment: {e}")

    # save new
    cv2.imwrite('result.jpg', img)

model_run('../data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-25/camera7/2015-11-25_1119.jpg')