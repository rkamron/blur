from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

# loading one image at a time for now
image_path = "data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-16/camera5/2015-11-16_1148.jpg"
image = Image.open(image_path)

# Load the feature extractor and model
feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')

# Preprocess the image with padding enabled
inputs = feature_extractor(images=image, return_tensors="pt", padding=True)

# running the model
outputs = model(**inputs)

# extracting bounding boxes and logits
logits = outputs.logits
bboxes = outputs.pred_boxes

# converting to [x_min, y_min, x_max, y_max] 
bboxes = bboxes.detach().cpu() 
bboxes = bboxes * torch.tensor([image.width, image.height, image.width, image.height]) 

bboxes = bboxes.tolist()

confidence_scores = torch.sigmoid(logits)
print(confidence_scores)

# Draw black rectangles over the detected license plates
draw = ImageDraw.Draw(image)
for box in bboxes[0]:
    #print("\nbox: ", box)
    # print("\nbox.tolist: ", box.tolist())
    x_min, y_min, x_max, y_max = box
    
    # validating coordinates
    if x_min == 0 and y_min == 0 and x_max == 0 and y_max == 0:
        continue
    #swapping x's and y's if needed
    if x_min > x_max:
        x_min, x_max = x_max, x_min  # Swap x_min and x_max
    if y_min > y_max:
        y_min, y_max = y_max, y_min  # Swap y_min and y_max

    draw.rectangle([x_min, y_min, x_max, y_max], fill="black")




output_path = "output_image_with_covered_plates.jpg"
image.save(output_path)
print(f"Output image saved to {output_path}")

image.show()