from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

image_path = "data/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/OVERCAST/2015-11-16/camera5/2015-11-16_1148.jpg"
image = Image.open(image_path)

#https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection
feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')
model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')

inputs = feature_extractor(images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

logits = outputs.logits
bboxes = outputs.pred_boxes

bboxes = bboxes.detach().cpu()
bboxes = bboxes * torch.tensor([image.width, image.height, image.width, image.height])

bboxes = bboxes.tolist()

confidence_scores = torch.sigmoid(logits).detach().cpu()

draw = ImageDraw.Draw(image)
for box, score in zip(bboxes[0], confidence_scores[0]):
    confidence = score[0].item()  # Extract scalar value from tensor
    confidence = confidence*100
    print("\nConfidence: ", confidence)
    if confidence < 0.3:
        continue
    x_min, y_min, x_max, y_max = box
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    draw.rectangle([x_min, y_min, x_max, y_max], fill="black")

output_path = "output_image_with_covered_plates.jpg"
image.save(output_path)
print(f"Output image saved to {output_path}")

#image.show()