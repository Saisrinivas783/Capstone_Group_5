from ultralytics import YOLO
import supervision as sv
import cv2 as cv
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context


model = YOLO('yolov8m-world.pt')

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


outdoor_activities_classes = items = [
    "Towel",
    "Soap/body wash",
    "Shampoo & conditioner",
    "Washcloth, loofah, or sponge",
    "Bathrobe and/or slippers",
    "Hairbrush or comb",
    "Hair ties",
    "Bath salts, bubbles, or essential oils",
    "Moisturizer/lotion",
    "Razor and shaving cream",
    "Clean clothes/underwear",
    "Flip-flops",
    "Waterproof bag",
    "Rubber duck or bath toy"
]

class_names =outdoor_activities_classes

model.set_classes(class_names)


image="b1.jpg"
img=cv.imread(image)
results=model.predict(img)

detections=sv.Detections.from_ultralytics(results[0])
annotated_frame=bounding_box_annotator.annotate(
    scene=img.copy(),
    detections=detections
)
annotated_frame=label_annotator.annotate(
    scene=annotated_frame,
    detections=detections
)

cv.imwrite( 'o1.jpg',annotated_frame)





