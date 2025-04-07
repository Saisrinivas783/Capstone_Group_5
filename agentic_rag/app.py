from flask import Flask, request, jsonify,render_template
from activity_context_extractor import *
from knowledge_base_v2 import *
from validator_llm import *
from ultralytics import YOLO
import supervision as sv
import cv2 as cv


yolo_model = YOLO('yolov8m-world.pt')
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

app = Flask(__name__)



@app.route('/get-items', methods=['POST'])
def get_items():
    activity_text = request.form.get('activity')
    image_file = request.files.get('image')

    if not activity_text or not image_file:
        return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, error="Missing input text or image.")

    image_path = "static/uploaded.jpg"
    image_file.save(image_path)

    client, model_name = initialize_llm_client_extractor()
    client1, model_name1 = initialize_llm_client()
    G = load_or_build_graph()

    try:
        activity,context = context_extractor(activity_text, client, model_name)
        print("activity", activity)
        print("context", context)
        core_items, context_items= get_activity_items_by_context(G,activity,context)
        items= core_items + context_items
        response = generate_response(activity, context, items, client1, model_name1)
        print(response)
        clean_items_line = extract_items_line(response)
        recommend_items = extract_items_list(clean_items_line)
        essential_items = items + recommend_items
        print("FULL ITEMS")
        print(essential_items)

    except Exception as e:
        return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, error=f"LLM error: {str(e)}")

    try:
        yolo_model.set_classes(essential_items)
        img = cv.imread(image_path)
        results = yolo_model.predict(img)
        detections = sv.Detections.from_ultralytics(results[0])

        annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)

        output_path = "static/annotated.jpg"
        cv.imwrite(output_path, annotated)

        detected_labels = [essential_items[int(i)] for i in detections.class_id] if len(detections) else []

        return render_template("interactive-assist.html", items=detected_labels, essential_items=essential_items, image_path=output_path)
    except Exception as e:
        return render_template("interactive-assist.html", items=[], essential_items=essential_items, image_path=None, error=f"Detection error: {str(e)}")


@app.route('/')
def index():
    return render_template("interactive-assist.html")


if __name__ == '__main__':
    app.run(debug=True)




