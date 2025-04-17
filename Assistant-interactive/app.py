from flask import Flask, request, jsonify,render_template
from openai import OpenAI
from dotenv import load_dotenv
import re
import os
from ultralytics import YOLO
import supervision as sv
import cv2 as cv

yolo_model = YOLO('yolov8m-world.pt')
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


load_dotenv()
app = Flask(__name__)


def initialize_llm_client():
    """
    Initialize the OpenAI client with specified API key and model.
    Returns the client instance and model name.
    """
    #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
    api_key = "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"
    model_name = "deepseek/deepseek-r1:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, model_name

def initialize_vision_client():
    api_key = "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"
    model_name = "google/gemini-2.0-pro-exp-02-05:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, model_name

def generate_response(query: str, client, model_name: str) -> str:
    system_prompt = f""" You are a personal assistant. Your duty is to identify the essential items
    that need to be taken from home for the activity specified below...give me the outputs in the dorm 
    item1,item2 ..... format

    activity: {query}
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def obj_recognition_gemini(text: str, client1, model_name1: str):
    system_prompt = f'''
    Hello! I have a structured list of essential items grouped by various categories tailored for a specific activity. 
    The list contains detailed descriptions and may include multiple sub-items under each main item. 
    Please extract and provide a clean, simplified list of each specific item without any additional descriptions or sub-categories.

    Here is the text input: {text}

    Please format the output as a simple bullet list of items.
    '''
    try:
        response = client1.chat.completions.create(
            model=model_name1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

@app.route('/get-items', methods=['POST'])
def get_items():
    activity_text = request.form.get('activity')
    image_file = request.files.get('image')

    if not activity_text or not image_file:
        return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, error="Missing input text or image.")

    image_path = "static/uploaded.jpg"
    image_file.save(image_path)

    client, model_name = initialize_llm_client()
    client1, model_name1 = initialize_vision_client()

    try:
        response_text = generate_response(activity_text, client, model_name)
        print(response_text)
        # simplified_text = obj_recognition_gemini(response_text, client1, model_name1)
        # print(simplified_text)
        # essential_items = re.findall(r'\*\s*(.+)', simplified_text)
        essential_items = [item.strip() for item in response_text.split(',')]
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
# ---------------------------
# Utilities for Activity and Item Processing
# ---------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and stripping extra spaces.
    """
    if not text:
        return ""
    return text.strip().lower()

def merge_items(list1, list2):
    """
    Merge two lists of items into a unique set, preserving order.
    """
    seen = set()
    merged = []
    for item in list1 + list2:
        norm_item = normalize_text(item)
        if norm_item not in seen:
            seen.add(norm_item)
            merged.append(item)
    return merged

def summarize_item_list(items):
    """
    Generate a summary string of all items, separated by commas.
    """
    if not items:
        return ""
    return ", ".join(items)

def log_activity_to_console(activity: str, context: str, items: list):
    """
    Log the extracted activity, context, and essential items to the console.
    """
    print("=============================================")
    print(f"Activity Detected : {activity}")
    print(f"Context Detected  : {context}")
    print(f"Essential Items   : {summarize_item_list(items)}")
    print("=============================================")

@app.route('/')
def index():
    return render_template("interactive-assist.html")
@app.route('/health', methods=['GET'])
def health():
    return {"status": "running"}, 200

if __name__ == '__main__':
    app.run(debug=True)
