from flask import Flask, request, render_template
from openai import OpenAI
from dotenv import load_dotenv
import re
import os
import cv2 as cv
from ultralytics import YOLO
import supervision as sv
from neo4j import GraphDatabase

# Load environment variables from .env
load_dotenv()

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")

# Neo4j Configuration
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# YOLOv8 setup
yolo_model = YOLO(YOLO_MODEL_PATH)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Flask setup
app = Flask(__name__)

def initialize_llm_client():
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    return client, OPENROUTER_MODEL

def generate_activity_context(query: str, client, model_name: str) -> str:
    system_prompt = f"""You are a helpful assistant. Extract the following:

1. The name of the **activity**.
2. The **context** (e.g., outdoor, indoor, winter, night). If no context is provided, respond with 'None'.

Respond only in this format:
activity: <activity_name>
context: <context_name or None>

Now extract from this: {query}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

def generate_items_list(query: str, client, model_name: str) -> str:
    system_prompt = f"""You are a smart assistant. Based on the activity and context, return only a **comma-separated list** of essential items.

- No bullet points, no numbers.
- No labels like 'items:'.
- Just comma-separated items in a single line.

Input:
{query}

Output:
item1, item2, item3, ...
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

def fetch_items_from_graph(activity, context=None):
    with driver.session() as session:
        if context and context.lower() != "none":
            combo_check = session.run(
                """
                MATCH (a:Activity)-[:HAS_CONTEXT]->(c:Context)
                WHERE toLower(a.name) = toLower($activity) AND toLower(c.name) = toLower($context)
                RETURN c
                """,
                {"activity": activity, "context": context}
            )
            if combo_check.peek() is None:
                print(f"[Neo4j] Activity '{activity}' exists but context '{context}' is new.")
                return None
        else:
            activity_check = session.run(
                "MATCH (a:Activity) WHERE toLower(a.name) = toLower($activity) RETURN a",
                {"activity": activity}
            )
            if activity_check.peek() is None:
                print(f"[Neo4j] Activity '{activity}' not found.")
                return None

        activity_items = session.run(
            """
            MATCH (a:Activity)-[:REQUIRES]->(i:Item)
            WHERE toLower(a.name) = toLower($activity)
            RETURN i.name AS item
            """,
            {"activity": activity}
        )
        items = [r["item"] for r in activity_items]

        if context and context.lower() != "none":
            context_result = session.run(
                """
                MATCH (a:Activity)-[:HAS_CONTEXT]->(c:Context)-[:INFLUENCES]->(i:Item)
                WHERE toLower(a.name) = toLower($activity) AND toLower(c.name) = toLower($context)
                RETURN i.name AS item
                """,
                {"activity": activity, "context": context}
            )
            items += [r["item"] for r in context_result]

        return list(set(items))

@app.route('/', methods=['GET'])
def index():
    return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, summary_text="")

@app.route('/get-items', methods=['POST'])
def get_items():
    activity_description = request.form.get('activity')
    image_file = request.files.get('image')

    if not activity_description or not image_file:
        return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, summary_text="", error="Missing input text or image.")

    image_path = "static/uploaded.jpg"
    image_file.save(image_path)

    client, model_name = initialize_llm_client()

    try:
        extraction = generate_activity_context(activity_description, client, model_name)
        print("[LLM Extraction]:", extraction)

        activity_match = re.search(r'activity:\s*(.*?)(?:\n|$)', extraction, re.IGNORECASE)
        context_match = re.search(r'context:\s*(.*?)(?:\n|$)', extraction, re.IGNORECASE)

        activity = activity_match.group(1).strip().lower() if activity_match else None
        context = context_match.group(1).strip().lower() if context_match else "none"

        if not activity:
            return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, summary_text="", error="Could not extract activity.")
    except Exception as e:
        return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, summary_text="", error=f"Extraction error: {str(e)}")

    print(f"→ Querying Neo4j with: Activity='{activity}', Context='{context}'")
    essential_items = fetch_items_from_graph(activity, context)

    if essential_items is None:
        try:
            generation_response = generate_items_list(activity_description, client, model_name)
            print("[LLM Item List]:", generation_response)
            essential_items = [item.strip() for item in generation_response.split(',') if item.strip()]

            already_exists = False
            with driver.session() as session:
                if context and context != "none":
                    combo_check = session.run(
                        """
                        MATCH (a:Activity)-[:HAS_CONTEXT]->(c:Context)
                        WHERE toLower(a.name) = toLower($activity) AND toLower(c.name) = toLower($context)
                        RETURN c
                        """,
                        {"activity": activity, "context": context}
                    )
                    if combo_check.peek() is not None:
                        already_exists = True
                else:
                    activity_check = session.run(
                        "MATCH (a:Activity) WHERE toLower(a.name) = toLower($activity) RETURN a",
                        {"activity": activity}
                    )
                    if activity_check.peek() is not None:
                        already_exists = True

            if not already_exists:
                with driver.session() as session:
                    session.run(
                        """
                        MERGE (a:Activity {name: $activity})
                        WITH a
                        UNWIND $items AS item_name
                        MERGE (i:Item {name: item_name})
                        MERGE (a)-[:REQUIRES]->(i)
                        """,
                        {"activity": activity, "items": essential_items}
                    )

                    if context and context != "none":
                        session.run(
                            """
                            MERGE (a:Activity {name: $activity})
                            MERGE (c:Context {name: $context})
                            MERGE (a)-[:HAS_CONTEXT]->(c)
                            WITH c
                            UNWIND $items AS item_name
                            MERGE (i:Item {name: item_name})
                            MERGE (c)-[:INFLUENCES]->(i)
                            """,
                            {"activity": activity, "context": context, "items": essential_items}
                        )

                print(f"[Neo4j] Inserted new activity '{activity}' with items and context '{context}'")

        except Exception as e:
            return render_template("interactive-assist.html", items=[], essential_items=[], image_path=None, summary_text="", error=f"LLM generation/Neo4j update error: {str(e)}")

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

        summary_text = (
            f"The following essential items were detected in the image: {', '.join(detected_labels)}."
            if detected_labels else "No essential items were detected in the image."
        )

        return render_template(
            "interactive-assist.html",
            items=detected_labels,
            essential_items=essential_items,
            image_path=output_path,
            summary_text=summary_text
        )
    except Exception as e:
        return render_template("interactive-assist.html", items=[], essential_items=essential_items, image_path=None, summary_text="", error=f"Detection error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5050)
