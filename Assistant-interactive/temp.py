from flask import Flask, request, jsonify,render_template
from openai import OpenAI
from dotenv import load_dotenv
import re


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
    that need to be taken from home for the activity specified below.

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



@app.route('/get-items', methods=['GET'])
def get_items():
    activity_text = request.args.get('activity')

    if not activity_text:
        return render_template("interactive-assist.html", items=[])

    client, model_name = initialize_llm_client()
    client1, model_name1 = initialize_vision_client()

    try:
        response_text = generate_response(activity_text, client, model_name)
        simplified_text = obj_recognition_gemini(response_text, client1, model_name1)
        items = re.findall(r'\*\s*(.+)', simplified_text)
        return render_template("interactive-assist.html", items=items)
    except Exception as e:
        return render_template("interactive-assist.html", items=[f"Error: {str(e)}"])


@app.route('/')
def index():
    return render_template("interactive-assist.html")


if __name__ == '__main__':
    app.run(debug=True)
