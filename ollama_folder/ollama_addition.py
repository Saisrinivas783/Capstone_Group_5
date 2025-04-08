import re
import json
from ollama import Client
from knowledge_base_v2 import *

def initialize_llm_client_data():
    """
    Initialize the Ollama client and model name.
    """
    client = Client(host='http://localhost:11434')  # default local host
    model_name ="deepseek-r1:1.5b"  # or llama3, gemma, etc.
    return client, model_name


def llm_generate_response(activity: str, context: list, client, model_name: str) -> str:
    print(activity, context)

    context_text = ", ".join(context) if context else "no context"

    system_prompt = f"""
You are a helpful assistant that recommends only the items needed for an activity and its specified contexts.

Your task:
1. List the **core items** required for the activity: {activity}
2. Then list only the items that are relevant to the following context(s): {context_text}
3. Do NOT include any explanation, notes, or items unrelated to the activity or the provided contexts.
4. Do NOT include any core items under context.
5. Follow this exact format:

Core Items:
- item1
- item2

Context Items (context1):
- item3
- item4

Context Items (context2):
- item5

Only output items in this format. No extra information.

---
Activity: {activity}
Contexts: {context_text}
"""

    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return str(e)


def extract_core_and_context_items(response_text):
    core_items = []
    context_items = []

    # Find Core Items section
    core_match = re.search(r'Core Items:\s*(.*?)\n\n', response_text, re.DOTALL | re.IGNORECASE)
    if core_match:
        core_block = core_match.group(1)
        core_items = [item.strip("- ").strip() for item in core_block.strip().split('\n') if item.strip().startswith("-")]

    # Find Context Items section
    context_match = re.search(r'Context Items.*?:\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)
    if context_match:
        context_block = context_match.group(1)
        context_items = [item.strip("- ").strip() for item in context_block.strip().split('\n') if item.strip().startswith("-")]

    return core_items, context_items


if __name__ == "__main__":
    client, model_name = initialize_llm_client_data()
    G = load_or_build_graph()

    activity = "trekking"
    context = ["cold","night"]

    response_text = llm_generate_response(activity, context, client, model_name)
    core_items, context_items = extract_core_and_context_items(response_text)

    print("✅ Core Items:", core_items)
    print("🌦️ Context Items:", context_items)
