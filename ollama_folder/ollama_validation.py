import re
import json
from ollama import Client
from knowledge_base_v2 import *

def initialize_llm_client_validation():
    client = Client(host='http://localhost:11434')
    model_name = "deepseek-r1:1.5b"  # or llama3, gemma, etc.
    return client, model_name

def validation(activity: str, context: list, items: list, client, model_name: str) -> str:
    system_prompt = f"""
You are a smart assistant that acts as both a validator and a recommender for planning an activity.

You will receive:
- An activity name
- A list of context(s)
- A list of items the user already plans to take

Your tasks:
1. Think carefully about everything a person would need for that activity.
2. Add any extra items that are important based on the given context(s) (e.g. weather, time, location).
3. Compare this mental checklist with the provided item list.
4. If any essential or commonly recommended items are **missing**, return ONLY those missing items as a **comma-separated list**.
5. If nothing is missing, return: `none`

🚫 STRICT FORMAT:
- Do NOT explain
- Do NOT add formatting or bullets
- Do NOT wrap in tags or quotes
- Output MUST be one of:
    - item1, item2, item3
    - none

---
Activity: {activity}
Contexts: {context}
Planned Items: {items}
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



def extract_after_last_think(response_text):
    closing_tag = "</think>"
    parts = response_text.rsplit(closing_tag, 1)  # split from the right
    if len(parts) > 1:
        return parts[1].strip()
    return response_text.strip()


if __name__ == "__main__":
    client, model_name = initialize_llm_client_validation()
    G = load_or_build_graph()

    activity = 'trekking'
    context = ['rain',"cold"]
    core_items, context_items = get_activity_items_by_context(G, activity, context)

    print("core_items", core_items)
    print("context_items", context_items)

    items = core_items + context_items
    print("All existing items:", items)

    response = validation(activity, context, items, client, model_name)
    print(response)
    items = extract_after_last_think(response)
    print("🎯 Extracted Items List:", items)


