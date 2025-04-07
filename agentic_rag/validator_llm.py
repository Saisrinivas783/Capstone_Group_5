from openai import OpenAI
from dotenv import load_dotenv
from knowledge_base_v2 import *
import re

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

def generate_response(activity: str, context: list,items: list,client, model_name: str) -> str:
    system_prompt = f""" You are a assistant .you duty is to validate and recommend the items i give 
     give you.. I will give you the activity and  context and its corresponding items to take along 
     with them for activity. If any essential item is missed in the items list recommend that item according
     to the context and activity..... i want the missing or recommeded items in the form of
     item1,item2,item3,......

    activity: {activity}
    context: {context}
    items: {items}
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},

            ],
            max_tokens=500,
            temperature=0.7
        )



        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def extract_items_line(response_text):
    # Try to capture items before **Explanation:**
    match = re.search(r'^([\w\s\-,().]+?)\s*\n\s*\*\*Explanation', response_text,
                      re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: capture first non-empty line with commas
    for line in response_text.split('\n'):
        if ',' in line and not line.strip().startswith('*'):
            return line.strip()

    return ""


def extract_items_list(text):
    if "recommended items:" in text.lower():
        # Split at the colon, take the part after it
        items_part = text.split(":", 1)[1].strip()
    else:
        items_part = text.strip()

    # Split into individual items
    items = [item.strip() for item in items_part.split(",") if item.strip()]
    return items

if __name__ == "__main__":

    client, model_name = initialize_llm_client()
    G = load_or_build_graph()
    activity='cycling'
    context=['rain']
    core_items,context_items=get_activity_items_by_context(G,activity,context)
    print("core_items",core_items)
    print("context_items",context_items)
    items=core_items+context_items
    print(items)
    response=generate_response(activity, context, items, client, model_name)
    print(response)
    clean_items_line = extract_items_line(response)
    recommend_items = extract_items_list(clean_items_line)
    full_items=items+recommend_items
    print("FULL ITEMS")
    print(full_items)

