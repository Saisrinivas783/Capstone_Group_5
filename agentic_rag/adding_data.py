from openai import OpenAI
from dotenv import load_dotenv
from pandas._testing import contexts

from knowledge_base_v2 import *
import re
import json

def initialize_llm_client_data():
    """
    Initialize the OpenAI client with specified API key and model.
    Returns the client instance and model name.
    """
    #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
    api_key = "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"
    model_name = "deepseek/deepseek-r1:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, model_name

def llm_generate_response(activity: str, context: list,client, model_name: str) -> str:

    print(activity, context)
    system_prompt = f"""
    You are a helpful assistant that recommends essential items for specific activities based on provided contexts.

    I will give you:
    - An activity (e.g., "cricket")
    - A list of context(s) (e.g., "afternoon", "rain", "cold")

    Your task:
    1. Recommend a list of core items (both required and optional) needed for the activity.
    2. ONLY include context-specific items for the given contexts. Do not add unrelated contexts.
    3. Follow the structure exactly. Replace placeholders with real values.

    Return your response in this exact JSON format:

    {{
      "activity": "{activity}",
      "core_items": {{
        "item1": "requires",
        "item2": "optional"
      }},
      "contexts": {{
        "{context[0]}": {{
          "context_item1": "requires",
          "context_item2": "optional"
        }},
        "...": {{
          ...
        }}
      }}
    }}

    Use only these contexts: {context}
    
    
    activity:{activity}
    contexts:{context}

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






if __name__ == "__main__":
        client, model_name = initialize_llm_client_data()
        G = load_or_build_graph()
        activity="football"
        context=["afternoon","night"]

        response_text=llm_generate_response(activity,context,client,model_name)
        print(response_text)






