from openai import OpenAI
from dotenv import load_dotenv
from knowledge_base_v2 import *
import re
import ast


def initialize_llm_client_extractor():
    """
    Initialize the OpenAI client with specified API key and model.
    Returns the client instance and model name.
    """
    #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
    api_key = "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"
    model_name = "deepseek/deepseek-r1:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, model_name


def context_extractor(query: str,client, model_name: str) -> str:
    system_prompt = f"""
    You are a precise assistant that extracts structured information from user queries.

    Your task is:
    1. Identify and return the **exact activity phrase** used by the user.
       - Do not change, normalize, or reword the activity.
       - If the user says "hike", return "hike".
       - If the user says "hiking", return "hiking".
       - Preserve the original form as stated in the query.

    2. Extract any **context** (e.g., weather, time, environment) if mentioned.

    Return the output in the following strict JSON format:

    {{
      "activity": "<exact_activity_phrase>",
      "context": ["<context1>", "<context2>"]  // or "no context" if none found
    }}

    Only return the JSON — no explanation or extra text.
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
        response_text= response.choices[0].message.content
        print(response_text)
        activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', response_text)
        match = re.search(r'"context"\s*:\s*(\[[^\]]*\])', response_text)
        if match:
            context_str = match.group(1)
            # Safely evaluate the list string into an actual Python list
            context_list = ast.literal_eval(context_str)
            context=[c.strip() for c in context_list]
        else:
            context=[]


        activity = activity_match.group(1) if activity_match else None

        return activity, context

    except Exception as e:
        return str(e),""

if __name__ == "__main__":

    client, model_name = initialize_llm_client_extractor()
    query="i want to go for a hiking in rain at night"
    activity,context = context_extractor(query,client, model_name)
    print(activity)
    print(context)

