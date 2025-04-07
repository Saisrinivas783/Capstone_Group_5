
import json
import base64
import re
from openai import OpenAI

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


def context_activity_extractor(query: str, client, model_name: str) -> str:
    system_prompt = "You're a smart parser. Extract activity and context from the user's input."
    prompt = f"""
    User query: "{query}"

    Extract the main activity and any important context (e.g., time, weather, companions, etc). 
    Return it as a JSON like:

    {{
      "activity": "...",
      "context": ["...", "..."]
    }}
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

def parse_llm_output(llm_output):
    # Extract activity
    activity_match = re.search(r'\*\*Activity:\*\* (.+)', llm_output)
    activity = activity_match.group(1).strip().lower() if activity_match else None

    # Extract context (manually or from Environment/Key Risks)
    context = []
    env_match = re.search(r'\*\*Context:\*\* (.+)', llm_output)
    risk_match = re.search(r'\*\*Key Risks:\*\* (.+)', llm_output)

    if env_match:
        context += [w.strip().lower() for w in re.split('[,;]', env_match.group(1))]

    if risk_match:
        context += [w.strip().lower() for w in re.split('[,;]', risk_match.group(1))]

    # Optional: clean up context words
    context = list(set([c for c in context if len(c) > 2]))

    return {
        "activity": activity,
        "context": context
    }

activity_text=  "I'm going for swimming in the hot sunny day in evening"
client, model_name = initialize_llm_client()
response_text = context_activity_extractor(activity_text, client, model_name)
print(response_text)
result = parse_llm_output(response_text)
print(result)