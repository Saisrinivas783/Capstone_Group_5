import re
import ast
from ollama import Client

def initialize_llm_client_extractor():
    client = Client(host='http://localhost:11434')  # Default Ollama endpoint
    model_name = "deepseek-r1:1.5b"  # or use "llama3", "gemma", etc.
    return client, model_name

def context_extractor(query: str, client, model_name: str):
    system_prompt = """
You are a precise assistant that extracts structured information from user queries.

Your task is:
1. Identify and return the **exact activity phrase** used by the user.
   - Do not change, normalize, or reword the activity.
   - If the user says "hike", return "hike".
   - If the user says "hiking", return "hiking".
   - Preserve the original form as stated in the query.

2. Extract any **context** (e.g., weather, time, environment) if mentioned.

Return the output in the following strict JSON format:

{
  "activity": "<exact_activity_phrase>",
  "context": ["<context1>", "<context2>"]  // or "no context" if none found
}

Only return the JSON — no explanation or extra text.
"""

    try:
        full_prompt = system_prompt + f"\n\nUser Query: {query}"

        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        response_text = response['message']['content']
        print("🔍 LLM Response:\n", response_text)

        activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', response_text)
        context_match = re.search(r'"context"\s*:\s*(\[[^\]]*\])', response_text)

        context = []
        if context_match:
            context_list = ast.literal_eval(context_match.group(1))
            context = [c.strip() for c in context_list]

        activity = activity_match.group(1) if activity_match else None
        return activity, context

    except Exception as e:
        print("❌ Error:", e)
        return None, []


if __name__ == "__main__":
    client, model_name = initialize_llm_client_extractor()
    query = "i am going for hiking"
    activity, context = context_extractor(query, client, model_name)
    print("🧭 Activity:", activity)
    print("🌦️ Context:", context)

