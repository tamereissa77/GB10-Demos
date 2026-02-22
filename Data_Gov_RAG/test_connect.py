import openai
import os
import sys

# Force no proxy
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

print(f"Testing OpenAI connection from Python {sys.version}")
print(f"OpenAI version: {openai.__version__}")

try:
    client = openai.OpenAI(
        base_url="http://llama:11434/v1",
        api_key="ollama"
    )
    print("Client initialized. Attempting to list models...")
    models = client.models.list()
    print("Success! Models found:")
    for m in models.data:
        print(f" - {m.id}")

    print("\nAttempting chat completion...")
    completion = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("Chat completion success:", completion.choices[0].message.content)

except Exception as e:
    print(f"\nFAILURE: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
