import openai
import os
import sys

# Force no proxy
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

print("Testing RAG Response Logic...")

client = openai.OpenAI(
    base_url="http://llama:11434/v1",
    api_key="ollama"
)

# Simulate the context retrieved for "Finance" role (Project X and Holiday Party, but NO Payroll)
context_docs = [
    "Project X is over budget by $2M and is failing.",
    "The office holiday party is on Friday at 5 PM."
]
context_text = "\n\n".join(context_docs)

# The NEW System Prompt we just added
system_prompt = (
    "You are a secure corporate assistant. Only answer based on the provided context. "
    "If the answer is not in the context, state that the information is not present in the documents accessible to the user. "
    "Do not use outside knowledge."
)

user_query = "How much is CFO Salary"

full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {user_query}"

print(f"\n--- Prompt Sent to LLM ---\n{full_prompt}\n--------------------------\n")

try:
    completion = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1
    )
    response = completion.choices[0].message.content
    print(f"\n--- LLM Response ---\n{response}\n--------------------")
    
    # Simple validation using lower case check
    if "not present" in response.lower() or "not in the documents" in response.lower() or "cannot find" in response.lower():
        print("\n✅ SUCCESS: Response indicates info not found (without confusing permission error).")
    else:
        print("\n⚠️ WARNING: Response might still be confusing. Check above.")

except Exception as e:
    print(f"\nFAILURE: {e}")
