import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENROUTER_KEY", "").strip()
model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instant").strip()

if not key:
    print("ERROR: OPENROUTER_KEY not found in .env")
    exit(1)

print(f"Key starts with: {key[:10]}...")
print(f"Model: {model}")

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "Respond with OK only."},
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 10,
    "temperature": 0.4,
    "stream": False
}

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",
    "X-Title": "Test"
}

try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"SUCCESS: {result['choices'][0]['message']['content'].strip()}")
    else:
        print(f"ERROR BODY: {response.text[:500]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")