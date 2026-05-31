import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY", "").strip()
if not key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

print(f"Key starts with: {key[:10]}...")

try:
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Hello, respond with OK only.", generation_config={"max_output_tokens": 10})
    print(f"SUCCESS: {response.text.strip()}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

    # dit werkt niet. je krijgt 0 tokens en je mag het niet gebruiken zonder te betalen, dus gemini is geen optie.