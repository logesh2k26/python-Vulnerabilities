import google.generativeai as genai
import sys
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env file")
    sys.exit(1)
genai.configure(api_key=API_KEY)

try:
    print("AVAILABLE MODELS:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("ERROR:", str(e))
    sys.exit(1)
