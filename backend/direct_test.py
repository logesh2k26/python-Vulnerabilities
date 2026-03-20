import google.generativeai as genai
import os
import sys

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("ERROR: Set the GEMINI_API_KEY environment variable")
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
