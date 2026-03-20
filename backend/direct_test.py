import google.generativeai as genai
import sys

API_KEY = "AIzaSyDO6ckxCC-j1h0j04WolYWJvFwKxG-URgU"
genai.configure(api_key=API_KEY)

try:
    print("AVAILABLE MODELS:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("ERROR:", str(e))
    sys.exit(1)
