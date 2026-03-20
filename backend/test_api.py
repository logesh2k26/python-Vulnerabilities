import requests
data = {
    "vulnerability_type": "sql_injection",
    "description": "SQL Injection",
    "code_snippet": "cursor.execute(f'SELECT * FROM users WHERE user = {var}')"
}
try:
    response = requests.post("http://127.0.0.1:8000/api/v1/explain", json=data)
    print("STATUS", response.status_code)
    print("RESPONSE", response.text)
except Exception as e:
    print("ERROR", str(e))
