
import requests
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print(f"Server is up! Status: {r.json()}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Server timed out.")
    return False

def test_analyze_vulnerable():
    print("\nTesting vulnerable code analysis...")
    code = """
import os
def hack(cmd):
    os.system(cmd)
    """
    
    payload = {
        "content": code,
        "filename": "hack.py"
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/v1/analyze", json=payload)
        if r.status_code == 200:
            data = r.json()
            is_vuln = data.get("is_vulnerable")
            print(f"Result: {data.get('label')}")
            print(f"Confidence: {data.get('overall_confidence')}")
            
            vulns = data.get("vulnerabilities", [])
            print(f"Found {len(vulns)} vulnerabilities:")
            for v in vulns:
                print(f"- {v['type']}: {v['description']}")
                
            if is_vuln and any(v['type'] == 'command_injection' for v in vulns):
                print("PASS: Vulnerability detected correctly.")
                return True
            else:
                print("FAIL: Vulnerability not detected as expected.")
                return False
        else:
            print(f"FAIL: Status {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"FAIL: Exception {e}")
        return False

def test_analyze_safe():
    print("\nTesting safe code analysis...")
    code = """
def greet(name):
    print(f"Hello {name}")
    """
    
    payload = {
        "content": code,
        "filename": "greet.py"
    }
    
    try:
        r = requests.post(f"{BASE_URL}/api/v1/analyze", json=payload)
        if r.status_code == 200:
            data = r.json()
            is_vuln = data.get("is_vulnerable")
            print(f"Result: {data.get('label')}")
            
            if not is_vuln:
                print("PASS: Code correctly identified as safe.")
                return True
            else:
                print("FAIL: Safe code flagged as vulnerable.")
                return False
        else:
            print(f"FAIL: Status {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"FAIL: Exception {e}")
        return False

if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)
        
    v_pass = test_analyze_vulnerable()
    s_pass = test_analyze_safe()
    
    if v_pass and s_pass:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nTESTS FAILED")
        sys.exit(1)
