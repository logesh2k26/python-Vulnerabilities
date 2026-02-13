# Sample vulnerable Python file for testing

import os
import pickle
import subprocess

# Vulnerability: Hardcoded credentials
DATABASE_PASSWORD = "super_secret_password_123"
API_KEY = "sk-abcdef1234567890"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def process_user_input(user_data):
    """
    Vulnerability: eval() with user input.
    CWE-94: Improper Control of Generation of Code
    """
    result = eval(user_data)
    return result


def execute_command(cmd):
    """
    Vulnerability: Command injection via os.system().
    CWE-78: OS Command Injection
    """
    os.system(cmd)
    
    # Also vulnerable
    subprocess.call(cmd, shell=True)


def load_untrusted_data(serialized_data):
    """
    Vulnerability: Unsafe deserialization with pickle.
    CWE-502: Deserialization of Untrusted Data
    """
    return pickle.loads(serialized_data)


def search_users(user_id):
    """
    Vulnerability: SQL Injection.
    CWE-89: SQL Injection
    """
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchall()


def read_file(filename):
    """
    Vulnerability: Path Traversal.
    CWE-22: Path Traversal
    """
    with open(f"/data/{filename}", "r") as f:
        return f.read()


def fetch_url(url):
    """
    Vulnerability: SSRF - Server Side Request Forgery.
    CWE-918: Server-Side Request Forgery
    """
    import requests
    return requests.get(url).text
