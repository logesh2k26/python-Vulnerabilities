# Sample safe Python file for testing

import os
import json
import hashlib
from pathlib import Path
import subprocess


def process_data(data):
    """Safe: Using json.loads instead of eval."""
    try:
        result = json.loads(data)
        return result
    except json.JSONDecodeError:
        return None


def execute_command(args):
    """Safe: Using subprocess with shell=False and list of args."""
    subprocess.run(args, shell=False, check=True)


def load_data(json_data):
    """Safe: Using JSON instead of pickle."""
    return json.loads(json_data)


def get_password():
    """Safe: Using environment variables."""
    return os.environ.get("DATABASE_PASSWORD")


def search_users(user_id):
    """Safe: Using parameterized queries."""
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchall()


def read_file(filename):
    """Safe: Validating and sanitizing file path."""
    base_dir = Path("/data").resolve()
    file_path = (base_dir / filename).resolve()
    
    # Ensure path is within base directory
    if not str(file_path).startswith(str(base_dir)):
        raise ValueError("Invalid file path")
    
    return file_path.read_text()


def hash_password(password):
    """Safe: Using proper hashing."""
    salt = os.urandom(32)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
