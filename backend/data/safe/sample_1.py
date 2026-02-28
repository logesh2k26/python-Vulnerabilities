import json

def load_config(json_str):
    """Safe: Using json.loads instead of eval."""
    return json.loads(json_str)

def get_item(data, key):
    """Safe: Secure dictionary access."""
    return data.get(key)
