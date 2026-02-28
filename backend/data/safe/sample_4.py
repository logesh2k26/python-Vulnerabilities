from pathlib import Path

def read_secure_file(filename):
    """Safe: Validating file paths."""
    base_path = Path("/app/data").resolve()
    target_path = (base_path / filename).resolve()
    
    if not str(target_path).startswith(str(base_path)):
        raise ValueError("Unsafe path detected")
        
    return target_path.read_text()
