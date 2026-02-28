import subprocess

def list_files(directory):
    """Safe: Using subprocess.run with shell=False."""
    subprocess.run(["ls", directory], shell=False, check=True)
