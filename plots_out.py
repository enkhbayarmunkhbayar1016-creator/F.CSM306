import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "zuragnuud.png")

# Ensure directory exists
os.makedirs(OUT_DIR, exist_ok=True)

def out_path(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)
