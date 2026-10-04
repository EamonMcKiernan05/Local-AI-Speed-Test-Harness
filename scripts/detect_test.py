"""Quick live backend-detection check against local endpoints (run on the box).

Usage: python3 scripts/detect_test.py [url ...]   (defaults to the two common local ports)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench

urls = sys.argv[1:] or ["http://127.0.0.1:8080", "http://127.0.0.1:8099"]
for url in urls:
    try:
        d = bench.detect(url)
        print(url, "->", d["backend"], "|", d["model"], "| ctx", d["ctx"])
    except Exception as e:
        print(url, "ERROR:", e)
