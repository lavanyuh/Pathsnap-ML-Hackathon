#!/usr/bin/env python3
"""Launcher for PathSnap FastAPI Map Matching Service (Laptop 1)."""
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f"🚀 PathSnap Distributed Map Matching API Starting")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Access UI at: http://localhost:{port}/")
    print(f"   From Laptop 2: http://<LAPTOP_1_IP>:{port}/")
    print(f"=======================================================\n")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False, log_level="info")
