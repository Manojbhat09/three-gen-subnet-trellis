#!/usr/bin/env python3
"""Test script to isolate import issues"""

print("Starting import test...")

try:
    print("1. Testing basic imports...")
    import sys
    import os
    print("✅ Basic imports successful")
    
    print("2. Testing torch import...")
    import torch
    print(f"✅ Torch imported: {torch.__version__}")
    
    print("3. Testing pydantic import...")
    import pydantic
    print(f"✅ Pydantic imported: {pydantic.__version__}")
    
    print("4. Testing validation.engine import...")
    import validation.engine
    print("✅ Validation engine imported")
    
    print("5. Testing data_structures import...")
    from validation.engine.data_structures import RequestData
    print("✅ RequestData imported")
    
    print("6. Testing other imports...")
    from validation.engine.io.ply.loader import PlyLoader
    print("✅ PlyLoader imported")
    
    from validation.engine.rendering.renderer import Renderer
    print("✅ Renderer imported")
    
    from validation.engine.validation_engine import ValidationEngine
    print("✅ ValidationEngine imported")
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc() 