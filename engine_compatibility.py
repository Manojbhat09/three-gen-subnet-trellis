# Simple compatibility layer for subnet_accurate_validator.py
# This avoids complex import chains and provides direct access to needed classes

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Import using absolute paths to avoid conflicts with built-in modules
from validation.engine.data_structures import RequestData, ValidationResultData
from validation.engine.io.ply.loader import PlyLoader
from validation.engine.rendering.renderer import Renderer
from validation.engine.validation_engine import ValidationEngine
from validation.serve import decode_and_validate_txt

# Export everything needed
__all__ = [
    'RequestData', 
    'ValidationResultData', 
    'PlyLoader', 
    'Renderer', 
    'ValidationEngine',
    'decode_and_validate_txt'
] 