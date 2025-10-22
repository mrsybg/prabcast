# Simple redirect to layout.py for Streamlit
# This file serves as an entry point that imports the main application
# DON'T use "from app import" - it triggers __init__.py which loads models before state init
# Instead, load layout module directly

import sys
import importlib.util
from pathlib import Path

# Get absolute path to layout.py
layout_path = Path(__file__).parent / "layout.py"

# Load layout.py directly without triggering app.__init__
spec = importlib.util.spec_from_file_location("app.layout", layout_path)
layout_module = importlib.util.module_from_spec(spec)
sys.modules["app.layout"] = layout_module
spec.loader.exec_module(layout_module)