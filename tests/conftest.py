"""
Pytest configuration for test suite.
Sets up Python path so tests can import from src/ directory.
"""
import sys
import os

# Add src directory to Python path so tests can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)