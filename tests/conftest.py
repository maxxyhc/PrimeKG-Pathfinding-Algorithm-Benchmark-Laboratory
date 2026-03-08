import sys
import os

# Add notebook directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(project_root, 'notebook')
sys.path.insert(0, notebook_path)