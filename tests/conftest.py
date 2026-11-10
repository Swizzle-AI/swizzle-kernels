import sys
import os

staging_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if staging_root not in sys.path:
    sys.path.insert(0, staging_root)
