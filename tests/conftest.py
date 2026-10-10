import os
import sys

# tests run on CPU: the GPU is shared with Isaac and other agents
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
