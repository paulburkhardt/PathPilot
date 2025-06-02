import numpy as np
from typing import Dict, Any

def test_function():
    return "Hello world"

def L2_distance(point_1: np.ndarray, point_2: np.ndarray):
    return np.linalg.norm(point_1 - point_2)

