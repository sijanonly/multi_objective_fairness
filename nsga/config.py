import numpy as np
from typing import NamedTuple, List


class NSGAConfig(NamedTuple):
    generations: List[float]
    populations: List[float]
    model_type: str
    X_sensitive_a1: np.ndarray
