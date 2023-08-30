import numpy as np

from typing import List
from dataclasses import dataclass


@dataclass
class Tube:
    a: np.array  # Start Point 3
    b: np.array  # End Point 3
    r1: float  # Start Radius
    r2: float  # End Radius


@dataclass
class CollatedTube:
    a: np.array  # Nx3
    b: np.array  # Nx3
    r1: np.array  # N
    r2: np.array  # N


def collate_tubes(tubes: List[Tube]) -> CollatedTube:

    a = np.stack([tube.a for tube in tubes])
    b = np.stack([tube.b for tube in tubes])

    r1 = np.array([tube.r1 for tube in tubes])
    r2 = np.array([tube.r2 for tube in tubes])

    return CollatedTube(a, b, r1, r2)
