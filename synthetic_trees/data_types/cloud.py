import numpy as np

from dataclasses import dataclass
from synthetic_trees.util.o3d_abstractions import o3d_cloud


@dataclass
class Cloud:
    xyz: np.array
    rgb: np.array
    class_l: np.array = None

    def __len__(self):
        return xyz.shape[0]

    def __str__(self):
        return f'Cloud with {self.xyz.shape[0]} points '

    def to_o3d_cloud(self):
        return o3d_cloud(self.xyz, colours=self.rgb)
