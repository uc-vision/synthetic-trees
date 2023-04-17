import numpy as np

from dataclasses import dataclass
from ..util.o3d_abstractions import o3d_cloud


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

    def to_o3d_cloud_labelled(self, cmap=np.asarray([[0.53, 0.24, 0.13], [0.64, 0.92, 0.20], [1.0, 0.5, 0.3], [0.8, 1.0, 0.6]])):
        return o3d_cloud(self.xyz, colours=cmap[self.class_l])
