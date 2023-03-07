
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import List, Dict

from synthetic_trees.util.o3d_abstractions import o3d_path


@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: np.array
    radii: np.array
    child_id: int = -1

    @property
    def length(self):
        return np.sum(np.sqrt(np.sum(np.diff(self.xyz, axis=0)**2, axis=1)))

    def __len__(self):
        return self.xyz.shape[0]

    def __str__(self):
        return f"Branch {self._id} with {self.xyz} points. \
             and {self.radii} radii"

    def to_o3d_lineset(self, colour=(0, 0, 0)) -> o3d.cuda.pybind.geometry.LineSet:
        return o3d_path(self.xyz, colour)

    def to_o3d_mesh(self) -> o3d.cuda.pybind.geometry.LineSet:
        return
