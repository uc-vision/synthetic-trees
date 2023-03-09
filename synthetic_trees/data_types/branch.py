
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import List, Dict

from .tube import Tube
from synthetic_trees.util.o3d_abstractions import o3d_path, o3d_tube_mesh

from ..util.queries import pts_on_nearest_tube

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
    
    def to_tubes(self) -> List[Tube]:
      a_, b_, r1_, r2_ = self.xyz[:-1], self.xyz[1:], self.radii[:-1], self.radii[1:]
      return [Tube(a, b, r1, r2) for a, b, r1, r2 in zip(a_, b_, r1_, r2_)]
   
    def closest_pt(self, pt: np.array): # closest point on skeleton to query point
      return pts_on_nearest_tube(pt, self.to_tubes())
    
    def point_sample(self, sampling_rate=0.001) -> o3d.cuda.pybind.geometry.PointCloud:
      pts, radii = [], []
    
    def to_o3d_lineset(self, colour=(0, 0, 0)) -> o3d.cuda.pybind.geometry.LineSet:
      return o3d_path(self.xyz, colour)

    def to_o3d_tube(self):
      return o3d_tube_mesh(self.xyz, self.radii)