import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import List, Dict

from synthetic_trees.util.o3d_abstractions import o3d_merge_linesets, o3d_merge_meshes

from .branch import BranchSkeleton


@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]

    def __len__(self):
        return len(self.branches)

    def __str__(self):
        return f"Tree Skeleton ({self._id}) has {len(self)} branches..."

    def to_o3d_lineset(self) -> o3d.cuda.pybind.geometry.LineSet:
        return o3d_merge_linesets([branch.to_o3d_lineset() for branch in self.branches.values()])
    
    def to_o3d_mesh(self) -> o3d.cuda.pybind.geometry.TriangleMesh:
        return o3d_merge_meshes([branch.to_o3d_tube() for branch in self.branches.values()])
