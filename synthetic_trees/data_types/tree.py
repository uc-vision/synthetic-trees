import numpy as np
import open3d as o3d

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict

from ..util.o3d_abstractions import o3d_merge_linesets, o3d_merge_meshes
from ..util.misc import flatten_list

from ..util.operations import sample_tubes
from .branch import BranchSkeleton
from .tube import Tube


@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]

    def __len__(self):
        return len(self.branches)

    def __str__(self):
        return (f"Tree Skeleton ({self._id}) has {len(self)} branches...")

    def to_tubes(self) -> List[Tube]:
        return flatten_list([branch.to_tubes() for branch in self.branches.values()])

    def to_o3d_tubes(self) -> o3d.cuda.pybind.geometry.TriangleMesh:
        return o3d_merge_meshes([branch.to_o3d_tube() for branch in self.branches.values()])

    def to_o3d_lineset(self) -> o3d.cuda.pybind.geometry.LineSet:
        return o3d_merge_linesets([branch.to_o3d_lineset() for branch in self.branches.values()])

    def point_sample(self, sample_rate=0.01) -> o3d.cuda.pybind.geometry.PointCloud:
        return sample_tubes(self.to_tubes(), sample_rate)


def repair_skeleton(skeleton: TreeSkeleton):
    """ By default the skeletons are not connected between branches.
        this function connects the branches to their parent branches by finding
        the nearest point on the parent branch - relative to radius. 
        It returns a new skeleton with no reference to the original.
    """
    skeleton = deepcopy(skeleton)

    for branch in list(skeleton.branches.values()):

        if branch.parent_id == -1 or branch.parent_id == 0:
            continue

        parent_branch = skeleton.branches[branch.parent_id]

        connection_pt, connection_rad = parent_branch.closest_pt(
            pt=branch.xyz[[0]])
        
        print(connection_pt.shape, connection_rad.shape)

        branch.xyz = np.insert(branch.xyz, 0, connection_pt, axis=0)
        branch.radii = np.insert(branch.radii, 0, connection_rad, axis=0)

    return skeleton


def prune_skeleton(skeleton: TreeSkeleton, min_radius_threshold=0.01, length_threshold=0.02, root_id=1):
    """ In the skeleton format we are using each branch only knows it's parent 
        but not it's child (could work this out by doing a traversal). If a branch doesn't
        meet the initial radius threshold or length threshold we want to remove it and all
        it's predecessors... 
        Because of the way the skeleton is initalized however we know that earlier branches
        are guaranteed to be of lower order.
        minimum_radius_threshold: some point of the branch must be above this to not remove the branch
        length_threshold: the total length of the branch must be greater than this point
    """
    branches_to_keep = {root_id: skeleton.branches[root_id]}

    for branch_id, branch in skeleton.branches.items():

        if branch.parent_id == -1:
            continue

        if branch.parent_id in branches_to_keep:
            if branch.length > length_threshold and branch.radii[0] > min_radius_threshold:
                branches_to_keep[branch_id] = branch

    return TreeSkeleton(skeleton._id, branches_to_keep)
