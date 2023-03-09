import numpy as np

from typing import List
from copy import deepcopy

from .tree import TreeSkeleton
from .cloud import Cloud

# from Prescient_Tree.geometries.conversion import tree_skeleton_as_tubes, branch_skeleton_as_tubes
# from Prescient_Tree.geometries.tube import Tube
# from Prescient_Tree.geometries.branch import BranchSkeleton

# from Prescient_Tree.util.math.maths import np_normalized
# from Prescient_Tree.util.math.queries import pts_to_nearest_tube, pts_to_nearest_tube_keops, skeleton_to_points


def repair_skeleton(skeleton: TreeSkeleton):
  """ By default the skeletons are not connected between branches.
      this function connects the branches to their parent branches by finding
      the nearest point on the parent branch - relative to radius. 
      It returns a new skeleton with no reference to the original.
  """
  skeleton = deepcopy(skeleton)

  for branch in list(skeleton.branches.values()):
        
    if branch.parent_id == -1 or  branch.parent_id == 0:
      continue
    
    parent_branch = skeleton.branches[branch.parent_id]
    
    connection_pt, connection_rad = parent_branch.closest_pt(pt=branch.xyz[[0]])
        
    branch.xyz = np.insert(branch.xyz, 0, connection_pt, axis=0)
    branch.radii = np.insert(branch.radii, 0, connection_rad, axis=0)

  return skeleton


# def prune_skeleton(skeleton: TreeSkeleton, min_radius_threshold=1, length_threshold=5, root_id=0):
#   """ In the skeleton format we are using each branch only knows it's parent 
#       but not it's child (could work this out by doing a traversal). If a branch doesn't
#       meet the initial radius threshold or length threshold we want to remove it and all
#       it's predecessors... 
#       Because of the way the skeleton is initalized however we know that earlier branches
#       are guaranteed to be of lower order.
#       minimum_radius_threshold: some point of the branch must be above this to not remove the branch
#       length_threshold: the total length of the branch must be greater than this point
#   """
#   branches_to_keep = {root_id: skeleton.branches[root_id]}

#   for branch_id, branch in skeleton.branches.items():
    
#     if branch.parent_id == -1:
#       continue
    
#     if branch.parent_id in branches_to_keep:
#       if branch.length > length_threshold and branch.radii[0] > min_radius_threshold:
#         branches_to_keep[branch_id] = branch

#   return TreeSkeleton(skeleton._id, branches_to_keep)