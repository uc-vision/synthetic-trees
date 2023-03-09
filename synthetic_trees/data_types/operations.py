import numpy as np

from typing import List

from .tube import Tube

# from Prescient_Tree.geometries.conversion import tree_skeleton_as_tubes, branch_skeleton_as_tubes
# from Prescient_Tree.geometries.tube import Tube
# from Prescient_Tree.geometries.branch import BranchSkeleton

# from Prescient_Tree.util.math.maths import np_normalized
# from Prescient_Tree.util.math.queries import pts_to_nearest_tube, pts_to_nearest_tube_keops, skeleton_to_points







def sample_tubes(tubes: List[Tube], spacing):
           
  pts, radius = [], []

  for i, tube in enumerate(tubes):
   
    start = tube.a
    end = tube.b
    
    start_rad = tube.r1
    end_rad = tube.r2
          
    v = end - start 
    length = np.linalg.norm(v)
    
    direction = v / length
    num_points = np.ceil(length / spacing)
    
    if int(num_points) > 0.0:
  
      spaced_points = np.arange(0, float(length),  step=float(length/num_points)).reshape(-1,1)
      lin_radius = np.linspace(start_rad, end_rad, spaced_points.shape[0], dtype=float)
      
      pts.append(start + direction * spaced_points)   
      radius.append(lin_radius)
      
  return  np.concatenate(pts, axis=0), np.concatenate(radius, axis=0)
