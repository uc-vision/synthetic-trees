import numpy as np

from typing import List

from data_types.tube import Tube, CollatedTube, collate_tubes


""" 
For the following :
  N : number of pts
  M : number of tubes
"""
def points_to_collated_tube_projections(pts: np.array, collated_tube: CollatedTube, eps=1e-12): # N x 3, M x 2

  ab = collated_tube.b - collated_tube.a # M x 3

  ap = pts[:, np.newaxis] - collated_tube.a[np.newaxis, ...] # N x M x 3

  t = np.clip(np.einsum('nmd,md->nm', ap, ab) / (np.einsum('md,md->m', ab, ab) + eps), 0.0 , 1.0) # N x M
  proj = collated_tube.a[np.newaxis, ...]  + np.einsum('nm,md->nmd', t, ab) # N x M x 3
  return proj, t


def projection_to_distance_matrix(projections, pts): # N x M x 3
  return np.sqrt(np.sum(np.square(projections - pts[:,np.newaxis,:]), 2)) # N x M


def pts_to_nearest_tube(pts:np.array, tubes: List[Tube]):
  """ Vectors from pt to the nearest tube """
  
  collated_tube = collate_tubes(tubes)
  projections, t = points_to_collated_tube_projections(pts, collated_tube) # N x M x 3

  r = (1 - t) * collated_tube.r1 + t * collated_tube.r2

  distances = projection_to_distance_matrix(projections, pts)   # N x M

  distances = (distances - r)
  idx = np.argmin(distances, 1) # N

  #assert idx.shape[0] == pts.shape[0]

  return projections[np.arange(pts.shape[0]), idx] - pts, idx, r[np.arange(pts.shape[0]), idx] # vector, idx , radius


def pts_on_nearest_tube(pts:np.array, tubes: List[Tube]):
  
  vectors, index, radius = pts_to_nearest_tube(pts, tubes)
  return pts + vectors, radius