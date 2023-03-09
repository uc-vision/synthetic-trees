import numpy as np

from typing import List

from data_types.tube import Tube


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


def sample_o3d_lineset(ls, sample_rate):
    
  edges = np.asarray(ls.lines)
  xyz = np.asarray(ls.points)
  
  pts, radius = [], []

  for i, edge in enumerate(edges):
   
    start = xyz[edge[0]]
    end = xyz[edge[1]]
              
    v = end - start 
    length = np.linalg.norm(v)
    direction = v / length
    num_points = np.ceil(length / sample_rate)
    
    if int(num_points) > 0.0:

      spaced_points = np.arange(0, float(length),  step=float(length/num_points)).reshape(-1,1)
      pts.append(start + direction * spaced_points)   
      
  return  np.concatenate(pts, axis=0)
  
  