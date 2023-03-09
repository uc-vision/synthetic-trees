import torch

def recall(gt_points, test_points, gt_radii, thresholds=[0.1]): # recall (completeness)

  results = []
  dist, idx = nn_keops(gt_points, test_points)
  idx = idx.reshape(-1)
  dist = dist.reshape(-1)

  for t in thresholds:

    mask = dist < (gt_radii * t) 
    
    valid_percentage = (torch.sum(mask) / gt_points.shape[0]) * 100
    
    results.append(valid_percentage.item())
  
  return torch.tensor(results)


def precision(gt_points, test_points, gt_radii, thresholds=[0.1]): # precision (how close / accurate)

  results = []
  dist, idx = nn_keops(test_points, gt_points)
  idx = idx.reshape(-1)
  dist = dist.reshape(-1)

  for t in thresholds:
  
    mask = dist < (gt_radii[idx] * t) 

    valid_percentage = (torch.sum(mask) / test_points.shape[0]) * 100
    
    results.append(valid_percentage.item())
  
  return torch.tensor(results)
