import torch
import frnn

from util.queries import nn_frnn, nn_keops


def recall(gt_points, test_points, gt_radii, thresholds=[0.1]): # recall (completeness)

  results = []
  dist, idx = nn_keops(gt_points, test_points)
  idx = idx.reshape(-1)
  dist = dist.reshape(-1)

  for t in thresholds:

    mask = dist < (gt_radii * t) 
    
    valid_percentage = (torch.sum(mask) / gt_points.shape[0]) * 100
    
    results.append(valid_percentage.item())
  
  return results


def precision(gt_points, test_points, gt_radii, thresholds=[0.1]): # precision (how close)

  results = []
  dist, idx = nn_keops(test_points, gt_points)
  idx = idx.reshape(-1)
  dist = dist.reshape(-1)

  for t in thresholds:
  
    mask = dist < (gt_radii[idx] * t) 

    valid_percentage = (torch.sum(mask) / test_points.shape[0]) * 100
    
    results.append(valid_percentage.item())
  
  return results


# def recall(gt_points, test_points, gt_radii, threshold=1.0):
#   idxs, dists, _ = nn(gt_points, test_points, r=gt_radii.max().item()) 
#   valid_idx = idxs[idxs != -1]
#   valid = (dists[valid_idx] < gt_radii[valid_idx] * threshold) 
#   return (torch.sum(valid).cpu().item() / gt_points.shape[0]) * 100


# def precision(test_points, gt_points, gt_radii, threshold=1.0):
#   idxs, dists, _ = nn(test_points, gt_points, r=gt_radii.max().item()) 
#   valid_idx = idxs[idxs != -1]
#   valid = dists[valid_idx] < gt_radii[valid_idx] * threshold

#   print(torch.sum(torch.isnan(valid.cpu())))
#   return (torch.sum(valid.cpu()).item() / test_points.shape[0]) * 100