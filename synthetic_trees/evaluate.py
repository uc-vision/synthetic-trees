import os

from typing import List, Tuple
from pathlib import Path
import argparse

import open3d as o3d


from data_types.tree import TreeSkeleton, repair_skeleton
from data_types.cloud import Cloud

from util.file import load_data_npz
from util.o3d_abstractions import o3d_viewer, o3d_load_lineset, o3d_nn

from util.operations import sample_o3d_lineset
from util.misc import to_torch


def evaluate_one(gt_skeleton: TreeSkeleton, output_skeleton: o3d.cuda.pybind.geometry.LineSet, sample_rate=0.01):
      
  skeleton = repair_skeleton(gt_skeleton)
  
  gt_xyzs, gt_radii = skeleton.point_sample(sample_rate)
  
  output_pts = sample_o3d_lineset(output_skeleton, sample_rate)
  
  gt_xyzs_c, gt_radii_c, output_pts_c = to_torch([gt_xyzs, gt_radii, output_pts], device=torch.device("cuda"))
  
  o3d_nn(output_pts_c, gt_xyzs_c, gt_radii_c)
  
  #geometries = []
  #geometries.append()

  #o3d_viewer(geometries)


def gt_skeleton_generator(paths: List[Path]) -> Tuple[Cloud, TreeSkeleton]:
    for path in paths:
      yield load_data_npz(path)[1]
  
def output_skeleton_generator(paths: List[Path]) -> o3d.cuda.pybind.geometry.LineSet:
    for path in paths:
      yield o3d_load_lineset(str(path))
  
  
def parse_args():
    
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("-d_gt", "--ground_truth_dir",
                        help="Directory of folder of tree.npz(s) *.npz", default=None, type=str)

    parser.add_argument("-d_o", "--output_dir",
                        help="Directory of folder of skeleton outputs *.ply", default=None, type=str)

    return parser.parse_args()


def main():

    args = parse_args()

    ground_truth_paths = list(Path(args.ground_truth_dir).glob("*.npz"))
    output_paths = list(Path(args.output_dir).glob("*.ply"))

    ground_truth_tree_names = [path.stem for path in ground_truth_paths]
    output_tree_names = [path.stem for path in output_paths]

    tree_names = list(set(ground_truth_tree_names).intersection(set(output_tree_names)))
    
    gt_paths = [p for p in ground_truth_paths if p.stem in tree_names]
    output_paths = [p for p in output_paths if p.stem in tree_names]

    
    for gt_skeleton, output_skeleton in zip(gt_skeleton_generator(gt_paths), output_skeleton_generator(output_paths)):

        evaluate_one(gt_skeleton, output_skeleton)

        print(output_skeleton)
        print(gt_skeleton)
    
          
    
    #evaluate(data)


if __name__ == "__main__":
    main()