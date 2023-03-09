import os

from typing import List, Tuple
from pathlib import Path
import argparse

import open3d as o3d


from data_types.tree import TreeSkeleton, repair_skeleton
from data_types.cloud import Cloud

from util.file import load_data_npz
from util.o3d_abstractions import o3d_viewer
from util.operations import sample_o3d_lineset


def evaluate_one(cld: Cloud, skeleton: TreeSkeleton, ls: o3d.cuda.pybind.geometry.LineSet, str, sample_rate=0.01):
      
  skeleton = repair_skeleton(skeleton)
  
  gt_xyzs, gt_radii = skeleton.point_sample(sample_rate)
  
  #evaluation_pts = sample_o3d_lineset(ls_to_evaluate, sample_rate)
  
  #geometries.append(skeleton.to_o3d_lineset())

  #o3d_viewer(geometries)

def evaluate(data: List[Tuple[Cloud, TreeSkeleton]], num_processes=1):
    
    geometries, names = [], []
    
    for (cloud, skeleton), path in data:
        evaluate_one(cloud, skeleton, path.stem) 


def ground_truth_skeleton_generator(paths: List[Path]) -> Tuple[Cloud, TreeSkeleton]:
    for path in paths:
      yield load_data_npz(path)
  



def parse_args():
    
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("-d_gt", "--ground_truth_dir",
                        help="Directory of folder of tree.npz(s) *.npz", default=None, type=str)

    parser.add_argument("-d_o", "--output_dir",
                        help="Directory of folder of skeleton outputs *.ply", default=None, type=str)

    return parser.parse_args()


def main():

    args = parse_args()

    ground_truth_paths = Path(args.ground_truth_dir).glob("*.npz")
    output_paths = Path(args.output_dir).glob("*.ply")

    ground_truth_tree_names = [path.stem for path in ground_truth_paths]
    output_tree_names = [path.stem for path in output_paths]
    
    tree_names = list(set(ground_truth_tree_names).intersection(set(output_tree_names)))
    
    
    for gt_skeleton in ground_truth_skeleton_generator(paths):
          
    
    #print(len(tree_names))
    
    #evaluate(data)


if __name__ == "__main__":
    main()