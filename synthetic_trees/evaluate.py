import os

from typing import List, Tuple

import argparse
import open3d


from synthetic_trees.data_types.tree import TreeSkeleton, repair_skeleton
from synthetic_trees.data_types.cloud import Cloud
from synthetic_trees.util.file import load_data_npz
from synthetic_trees.util.o3d_abstractions import o3d_viewer


def evaluate_one(cld: Cloud, skeleton: TreeSkeleton, ls, sample_rate=0.01):
      
  skeleton = repair_skeleton(skeleton)
  gt_xyzs, gt_radii = skeleton.point_sample(sample_rate)
  
  #geometries.append(skeleton.to_o3d_lineset())


  #o3d_viewer(geometries)



def evaluate(data: List[Tuple[Cloud, TreeSkeleton]]):
    geometries, names = [], []
    for (cloud, skeleton), path in data:
        evaluate_one(cloud, skeleton, path) 


def parse_args():
    
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("-p", "--file_path",
                        help="File Path of tree.npz", default=None, type=str)
    parser.add_argument("-d", "--directory",
                        help="Directory of folder of tree.npz(s)", default=None, type=str)

    return parser.parse_args()


def main():

    args = parse_args()

    if args.file_path is not None:
        data: List[Tuple[Tuple[Cloud, TreeSkeleton], str]
                   ] = [(load_data_npz(args.file_path), args.file_path)]

    if args.directory is not None:          
        data: List[Tuple[Tuple[Cloud, TreeSkeleton], str]] = [(load_data_npz(os.path.join(args.directory,path)), path)
                                                  for path in os.listdir(args.directory) if path.split(".")[1] =="npz"]

    evaluate(data)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
