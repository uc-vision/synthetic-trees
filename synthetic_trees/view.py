import os
import argparse
import argcomplete

from typing import List, Tuple

import open3d


from synthetic_trees.data_types.tree import TreeSkeleton
from synthetic_trees.data_types.cloud import Cloud
from synthetic_trees.util.file import load_data_npz
from synthetic_trees.util.o3d_abstractions import o3d_viewer


def view_synthetic_data(data: List[Tuple[Cloud, TreeSkeleton]]):
    geometries, names = [], []
    for (cloud, skeleton), path in data:
          
        tree_name = path.split(".")[0]
        
        names.append(f"{tree_name}_cloud")
        names.append(f"{tree_name}_skeleton")

        geometries.append(cloud.to_o3d_cloud())
        geometries.append(skeleton.to_o3d_lineset())

    o3d_viewer(geometries, names)


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
                   ] = [load_data_npz(args.file_path), args.file_path]

    if args.directory is not None:          
        data: List[Tuple[Tuple[Cloud, TreeSkeleton], str]] = [(load_data_npz(os.path.join(args.directory,path)), path)
                                                  for path in os.listdir(args.directory) if path.split(".")[1] =="npz"]

    view_synthetic_data(data)


if __name__ == "__main__":
    main()
