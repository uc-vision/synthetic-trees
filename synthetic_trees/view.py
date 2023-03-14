import os
import argparse

from typing import List, Tuple

from pathlib import Path

import open3d
import glob

from synthetic_trees.data_types.tree import TreeSkeleton, repair_skeleton, prune_skeleton
from synthetic_trees.data_types.cloud import Cloud
from synthetic_trees.util.file import load_data_npz
from synthetic_trees.util.o3d_abstractions import o3d_viewer


def view_synthetic_data(data: List[Tuple[Cloud, TreeSkeleton]], line_width=1):
    geometries, names = [], []
    for (cloud, skeleton), path in data:

        tree_name = path.stem

        names.append(f"{tree_name}_cloud")
        names.append(f"{tree_name}_skeleton")
        names.append(f"{tree_name}_skeleton_mesh")

        geometries.append(cloud.to_o3d_cloud())
        geometries.append(skeleton.to_o3d_lineset())
        geometries.append(skeleton.to_o3d_tubes())

    o3d_viewer(geometries, names, line_width=line_width)


def parse_args():

    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("-p", "--file_path",
                        help="File Path of tree.npz", default=None, type=str)
    parser.add_argument("-d", "--directory",
                        help="Directory of folder of tree.npz(s)", default=None, type=str)
    parser.add_argument("-lw", "--line_width",
                        help="Width of visualizer lines", default=1, type=int)
    return parser.parse_args()


def main():

    args = parse_args()

    if args.file_path is not None:
        data: Tuple[Tuple[Cloud, TreeSkeleton], str] = [
            (load_data_npz(args.file_path), Path(args.file_path))]

    if args.directory is not None:
        data: List[Tuple[Tuple[Cloud, TreeSkeleton], str]] = [(load_data_npz(path), path)
                                                              for path in Path(args.directory).glob("*.npz")]

    view_synthetic_data(data, args.line_width)


if __name__ == "__main__":
    main()
