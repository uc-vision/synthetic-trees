import os
import argparse

from typing import List, Tuple

from pathlib import Path

import open3d
import glob

from synthetic_trees.util.file import load_o3d_cloud
from synthetic_trees.util.o3d_abstractions import o3d_viewer


def view_clouds(data: list):
    geometries, names = [], []

    for cloud, path in data:

        name = path.stem
        names.append(f"{name}")
        geometries.append(cloud)

    o3d_viewer(geometries, names)


def parse_args():

    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("-p", "--file_path",
                        help="File Path of clouds", default=None, type=str)
    parser.add_argument("-d", "--directory",
                        help="Directory of folder of clouds", default=None, type=str)
    return parser.parse_args()


def main():

    args = parse_args()

    if args.file_path is not None:
        data = [(load_o3d_cloud(
            args.file_path), Path(args.file_path))]

    if args.directory is not None:
        data = [(load_o3d_cloud(str(path.as_posix())), path)
                for path in Path(args.directory).glob('*.ply')]

    view_clouds(data)


if __name__ == "__main__":
    main()
