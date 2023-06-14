import argparse
from pathlib import Path
from synthetic_trees.view import parse_args


from synthetic_trees.util.file import load_o3d_cloud
from synthetic_trees.util.o3d_abstractions import ViewerItem, o3d_viewer
from synthetic_trees.view import paths_from_args


def view_clouds(data: list):
    geometries = [ViewerItem(f"{path.stem}_cloud", cloud, visible=i == 0) 
                  for i, (cloud, path) in enumerate(data)]

    o3d_viewer(geometries)


def main():
    args = parse_args()
    data = [(load_o3d_cloud(filename), filename) 
            for filename in paths_from_args(args, '*.ply')]

    view_clouds(data)


if __name__ == "__main__":
    main()
