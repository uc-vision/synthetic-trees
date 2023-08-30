import argparse
from pathlib import Path

from synthetic_trees.util.file import load_data_npz
import open3d as o3d




def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("file_path",
                        help="File Path of tree.npz",type=Path)
    parser.add_argument("output_file",
                        help="File path to write cloud", type=Path)    
    return parser.parse_args()



def main():

    args = parse_args()
    assert args.file_path.exists(), f"File {args.file_path} does not exist"

    cloud, skeleton = load_data_npz(args.file_path)
    o3d.t.io.write_point_cloud(str(args.output_file), cloud.to_tensor_cloud())
        


if __name__ == "__main__":
    main()
