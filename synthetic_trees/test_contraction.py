import argparse
from dataclasses import asdict

from pathlib import Path
import torch

from synthetic_trees.data_types.tube import collate_tubes
from synthetic_trees.util.file import load_data_npz

import geometry_grid.torch_geometry as torch_geom
from geometry_grid.taichi_geometry.grid import  Grid, morton_sort

from geometry_grid.taichi_geometry.dynamic_grid import DynamicGrid
from geometry_grid.taichi_geometry.counted_grid import CountedGrid

from geometry_grid.taichi_geometry.point_query import point_query
from geometry_grid.taichi_geometry.attract_query import attract_query
from geometry_grid.taichi_geometry.point_distances import batch_distances_grad


from geometry_grid.render_util import display_distances
from open3d_vis import render
import open3d as o3d


import taichi as ti


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument("file_path",
                        help="File Path of tree.npz",type=Path)
    parser.add_argument("--debug", action="store_true",
                        help="Enable taichi debug mode")
    
    parser.add_argument("--device", default="cuda:0", help="Device to run on")

    return parser.parse_args()

def display_vectors(points, v):
  o3d.visualization.draw(
      [render.segments(points, points + v, color=(1, 0, 0)),
        render.point_cloud(points, color=(0, 0, 1))],

      point_size=6
  )


def main():
    args = parse_args()

    ti.init(arch=ti.gpu, debug=args.debug, log_level=ti.DEBUG)

    cloud, skeleton = load_data_npz(args.file_path)
    # view_synthetic_data([(data, args.file_path)])


    device = torch.device(args.device)
    np_tubes = collate_tubes(skeleton.to_tubes())

    tubes = {k:torch.from_numpy(x).to(dtype=torch.float32, device=device) for k, x in asdict(np_tubes).items()}

    segments = torch_geom.Segment(tubes['a'], tubes['b'])
    radii = torch.concatenate((tubes['r1'], tubes['r2']), -1)

    tubes = torch_geom.Tube(segments, radii)
    bounds = tubes.bounds.union_all()

    points = torch.from_numpy(cloud.xyz).to(dtype=torch.float32, device=device)
    points = morton_sort(points, n=256)


    print("Generate grid...")
    seg_grid = CountedGrid.from_torch(
        Grid.fixed_size(bounds, (16, 16, 16)), segments)


    point_grid = DynamicGrid.from_torch(
        Grid.fixed_size(bounds, (64, 64, 64)), torch_geom.Point(points))


    forces = attract_query(point_grid.index, points, 
                          sigma=0.1, max_distance=0.1)
    
    # reg = forces * 0.001

    dist, idx = point_query(seg_grid, points, 0.5)
    point_grad, _ = batch_distances_grad(segments[idx], points, dist, dist.clone())


    display_vectors(points, point_grad)
    
    # print("Grid size: ", seg_grid.grid.size)
    # cells, counts = seg_grid.active_cells()
  
    # max_dist = dist[torch.isfinite(dist)].max()

    # display_distances(tubes, seg_grid.grid.get_boxes(cells), 
    #                 points, dist / max_dist )
    



if __name__ == "__main__":
    main()
