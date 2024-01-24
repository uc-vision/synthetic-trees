from __future__ import annotations

import numpy as np
import torch
import time

from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


def flatten_list(l):
    return [item for sublist in l for item in sublist]


@dataclass
class Tube:
    a: np.array
    b: np.array  # Nx3
    r1: float
    r2: float


@dataclass
class CollatedTube:
    a: np.array  # Nx3
    b: np.array  # Nx3
    r1: np.array  # N
    r2: np.array  # N

    def _to_torch(self, device=torch.device("cuda")):
        a = torch.tensor(self.a, device=device)
        b = torch.tensor(self.b, device=device)
        r1 = torch.tensor(self.r1, device=device)
        r2 = torch.tensor(self.r2, device=device)
        return CollatedTube(a, b, r1, r2)


def collate_tubes(tubes: List[Tube]) -> CollatedTube:
    a = np.concatenate([tube.a for tube in tubes]).reshape(-1, 3)
    b = np.concatenate([tube.b for tube in tubes]).reshape(-1, 3)

    r1 = np.asarray([tube.r1 for tube in tubes]).reshape(1, -1)
    r2 = np.asarray([tube.r2 for tube in tubes]).reshape(1, -1)

    return CollatedTube(a, b, r1, r2)


@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]

    def to_tubes(self) -> List[Tube]:
        return flatten_list([branch.to_tubes() for branch in self.branches.values()])


@dataclass
class Cloud:
    xyz: np.array
    rgb: np.array
    class_l: np.array = None
    vector: np.array = None


@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: np.array
    radii: np.array
    child_id: int = -1

    def to_tubes(self) -> List[Tube]:
        a_, b_, r1_, r2_ = (
            self.xyz[:-1],
            self.xyz[1:],
            self.radii[:-1],
            self.radii[1:],
        )

        return [Tube(a, b, r1, r2) for a, b, r1, r2 in zip(a_, b_, r1_, r2_)]


def points_to_collated_tube_projections_gpu(
    pts: np.array,
    collated_tube: CollatedTube,
    device=torch.device("cuda"),
    eps=1e-12,
):
    ab = collated_tube.b - collated_tube.a  # M x 3 -> tube direction

    ap = pts.unsqueeze(1) - collated_tube.a.unsqueeze(0)  # N x M x 3

    t = (
        torch.einsum("nmd,md->nm", ap, ab) / (torch.einsum("md,md->m", ab, ab) + eps)
    ).clip(
        0.0, 1.0
    )  # N x M
    proj = collated_tube.a.unsqueeze(0) + torch.einsum("nm,md->nmd", t, ab)  # N x M x 3

    return proj, t


def projection_to_distance_matrix_gpu(projections, pts):  # N x M x 3
    return (projections - pts.unsqueeze(1)).square().sum(2).sqrt()


def pts_to_nearest_tube_gpu(
    pts: np.array, collated_tube: CollatedTube, device=torch.device("cuda")
):
    """Vectors from pt to the nearest tube"""

    # collated_tube = collate_tubes(tubes)

    projections, t = points_to_collated_tube_projections_gpu(
        pts, collated_tube, device=torch.device("cuda")
    )  # N x M x 3
    r = (1 - t) * collated_tube.r1 + t * collated_tube.r2

    distances = projection_to_distance_matrix_gpu(projections, pts)  # N x M

    distances = distances - r
    idx = torch.argmin(distances, 1)  # N

    return (
        (projections[torch.arange(pts.shape[0]), idx] - pts),
        (idx),
        (r[torch.arange(pts.shape[0]), idx]),
    )


def unpackage_data(data: dict) -> Tuple[Cloud, TreeSkeleton]:
    tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"]
    skeleton_radii = data["skeleton_radii"]
    sizes = data["branch_num_elements"]

    cld = Cloud(
        xyz=data["xyz"],
        rgb=data["rgb"],
        class_l=data["class_l"],
        vector=data["vector"],
    )

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return cld, TreeSkeleton(tree_id, branches)


def load_data_npz(path: Path) -> Tuple[Cloud, TreeSkeleton]:
    return unpackage_data(np.load(str(path)))


if __name__ == "__main__":
    cloud, skeleton = load_data_npz(
        "/local/point_cloud_datasets/synthetic-trees/tree_dataset/branches/pine/pine_15.npz"
    )

    collated_tube = collate_tubes(skeleton.to_tubes())._to_torch()
    pts = torch.from_numpy(cloud.xyz).float().to(torch.device("cuda"))

    start_time = time.time()
    torch.cuda.synchronize()

    for i in range(0, len(pts), 8000):
        batch_pts = pts[i : i + 8000]
        vector, idx, radius = pts_to_nearest_tube_gpu(batch_pts, collated_tube)

    torch.cuda.synchronize()

    print(f"Done {time.time() - start_time}")

    print(vector.shape)
