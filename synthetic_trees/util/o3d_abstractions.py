import open3d as o3d
import numpy as np


def o3d_cloud(points, colour=None, colours=None, normals=None):

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    if colour is not None:
        return cloud.paint_uniform_color(colour)
    elif colours is not None:
        cloud.colors = o3d.utility.Vector3dVector(colours)

    return cloud


def o3d_merge_linesets(line_sets, colour=(0, 0, 0)):

    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate(
        [ls.lines + offset for ls, offset in zip(line_sets, offsets)])

    return o3d_line_set(points, idxs).paint_uniform_color(colour)


def o3d_line_set(vertices, edges, colour=None):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(
        vertices), o3d.utility.Vector2iVector(edges))
    if colour is not None:
        return ls.paint_uniform_color(colour)
    return ls


def o3d_path(vertices, colour=None):
    idx = np.arange(vertices.shape[0]-1)
    edge_idx = np.column_stack((idx, idx+1))
    if colour is not None:
        return o3d_line_set(vertices, edge_idx, colour)
    return o3d_line_set(vertices, edge_idx)


def o3d_viewer(items, names=[], line_width=1):

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = line_width

    geometries = []
    if len(names) == 0:
        names = np.arange(0, len(items))

    for name, item in zip(names, items):
        if type(item) == o3d.cuda.pybind.geometry.LineSet:
            geometries.append(
                {"name": f"{name}", "geometry": item, "material": line_mat, "is_visible": False})
        else:
            geometries.append(
                {"name": f"{name}", "geometry": item, "material": mat, "is_visible": False})

    o3d.visualization.draw(geometries, line_width=line_width)
