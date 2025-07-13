import math
import numpy as np
import open3d as o3d
from sklearn.cluster import MeanShift
import trimesh
from trimesh import Trimesh
from superprimitive_fusion.utils import (
    trimesh_to_o3d,
    o3d_to_trimesh,
    smooth_normals,
    calc_local_spacing,
    find_cyl_neighbours,
    calc_local_spacing,
    compute_overlap_set,
    trilateral_shift,
    get_o3d_colours_from_trimesh,
)
# Main execution
if __name__ == "__main__":
    # Load meshes
    mesh1 = trimesh.load_mesh('notebooks/meshes/bottle_1.ply')
    mesh2 = trimesh.load_mesh('notebooks/meshes/bottle_2.ply')
    points1 = np.asarray(mesh1.vertices)
    points2 = np.asarray(mesh2.vertices)
    points = np.concatenate([points1, points2])

    # Get colours
    colours1 = mesh1.visual.vertex_colors
    colours2 = mesh2.visual.vertex_colors
    colours = np.concat([colours1, colours2])

    # Compute  and smooth normals
    mesh1_o3d = trimesh_to_o3d(mesh1).compute_vertex_normals()
    mesh2_o3d = trimesh_to_o3d(mesh2).compute_vertex_normals()
    
    normals1 = np.asarray(mesh1_o3d.vertex_normals)
    normals2 = np.asarray(mesh2_o3d.vertex_normals)
    normals = np.concatenate([normals1, normals2])
    
    normals = smooth_normals(points, normals)

    # Build KDTree for combined points
    tree = o3d.geometry.KDTreeFlann(points.T)

    # Compute local spacing and density
    local_spacing_1, local_density_1 = calc_local_spacing(points1, points1)
    local_spacing_2, local_density_2 = calc_local_spacing(points2, points2)
    local_spacing = np.concatenate([local_spacing_1, local_spacing_2])
    local_density = np.concatenate([local_density_1, local_density_2])

    # Scan IDs (0 for mesh1, 1 for mesh2)
    scan_ids = np.concatenate([
        np.zeros(len(points1)), np.ones(len(points2))
        ])

    # Detect overlap
    overlap_idx, _ = compute_overlap_set(
        points, normals, local_spacing, scan_ids, 
        h_alpha=2.5, r_alpha=2, tree=tree
    )

    # Trilateral shift
    new_pts = points
    for i in range(3):
        new_pts = trilateral_shift(
            new_pts, normals, local_spacing, local_density,
            overlap_idx, tree, r_alpha=2, h_alpha=2.5
        )

    # Visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_pts)
    pcd.colors = get_o3d_colours_from_trimesh(colours)
    o3d.visualization.draw_geometries([pcd])