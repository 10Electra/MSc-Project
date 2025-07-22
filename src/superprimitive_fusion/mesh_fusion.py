import numpy as np
import open3d as o3d  # type: ignore

from superprimitive_fusion.mesh_fusion_utils import (
    smooth_normals,
    calc_local_spacing,
    compute_overlap_set_cached,
    smooth_overlap_set_cached,
    precompute_cyl_neighbours,
    trilateral_shift_cached,
    find_boundary_edges,
    topological_trim,
    merge_nearby_clusters,
)

from line_profiler import profile
def flip_if_inwards(mesh: o3d.geometry.TriangleMesh):
    mesh.compute_triangle_normals()
    c_mesh   = mesh.get_center()
    tris     = np.asarray(mesh.triangles)
    verts    = np.asarray(mesh.vertices)
    tri_ctr  = verts[tris].mean(axis=1)              # (F,3)
    tri_norm = np.asarray(mesh.triangle_normals)     # (F,3)
    score    = np.sum((tri_ctr - c_mesh) * tri_norm) # signed “flux”

    if score < 0:                                    # pointing inside
        mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
        mesh.compute_vertex_normals()
    return mesh

@profile
def fuse_meshes(
    mesh1: o3d.geometry.TriangleMesh,
    mesh2: o3d.geometry.TriangleMesh,
    h_alpha: float = 2.5,
    r_alpha: float = 2.0,
    trilat_iters: int = 2,
    nrm_smth_iters: int = 1,
    shift_all: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Fuses two registered open3d triangle meshes.

    Args:
        mesh1 (o3d.geometry.TriangleMesh): first mesh to fuse
        mesh2 (o3d.geometry.TriangleMesh): second mesh to fuse
        h_alpha (float, optional): h_alpha * local_spacing = cylinder search region height. Defaults to 2.5.
        r_alpha (float, optional): r_alpha * local_spacing = cylinder search region radius. Defaults to 2.0.

    Returns:
        o3d.geometry.TriangleMesh: output mesh
    """

    # ---------------------------------------------------------------------
    # Raw geometry & attribute extraction
    # ---------------------------------------------------------------------
    points1 = np.asarray(mesh1.vertices)
    points2 = np.asarray(mesh2.vertices)

    pointclouds = (points1, points2)
    points = np.vstack(pointclouds)

    colours1 = mesh1.vertex_colors
    colours2 = mesh2.vertex_colors
    colours = np.concatenate([colours1, colours2], axis=0)

    kd_tree = o3d.geometry.KDTreeFlann(points.T)

    # ---------------------------------------------------------------------
    # Normals
    # ---------------------------------------------------------------------
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()

    normals1 = np.asarray(mesh1.vertex_normals)
    normals2 = np.asarray(mesh2.vertex_normals)
    normals = np.concatenate([normals1, normals2], axis=0)

    scan_ids = np.concatenate([np.full(len(pts), i) for i, pts in enumerate(pointclouds)])

    normals = smooth_normals(points, normals, tree=kd_tree, k=8, T=0.7, n_iters=nrm_smth_iters)

    # ---------------------------------------------------------------------
    # Local geometric properties
    # ---------------------------------------------------------------------
    local_spacing_1, local_density_1 = calc_local_spacing(points1, points1, tree=kd_tree)
    local_spacing_2, local_density_2 = calc_local_spacing(points2, points2, tree=kd_tree)
    local_spacings = (local_spacing_1, local_spacing_2)

    local_spacing = np.concatenate(local_spacings)
    local_density = np.concatenate((local_density_1, local_density_2))

    global_avg_spacing = (1 / len(local_spacings)) * np.sum(
        [(1 / len(ls)) * np.sum(ls) for ls in local_spacings]
    )

    # ---------------------------------------------------------------------
    # Overlap detection
    # ---------------------------------------------------------------------
    nbr_cache = precompute_cyl_neighbours(points, normals, local_spacing, r_alpha, h_alpha, kd_tree)

    overlap_idx, overlap_mask = compute_overlap_set_cached(scan_ids, nbr_cache)
    overlap_idx, overlap_mask = smooth_overlap_set_cached(overlap_mask, nbr_cache)

    # ---------------------------------------------------------------------
    # Find overlap boundary edges
    # ---------------------------------------------------------------------
    tris1 = np.asarray(mesh1.triangles)
    tris2 = np.asarray(mesh2.triangles) + len(points1)  # shift indices
    all_tris = np.concatenate([tris1, tris2], axis=0)

    nonoverlap_tris = all_tris[~np.all(overlap_mask[all_tris], axis=1)]
    boundary_edges = find_boundary_edges(nonoverlap_tris)

    # ---------------------------------------------------------------------
    # Trilateral point shifting
    # ---------------------------------------------------------------------
    trilat_shifted_pts = points.copy()
    for _ in range(trilat_iters):
        trilat_shifted_pts = trilateral_shift_cached(trilat_shifted_pts, normals, local_spacing, local_density, overlap_idx, nbr_cache, r_alpha, h_alpha, shift_all)
    
    # ---------------------------------------------------------------------
    # Merge nearby clusters
    # ---------------------------------------------------------------------
    cluster_mapping, clustered_overlap_pnts, clustered_overlap_cols, clustered_overlap_nrms = merge_nearby_clusters(
        trilat_shifted_pts=trilat_shifted_pts,
        normals=normals,
        colours=colours,
        overlap_mask=overlap_mask,
        overlap_idx=overlap_idx,
        global_avg_spacing=global_avg_spacing,
        h_alpha=h_alpha,
        tree=kd_tree,
    )

    # ---------------------------------------------------------------------
    # Classify vertices
    # ---------------------------------------------------------------------
    tri_has_overlap_any = overlap_mask[all_tris].any(axis=1)
    overlap_any_idx = np.unique(all_tris[tri_has_overlap_any])

    border_mask = np.zeros(len(points), dtype=bool)
    border_mask[overlap_any_idx] = True
    border_mask[cluster_mapping != -1] = False

    nonoverlap_nonborder_mask = np.zeros(len(points), dtype=bool)
    nonoverlap_nonborder_mask[cluster_mapping == -1] = True
    nonoverlap_nonborder_mask[border_mask] = False

    n_overlap = len(clustered_overlap_pnts)
    n_border = border_mask.sum()
    n_free = nonoverlap_nonborder_mask.sum()

    new_points = np.concatenate(
        [
            clustered_overlap_pnts,
            trilat_shifted_pts[border_mask],
            trilat_shifted_pts[nonoverlap_nonborder_mask],
        ],
        axis=0,
    )
    new_colours = np.concatenate(
        [
            clustered_overlap_cols,
            colours[border_mask],
            colours[nonoverlap_nonborder_mask],
        ],
        axis=0,
    )
    new_normals = np.concatenate(
        [
            clustered_overlap_nrms,
            normals[border_mask],
            normals[nonoverlap_nonborder_mask],
        ],
        axis=0,
    )
    
    new_colours = np.clip(new_colours, 0, 1)
    
    # ---------------------------------------------------------------------
    # Complete mapping
    # ---------------------------------------------------------------------
    border_idx_from = np.arange(len(points))[border_mask]
    border_idx_to = np.arange(n_border) + n_overlap

    free_idx_from = np.arange(len(points))[nonoverlap_nonborder_mask]
    free_idx_to = np.arange(n_free) + n_overlap + n_border

    mapping = cluster_mapping.copy()
    mapping[border_idx_from] = border_idx_to
    mapping[free_idx_from] = free_idx_to

    # ---------------------------------------------------------------------
    # Mesh the overlap zone
    # ---------------------------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_points[:n_overlap])

    if new_normals[:n_overlap] is not None:
        pcd.normals = o3d.utility.Vector3dVector(new_normals[:n_overlap])
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.5 * global_avg_spacing, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

    r_min = 1 * global_avg_spacing
    radii = o3d.utility.DoubleVector(np.geomspace(r_min, r_min*4, num=5))

    overlap_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    overlap_mesh.remove_duplicated_vertices()
    overlap_mesh.remove_duplicated_triangles()
    overlap_mesh.remove_degenerate_triangles()
    overlap_mesh.remove_non_manifold_edges()
    overlap_mesh.compute_vertex_normals()

    overlap_mesh.vertex_colors = pcd.colors

    # ---------------------------------------------------------------------
    # Trim overlap mesh
    # ---------------------------------------------------------------------
    mapped_boundary_edges = mapping[boundary_edges]
    relevant_boundary_edges = mapped_boundary_edges[
        np.all(mapped_boundary_edges < len(overlap_mesh.vertices), axis=1)
    ]

    trimmed_overlap_mesh = topological_trim(
        overlap_mesh, relevant_boundary_edges
    )

    trimmed_overlap_tris = np.asarray(trimmed_overlap_mesh.triangles)
    fused_mesh_triangles = np.concatenate(
        [trimmed_overlap_tris, mapping[nonoverlap_tris]], axis=0
    )

    fused_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_points),
        triangles=o3d.utility.Vector3iVector(fused_mesh_triangles),
    )

    fused_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colours)

    fused_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(fused_mesh)
    fused_mesh_filled_t = fused_mesh_t.fill_holes(hole_size=r_min*10)
    fused_mesh = fused_mesh_filled_t.to_legacy()
    
    fused_mesh.orient_triangles()
    fused_mesh = flip_if_inwards(fused_mesh)

    fused_mesh.remove_unreferenced_vertices()
    fused_mesh.remove_duplicated_triangles()
    fused_mesh.remove_duplicated_vertices()
    fused_mesh.remove_degenerate_triangles()
    fused_mesh.remove_non_manifold_edges()
    fused_mesh.compute_vertex_normals()

    return fused_mesh


if __name__ == "__main__":
    mesh1 = o3d.io.read_triangle_mesh("./notebooks/meshes/bottle_1.ply")
    mesh2 = o3d.io.read_triangle_mesh("./notebooks/meshes/bottle_2.ply")

    fused_mesh = fuse_meshes(mesh1, mesh2, h_alpha=3)
    
    # o3d.visualization.draw_geometries([fused_mesh])