import numpy as np
import open3d as o3d  # type: ignore
import pymeshfix # type: ignore
from scipy.spatial import cKDTree # type: ignore

from superprimitive_fusion.mesh_fusion_utils import (
    smooth_normals,
    calc_local_spacing,
    compute_overlap_set_cached,
    smooth_overlap_set_cached,
    precompute_cyl_neighbours,
    update_weights,
    normal_shift_smooth,
    find_boundary_edges,
    topological_trim,
    merge_nearby_clusters,
    compact_by_faces,
    sanitise_mesh,
    colour_transfer,
)

from line_profiler import profile

@profile
def fuse_meshes(
    mesh1: o3d.geometry.TriangleMesh,
    weights1: np.ndarray,
    mesh2: o3d.geometry.TriangleMesh,
    weights2: np.ndarray,
    h_alpha: float = 2.5,
    r_alpha: float = 2.0,
    nrm_shift_iters: int = 2,
    nrm_smth_iters: int = 1,
    sigma_theta: float = 0.2,
    normal_diff_thresh: float = 45.0,
    shift_all: bool = False,
    fill_holes: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Fuses two registered open3d triangle meshes.

    Args:
        mesh1 (o3d.geometry.TriangleMesh): first mesh to fuse
        mesh2 (o3d.geometry.TriangleMesh): second mesh to fuse
        weights(1|2) (np.ndarray): weight (==1/sigma_z^2) for each vertex. (N,)
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
    
    assert weights1.ndim == 1 and len(weights1) == len(mesh1.vertices)
    assert weights2.ndim == 1 and len(weights2) == len(mesh2.vertices)
    weights = np.concatenate((weights1, weights2))

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
    overlap_idx, overlap_mask = smooth_overlap_set_cached(overlap_mask, nbr_cache, p_thresh=1)

    # ---------------------------------------------------------------------
    # Find overlap boundary edges
    # ---------------------------------------------------------------------
    tris1 = np.asarray(mesh1.triangles)
    tris2 = np.asarray(mesh2.triangles) + len(points1)  # shift indices
    all_tris = np.concatenate([tris1, tris2], axis=0)

    nonoverlap_tris = all_tris[~np.all(overlap_mask[all_tris], axis=1)]
    boundary_edges = find_boundary_edges(nonoverlap_tris)

    # ---------------------------------------------------------------------
    # Update weights
    # ---------------------------------------------------------------------
    updated_weights = update_weights(
        points,
        normals,
        weights,
        overlap_mask,
        scan_ids,
        nbr_cache,
        normal_diff_thresh=45.0,
        huber_delta=1.345,
        tau_max=None,
    )

    # ---------------------------------------------------------------------
    # Multilateral point shifting along normals
    # ---------------------------------------------------------------------
    normal_shifted_points = points.copy()
    for _ in range(nrm_shift_iters):
        normal_shifted_points = normal_shift_smooth(normal_shifted_points, normals, weights, local_spacing, local_density, overlap_idx, nbr_cache, r_alpha, h_alpha, sigma_theta, normal_diff_thresh, shift_all)
    
    kd_tree = o3d.geometry.KDTreeFlann(normal_shifted_points.T)

    # ---------------------------------------------------------------------
    # Merge nearby clusters
    # ---------------------------------------------------------------------
    merged_out = merge_nearby_clusters(
        normal_shifted_points=normal_shifted_points,
        normals=normals,
        weights=updated_weights,
        colours=colours,
        overlap_mask=overlap_mask,
        overlap_idx=overlap_idx,
        global_avg_spacing=global_avg_spacing,
        h_alpha=h_alpha,
        tree=kd_tree,
    )
    
    # Unpack merge output
    cluster_mapping, clustered_overlap_pnts, clustered_overlap_cols, clustered_overlap_nrms, clustered_overlap_wts = merged_out

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
            normal_shifted_points[border_mask],
            normal_shifted_points[nonoverlap_nonborder_mask],
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
    new_weights = np.concatenate(
        [
            clustered_overlap_wts,
            weights[border_mask],
            weights[nonoverlap_nonborder_mask],
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
    def density_aware_radii(pcd: o3d.geometry.PointCloud, k=10):
        pts = np.asarray(pcd.points)
        kdt = o3d.geometry.KDTreeFlann(pcd)
        dk = np.empty(len(pts))
        for i, p in enumerate(pts):
            _, _, d2 = kdt.search_knn_vector_3d(p, k+1)  # includes the point itself
            dk[i] = np.sqrt(d2[-1])                      # k-th neighbor distance
        h10, h50, h90 = np.percentile(dk, [10, 50, 90])
        r_min = 0.8 * h10
        r_mid = h50
        r_max = 1.2 * h90
        return [r_min, r_mid, r_max]

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

    radii = o3d.utility.DoubleVector(density_aware_radii(pcd, k=10))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(40))
    pcd.orient_normals_consistent_tangent_plane(50)
    
    # points_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # dists = points_pcd.compute_nearest_neighbor_distance()
    # s = np.median(dists)

    # radii = o3d.utility.DoubleVector([1.2*s, 2*s, 3*s, 4*s])

    # r_min = global_avg_spacing
    # radii = o3d.utility.DoubleVector(np.geomspace(r_min, r_min*4, num=5))

    overlap_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    overlap_mesh.remove_duplicated_triangles()
    overlap_mesh.remove_degenerate_triangles()
    overlap_mesh.remove_non_manifold_edges()
    overlap_mesh.compute_vertex_normals()

    # ---------------------------------------------------------------------
    # Trim overlap mesh
    # ---------------------------------------------------------------------
    mapped_boundary_edges = mapping[boundary_edges]
    relevant_boundary_edges = mapped_boundary_edges[
        np.all(mapped_boundary_edges < len(overlap_mesh.vertices), axis=1)
    ]

    Ncc = len(overlap_mesh.cluster_connected_triangles()[1])

    trimmed_overlap_mesh = topological_trim(
        overlap_mesh, relevant_boundary_edges, k=Ncc,
    )

    # ---------------------------------------------------------------------
    # Concatenate trimmed overlap mesh and nonoverlap meshes
    # ---------------------------------------------------------------------
    trimmed_overlap_tris = np.asarray(trimmed_overlap_mesh.triangles)
    fused_mesh_triangles = np.concatenate(
        [trimmed_overlap_tris, mapping[nonoverlap_tris]], axis=0
    )

    V0, F0, (C0, N0, W0), used_mask, remap = compact_by_faces(
        new_points, fused_mesh_triangles, [new_colours, new_normals, new_weights]
    )

    fused_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V0),
        triangles=o3d.utility.Vector3iVector(F0),
    )

    fused_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C0, 0, 1))

    # ---------------------------------------------------------------------
    # Clean up fused mesh
    # ---------------------------------------------------------------------
    # fused_mesh.remove_unreferenced_vertices()
    # fused_mesh.remove_duplicated_vertices()
    fused_mesh.remove_duplicated_triangles()
    fused_mesh.remove_degenerate_triangles()
    fused_mesh.remove_non_manifold_edges()

    if not fill_holes:
        fused_mesh.compute_vertex_normals()
        return fused_mesh, W0
    
    print('Hole-filling is not working at the moment.')
    raise NotImplementedError

    V0, F0 = sanitise_mesh(new_points, fused_mesh_triangles)
    C0 = new_colours

    mf = pymeshfix.PyTMesh(False)
    mf.load_array(V0, F0)

    mf.fill_small_boundaries(nbe=20)

    V1, F1 = mf.return_arrays()

    C1 = colour_transfer(V0, C0, V1)

    repaired = o3d.geometry.TriangleMesh()
    repaired.vertices = o3d.utility.Vector3dVector(V1)
    repaired.triangles = o3d.utility.Vector3iVector(F1)
    repaired.vertex_colors = o3d.utility.Vector3dVector(C1)
    repaired.compute_vertex_normals()
    
    return repaired, W0


if __name__ == "__main__":
    mesh1 = o3d.io.read_triangle_mesh("./notebooks/meshes/bottle_1.ply")
    mesh2 = o3d.io.read_triangle_mesh("./notebooks/meshes/bottle_2.ply")

    fused_mesh = fuse_meshes(mesh1, mesh2, h_alpha=3)
    
    # o3d.visualization.draw_geometries([fused_mesh])