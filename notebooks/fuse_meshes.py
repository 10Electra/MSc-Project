import numpy as np
import open3d as o3d
import trimesh
from utils import (
    trimesh_to_o3d,
    smooth_normals,
    calc_local_spacing,
    find_cyl_neighbours,
    compute_overlap_set,
    trilateral_shift,
    get_o3d_colours_from_trimesh,
    find_boundary_edges,
    topological_trim,
    merge_nearby_clusters,
)

def mesh_fusion(
        mesh1:trimesh.Trimesh,
        mesh2:trimesh.Trimesh,
    ):
    ### Load meshes and extract data
    mesh1_o3d = trimesh_to_o3d(mesh1)
    mesh2_o3d = trimesh_to_o3d(mesh2)
    points1 = np.asarray(mesh1.vertices)
    points2 = np.asarray(mesh2.vertices)
    pointclouds = (points1, points2)
    points = np.vstack(pointclouds)

    colours1 = mesh1.visual.vertex_colors
    colours2 = mesh2.visual.vertex_colors
    colours = np.concat([colours1, colours2])

    tree = o3d.geometry.KDTreeFlann(points.T)

    mesh1_o3d.compute_vertex_normals()
    mesh2_o3d.compute_vertex_normals()

    normals1 = np.asarray(mesh1_o3d.vertex_normals)
    normals2 = np.asarray(mesh2_o3d.vertex_normals)
    normals = np.concat([normals1, normals2], axis=0)

    scan_ids = np.concat([np.ones(len(these_points)) * i for (i, these_points) in enumerate(pointclouds)])

    ### Smooth normals and calculate local properties
    normals = smooth_normals(points, normals, k=8, T=0.7, n_iters=5)
    local_spacing_1, local_density_1 = calc_local_spacing(mesh1.vertices, np.asarray(mesh1_o3d.vertices))
    local_spacing_2, local_density_2 = calc_local_spacing(mesh2.vertices, np.asarray(mesh2_o3d.vertices))
    local_spacings = (local_spacing_1, local_spacing_2)
    local_spacing = np.concat(local_spacings)

    local_density = np.concat((local_spacing_1, local_spacing_2))

    global_avg_spacing = (1/len(local_spacings)) * np.sum([(1/len(localspacing)) * np.sum(localspacing) for localspacing in local_spacings])

    ### Calculate overlapping region
    h_alpha = 2.5
    r_alpha = 2
    overlap_idx, overlap_mask = compute_overlap_set(points, normals, local_spacing, scan_ids, h_alpha, r_alpha, tree)

    ### Find the edges that should constrain the overlap mesh
    tris1 = np.asarray(mesh1_o3d.triangles)
    tris2 = np.asarray(mesh2_o3d.triangles)
    all_tris = np.concat([tris1, tris2+len(points1)])

    nonoverlap_tris = all_tris[~np.all(overlap_mask[all_tris], axis=1)]
    boundary_edges = find_boundary_edges(nonoverlap_tris)

    ### Perform trilateral point shifting
    trilat_shifted_pts = points
    for i in range(5):
        trilat_shifted_pts = trilateral_shift(trilat_shifted_pts, normals, local_spacing, local_density, overlap_idx, tree, r_alpha, h_alpha)

    ### Merge nearby clusters of points
    cluster_mapping, clustered_overlap_pnts, clustered_overlap_cols, clustered_overlap_nrms = merge_nearby_clusters(
        trilat_shifted_pts=trilat_shifted_pts,
        normals=normals,
        colours=colours,
        overlap_mask=overlap_mask,
        global_avg_spacing=global_avg_spacing,
        h_alpha=h_alpha,
        find_cyl_neighbours=find_cyl_neighbours,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clustered_overlap_pnts)
    pcd.colors = get_o3d_colours_from_trimesh(200*np.ones_like(clustered_overlap_cols))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points[cluster_mapping==-1])
    pcd2.colors = get_o3d_colours_from_trimesh(100*np.zeros((len(points[cluster_mapping==-1]),4)))

    ### Classify vertices and create mapping
    tris1 = np.asarray(mesh1_o3d.triangles)
    tris2 = np.asarray(mesh2_o3d.triangles)

    tris2_shifted = tris2 + len(points1)
    all_tris = np.vstack([tris1, tris2_shifted])

    tri_has_overlap_any = overlap_mask[all_tris].any(axis=1)

    overlap_any_idx = np.unique(
        all_tris[tri_has_overlap_any]
    )

    border_mask = np.zeros(len(points), dtype=bool)
    border_mask[overlap_any_idx] = True # Add both border verts and overlap verts
    border_mask[cluster_mapping!=-1] = False # Remove overlap verts

    nonoverlap_nonborder_mask = np.zeros(len(points), dtype=bool)
    nonoverlap_nonborder_mask[cluster_mapping==-1] = True
    nonoverlap_nonborder_mask[border_mask] = False

    n_overlap, n_border, n_free = len(clustered_overlap_pnts), border_mask.sum(), nonoverlap_nonborder_mask.sum()
    #                        Overlapping             Border with overlap              Not overlapping or border (free)
    new_points  = np.concat([clustered_overlap_pnts, trilat_shifted_pts[border_mask], trilat_shifted_pts[nonoverlap_nonborder_mask]])
    new_colours = np.concat([clustered_overlap_cols, colours[border_mask],            colours[nonoverlap_nonborder_mask]])
    new_normals = np.concat([clustered_overlap_nrms, normals[border_mask],            normals[nonoverlap_nonborder_mask]])

    ### Complete mapping to include border and free vertices
    border_idx_from  = np.array(range(len(points)))[border_mask]
    border_idx_to    = np.array(range(n_border)) + n_overlap

    free_idx_from    = np.array(range(len(points)))[nonoverlap_nonborder_mask]
    free_idx_to      = np.array(range(n_free)) + n_overlap + n_border

    mapping = cluster_mapping # contains mappings for overlap area already
    mapping[border_idx_from] = border_idx_to
    mapping[free_idx_from]   = free_idx_to

    ### Mesh the overlap zone
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(new_points[:n_overlap])
    pcd.colors = get_o3d_colours_from_trimesh(new_colours[:n_overlap])

    if new_normals[:n_overlap] is not None:
        pcd.normals = o3d.utility.Vector3dVector(new_normals[:n_overlap])
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=2.5 * global_avg_spacing,  # ~10–30 neighbours
                                max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

    ball_r   = 1.1 * global_avg_spacing
    radii    = o3d.utility.DoubleVector([ball_r, ball_r * 1.5])

    overlap_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii)

    overlap_mesh.remove_duplicated_vertices()
    overlap_mesh.remove_duplicated_triangles()
    overlap_mesh.remove_degenerate_triangles()
    overlap_mesh.remove_non_manifold_edges()
    overlap_mesh.compute_vertex_normals()

    overlap_mesh.vertex_colors = pcd.colors
    ### Debug plot: overlap mesh, border verts, free mesh
    v = np.asarray(overlap_mesh.vertices)
    vidx = np.unique(mapping[boundary_edges])
    vidx = vidx[vidx < len(v)]

    ### Trim overlap mesh to the boundary edge loop
    mapped_boundary_edges = mapping[boundary_edges]
    relevant_boundary_edges = mapped_boundary_edges[np.all(mapped_boundary_edges<len(v), axis=1)]

    trimmed_overlap_mesh = topological_trim(overlap_mesh, relevant_boundary_edges)

    trimmed_overlap_tris = np.asarray(trimmed_overlap_mesh.triangles)
    fused_mesh_triangles = np.concatenate([trimmed_overlap_tris, mapping[nonoverlap_tris]], axis=0)
    fused_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(new_points),
        triangles=o3d.utility.Vector3iVector(fused_mesh_triangles)
    )

    fused_mesh.remove_unreferenced_vertices()
    fused_mesh.remove_duplicated_triangles()
    fused_mesh.remove_duplicated_vertices()
    fused_mesh.remove_degenerate_triangles()
    fused_mesh.remove_non_manifold_edges()

    new_colours = np.clip(new_colours, 0, 255)
    fused_mesh.vertex_colors = get_o3d_colours_from_trimesh(new_colours)

    fused_mesh.compute_vertex_normals()

    return fused_mesh