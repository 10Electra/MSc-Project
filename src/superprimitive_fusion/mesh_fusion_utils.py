import math
import numpy as np
import open3d as o3d # type: ignore
from typing import Tuple, Union
import copy
from collections import deque, defaultdict


def smooth_normals(points: np.ndarray,
                   normals: np.ndarray,
                   tree: o3d.geometry.KDTreeFlann = None,
                   k:int = 8,
                   T:float = 0.7,
                   n_iters:int = 5
) -> np.ndarray:
    """
    Smooth normals using a weighted average of k nearest neighbors.

    Args:
        points (np.ndarray): (N, 3) array of point coordinates.
        normals (np.ndarray): (N, 3) array of normal vectors (assumed unit length).
        k (int): Number of neighbors to use.
        T (float): Threshold for dot product weighting.
        n_iters (int): Number of smoothing iterations.

    Returns:
        np.ndarray: Smoothed normals, shape (N, 3).
    """
    
    def f(x): return x * x

    points = np.asarray(points)
    normals = np.asarray(normals)
    if tree is None:
        tree = o3d.geometry.KDTreeFlann(points.T)

    for _ in range(n_iters):
        new_normals = np.empty_like(normals)
        for i, p in enumerate(points):
            _, idx, _ = tree.search_knn_vector_3d(p, k + 1)  # returns self first
            idx = np.asarray(idx[1:], dtype=int)

            neigh = normals[idx]
            dots = neigh @ normals[i]
            w = f(np.maximum(0.0, dots - T))

            total = (w[:, None] * neigh).sum(0)
            nrm = np.linalg.norm(total)
            new_normals[i] = total / nrm if nrm > 0 else normals[i]

        normals = new_normals  # feed back for next sweep

    return normals


def calc_local_spacing(
        query: np.ndarray,
        points: Union[o3d.geometry.PointCloud, np.ndarray],
        k: int = 25,
        tree: o3d.geometry.KDTreeFlann = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate local sample spacing D(p_j) around each of the query points p_j.

    Args:
        query (np.ndarray): array of query points
        points (np.ndarray): array of all points in the pointcloud
        k (int, optional): number of nearest neighbours to find. Defaults to 25.
        tree (o3d.geometry.KDTreeFlann, optional): tree to seach for nearest neighbours. Defaults to None.

    Raises:
        ValueError: points must be a (N,3) array
        ValueError: query must be a (N,3) array
        ValueError: there must be at least k neighbours in the pointcloud

    Returns:
        Tuple[np.ndarray, np.ndarray]: local sample spacings, local sample densities
    """
    if not (isinstance(points, o3d.geometry.PointCloud) or isinstance(points, np.ndarray)):
        raise TypeError("points must be either an o3d pointcloud or a numpy ndarray")
    if isinstance(points, np.ndarray) and (points.ndim != 2 or points.shape[1] != 3):
        raise ValueError("points must be a (N,3) array")
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query must be a (N,3) array")

    if tree is None:
        tree = o3d.geometry.KDTreeFlann(points.T)

    q = query.reshape(-1, 3)
    n_query = q.shape[0]

    D = np.empty(n_query)
    rho = np.empty(n_query)

    for i, p in enumerate(q):
        # k + 1 because the first hit is p itself with distance=0
        _, idx, d2 = tree.search_knn_vector_3d(p, k + 1)

        if len(idx) < k + 1:
            raise ValueError(f"Found only {len(idx)-1} neighbours; "
                             f"increase point cloud size or reduce k.")

        r_k2 = d2[-1]               # Squared distance to k-th real neighbour
        rho[i] = k / (np.pi * r_k2)
        D[i]  = 1.0 / np.sqrt(rho[i])

    # Return scalars if a single point was given
    if query.ndim == 1 or query.shape[0] == 1:
        return D[0], rho[0]
    return D, rho


def find_cyl_neighbours(
    point: np.ndarray,
    normal: np.ndarray,
    local_spacing: float,
    h_alpha: float,
    r_alpha: float,
    points: np.ndarray,
    tree: o3d.geometry.KDTreeFlann,
    self_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find neighbouring points within a cylindrical region around a point.

    Args:
        point (np.ndarray): The query point (3,).
        normal (np.ndarray): The normal at the query point (3,).
        local_sampling_spacing (float): Local sampling spacing at the query point.
        alpha (float): Scaling factor for the cylinder; r = h/2 = alpha*LSS
        points (np.ndarray): All points in the cloud (N, 3).
        tree (o3d.geometry.KDTreeFlann): KDTree for the point cloud.
        self_idx: id of the query point - removed from results if given

    Returns:
        Tuple[np.ndarray, np.array]: Indices of neighbouring points.
    """
    r_c     = r_alpha * local_spacing
    h_half  = h_alpha * local_spacing
    R       = math.hypot(r_c, h_half)

    N, idx, d2 = tree.search_radius_vector_3d(point, R)
    idx  = np.asarray(idx, dtype=np.int32)
    d2  = np.asarray(d2, dtype=np.float64)
    pts  = points[idx]
    diff = pts - point

    h     = np.abs(diff @ normal)                      # axial dist
    rad2  = np.maximum(d2 - h*h, 0.0)                  # radial dist^2; clamp negatives

    # Build mask in squared space; remove query point by index if given
    mask  = (h <= h_half) & (rad2 <= r_c*r_c)
    if self_idx is not None:
        mask &= (idx != self_idx)

    return idx[mask], d2[mask]


def precompute_cyl_neighbours(
        points, normals, local_spacing,
        r_alpha, h_alpha, tree):
    """Returns list[ndarray] of neighbour indices for every point."""
    nbr_lists = []

    for i, (pnt, nrm, l_sp) in enumerate(zip(points, normals, local_spacing)):
        idx, _ = find_cyl_neighbours(pnt, nrm, l_sp, h_alpha, r_alpha, points, tree, i)
        nbr_lists.append(np.asarray(idx[1:], dtype=np.int32))  # drop self
    return nbr_lists


def compute_overlap_set(
    points:          np.ndarray,    # (N,3)
    normals:         np.ndarray,    # (N,3)
    local_spacing:   np.ndarray,    # (N,)
    scan_ids:        np.ndarray,    # (N,)
    h_alpha:           float,
    r_alpha:           float,
    tree:            o3d.geometry.KDTreeFlann,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        overlap_idx: 1-D array of indices belonging to the overlap set
        overlap_mask:   boolean mask of length N (True -> point in overlap set)
    """
    N = points.shape[0]
    overlap_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if overlap_mask[i]: # already decided via a neighbour
            continue

        neighbour_idx, _ = find_cyl_neighbours(
            point   = points[i],
            normal  = normals[i],
            local_spacing = local_spacing[i],
            h_alpha   = h_alpha,
            r_alpha   = r_alpha,
            points  = points,
            tree    = tree,
        )

        if neighbour_idx.size == 0: # isolated point – cannot overlap
            continue

        if np.any(scan_ids[neighbour_idx] != scan_ids[i]):
            overlap_mask[neighbour_idx] = True # whole cylinder
            overlap_mask[i]       = True # include the seed itself

    overlap_idx = np.nonzero(overlap_mask)[0]
    return overlap_idx, overlap_mask


def compute_overlap_set_cached(
    scan_ids:       np.ndarray,         # (N,)
    nbr_cache:      list[np.ndarray]    # (N,K)
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        overlap_idx: 1-D array of indices belonging to the overlap set
        overlap_mask:   boolean mask of length N (True -> point in overlap set)
    """
    N = scan_ids.shape[0]
    overlap_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if overlap_mask[i]: # already decided via a neighbour
            continue

        neighbour_idx = nbr_cache[i]

        if neighbour_idx.size == 0: # isolated point - cannot overlap
            continue

        if np.any(scan_ids[neighbour_idx] != scan_ids[i]):
            overlap_mask[neighbour_idx] = True # whole cylinder
            overlap_mask[i]             = True # include the seed itself

    overlap_idx = np.nonzero(overlap_mask)[0]
    return overlap_idx, overlap_mask

def smooth_overlap_set_cached(
        overlap_mask:   np.ndarray,
        nbr_cache:      list[np.ndarray],
        p_thresh:       float=0.5,
        non_ovlp_thresh:int=5,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Relabels small islands of non-overlapping vertices.

    Args:
        overlap_mask (np.ndarray): boolean mask; is vertex in overlap set?
        nbr_cache (list[np.ndarray]): list of precomputed neighbours per vertex

    Returns:
        Tuple[np.ndarray, np.ndarray]: overlap_idx, overlap_mask
    """
    mask_out = overlap_mask.copy()
    for i in np.arange(overlap_mask.shape[0]):

        if overlap_mask[i]:
            continue

        neighbour_idx = nbr_cache[i]

        if neighbour_idx.size == 0:
            print(f'point {i} has no neighbours')
            continue

        ovlp_nbrs = overlap_mask[neighbour_idx]
        
        # Proportion of neighbours in overlap set
        p = np.sum(ovlp_nbrs) / len(ovlp_nbrs)

        if p > p_thresh or (len(ovlp_nbrs) - np.sum(ovlp_nbrs) < non_ovlp_thresh):
            mask_out[i] = True

    idx_out = np.nonzero(mask_out)[0]

    return (idx_out, mask_out)


def trilateral_shift(
    points:           np.ndarray,               # (N,3)
    normals:          np.ndarray,               # (N,3)
    local_spacing:    np.ndarray,               # (N,) local sampling spacing
    local_density:    np.ndarray,               # (N,) local sampling density
    overlap_idx:      np.ndarray,               # (L,) indices to be updated
    tree:             o3d.geometry.KDTreeFlann, # open3d KD-tree
    r_alpha: float = 2.0,
    h_alpha: float = 2.0,
) -> np.ndarray:
    """
    Returns: updated (N,3) point array after one trilateral-shift pass.
    """

    new_pts = points.copy()

    for i in overlap_idx:
        p = points[i]
        n = normals[i]
        Dpj = local_spacing[i]

        nbr, _ = find_cyl_neighbours(
                    point   = p,
                    normal  = n,
                    local_spacing = Dpj,
                    h_alpha = h_alpha,
                    r_alpha = r_alpha,
                    points  = points,
                    tree    = tree,
                    self_idx= i
                )

        if nbr.size == 0:
            continue

        pts_nbr = points[nbr]               # (k,3)
        diff    = pts_nbr - p                # vectors from p to neighbours
        h_signed= diff @ n                   # signed axial offsets  (k,)
        h_abs   = np.abs(h_signed)
        r2      = np.square(diff).sum(axis=1) - h_abs*h_abs
        r2      = np.maximum(r2, 0.0)
        r_abs   = np.sqrt(r2)                # radial distance

        sigma_r = r_alpha * Dpj
        sigma_h = 2.0 * h_alpha * Dpj
        w_domain = np.exp(-(r_abs**2) / (2*sigma_r**2))
        w_range  = np.exp(-(h_abs**2) / (2*sigma_h**2))

        rho_i   = local_density[i]
        rho_nbr = local_density[nbr]
        d_rho   = rho_i - rho_nbr           # sign irrelevant after square
        sigma_rho = np.max(np.abs(d_rho)) + 1e-12
        w_rho   = np.exp(-(d_rho**2) / (2*sigma_rho**2))
        w       = w_domain * w_range * w_rho


        w_sum = w.sum()
        if w_sum < 1e-12:
            continue

        delta_h = (w @ h_signed) / w_sum
        new_pts[i] = p + delta_h * n        # move along normal only

    return new_pts


def trilateral_shift_cached(
    points:         np.ndarray,                 # (N,3)
    normals:        np.ndarray,                 # (N,3)
    local_spacing:  np.ndarray,                 # (N,) local sampling spacing
    local_density:  np.ndarray,                 # (N,) local sampling density
    overlap_idx:    np.ndarray,                 # (L,) indices to be updated
    nbr_cache:      list[np.ndarray],           # (N,K)
    r_alpha: float = 2.0,
    h_alpha: float = 2.0,
    shift_all: bool = False,                    # All pts or just overlap
) -> np.ndarray:
    """
    Returns: updated (N,3) point array after one trilateral-shift pass.
    """

    new_pts = points.copy()

    shift_idx = overlap_idx if not shift_all else range(len(points))

    for i in shift_idx:
        p = points[i]
        n = normals[i]
        Dpj = local_spacing[i]

        nbr = nbr_cache[i]

        if nbr.size == 0:
            continue

        pts_nbr = points[nbr]               # (k,3)
        diff    = pts_nbr - p               # vectors from p to neighbours
        h_signed= diff @ n                  # signed axial offsets  (k,)
        h_abs   = np.abs(h_signed)
        r2      = np.square(diff).sum(axis=1) - h_abs*h_abs
        r2      = np.maximum(r2, 0.0)
        r_abs   = np.sqrt(r2)               # radial distance

        sigma_r = r_alpha * Dpj
        sigma_h = 2.0 * h_alpha * Dpj
        w_domain = np.exp(-(r_abs**2) / (2*sigma_r**2))
        w_range  = np.exp(-(h_abs**2) / (2*sigma_h**2))

        rho_i   = local_density[i]
        rho_nbr = local_density[nbr]
        d_rho   = rho_i - rho_nbr           # sign irrelevant after square
        sigma_rho = np.max(np.abs(d_rho)) + 1e-12
        w_rho   = np.exp(-(d_rho**2) / (2*sigma_rho**2))
        w       = w_domain * w_range * w_rho


        w_sum = w.sum()
        if w_sum < 1e-12:
            continue

        delta_h = (w @ h_signed) / w_sum
        new_pts[i] = p + delta_h * n        # move along normal only

    return new_pts


def find_boundary_edges(triangles:np.ndarray):
    """
    Finds the boundary edges of some triangles
        
    Parameters:
    triangles : array-like, shape (K, 3)
        Array of triangles where each row contains 3 vertex indices
        
    Returns:
    boundary_edges : ndarray, shape (N, 2)
        Array of boundary edges where each row contains 2 vertex indices
    """    
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]], 
        triangles[:, [2, 0]]
    ])
    
    # Sort vertices in each edge
    edges = np.sort(edges, axis=1)
    
    # Find unique edges and their counts
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    # Return edges that appear exactly once
    return unique_edges[counts==1]


def topological_trim(mesh: o3d.geometry.TriangleMesh,
                     cut_edges: np.ndarray,
                     keep_largest: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Remove all faces that lie on the other side of one or more closed edge-loops.

    Parameters
    ----------
    mesh         : open3d.geometry.TriangleMesh (watertight or open)
    cut_edges    : (N,2) array of vertex indices that form one or more closed loops
    keep_largest : keep only the component with the largest total area (True),
                   otherwise return the whole list of outside components.

    Returns
    -------
    o3d.geometry.TriangleMesh with unwanted triangles removed and vertices compacted.
    """
    tris = np.asarray(mesh.triangles)
    # Build edge to tri dictionary
    edge2tri = defaultdict(list)
    for tidx, tri in enumerate(tris):
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge2tri[tuple(sorted(e))].append(tidx)

    # Mark loop edges
    cut = {tuple(sorted(e)) for e in cut_edges}

    # Adjacency graph that ignores loop edges
    neighbours: list[list] = [[] for _ in range(len(tris))]
    for e, faces in edge2tri.items():
        if e in cut or len(faces) != 2:
            continue
        a, b = faces
        neighbours[a].append(b)
        neighbours[b].append(a)

    # Label connected components via BFS
    comp = np.full(len(tris), -1, dtype=int)
    curr = 0
    for seed in range(len(tris)):
        if comp[seed] != -1:
            continue
        q = deque([seed])
        comp[seed] = curr
        while q:
            f = q.popleft()
            for n in neighbours[f]:
                if comp[n] == -1:
                    comp[n] = curr
                    q.append(n)
        curr += 1

    # Choose which components to keep
    if keep_largest and curr > 1:
        verts = np.asarray(mesh.vertices)[tris]
        areas = 0.5 * np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0],
                                              verts[:, 2] - verts[:, 0]), axis=1)
        comp_area = np.bincount(comp, weights=areas, minlength=curr)
        keep_comp = np.argmax(comp_area)
        tri_keep  = np.flatnonzero(comp == keep_comp)
    else:
        tri_keep = np.arange(len(tris))  # Keep all triangles

    # Triangle ids to drop from the mesh
    tri_drop = np.setdiff1d(np.arange(len(tris)), tri_keep, assume_unique=True)

    out = copy.copy(mesh)
    out.remove_triangles_by_index(tri_drop.tolist())
    # out.remove_unreferenced_vertices()
    return out


def merge_nearby_clusters(
    trilat_shifted_pts: np.ndarray,
    normals: np.ndarray,
    colours: np.ndarray,
    overlap_mask: np.ndarray,
    overlap_idx: np.ndarray,
    global_avg_spacing: float,
    h_alpha: float,
    tree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge nearby clusters of points in the overlap region.

    Args:
        points: (N, 3) array of original points.
        trilat_shifted_pts: (N, 3) array of shifted points.
        normals: (N, 3) array of normals.
        colours: (N, C) array of colours.
        overlap_mask: (N,) boolean array indicating overlap points.
        global_avg_spacing: float, global average spacing.
        h_alpha: float, parameter for find_cyl_neighbours.
        tree: kd-tree of (all) trilaterally-shifted points

    Returns:
        cluster_mapping: (N,) array mapping each point to its cluster id or -1.
        clustered_overlap_pnts: (K, 3) array of merged overlap points.
        clustered_overlap_cols: (K, C) array of merged overlap colours.
        clustered_overlap_nrms: (K, 3) array of merged overlap normals.
    """

    # Convenience view limited to the overlap region (M points)
    trilat_shifted_overlap_pts = trilat_shifted_pts[overlap_idx]

    delta  = np.sqrt(2.0) / 2.0
    sigma  = delta * global_avg_spacing

    # -1 means 'not yet assigned to any merged cluster'
    cluster_mapping = -np.ones(len(trilat_shifted_pts), dtype=int)

    clustered_overlap_pnts: list = []
    clustered_overlap_cols: list = []
    clustered_overlap_nrms: list = []

    # keep iterating until every overlap point belongs to a cluster
    while (cluster_mapping[overlap_idx] < 0).any():
        # choose a still‑unassigned local seed index inside the overlap set
        free_local_idx = np.flatnonzero(cluster_mapping[overlap_idx] < 0)
        id_local       = np.random.choice(free_local_idx)

        point  = trilat_shifted_overlap_pts[id_local]
        normal = normals[overlap_idx[id_local]]

        # Note: 'find_cyl_neighbours' returns global indices w.r.t. the full tree
        nbr_global, d2 = find_cyl_neighbours(
            point,
            normal,
            global_avg_spacing,
            h_alpha,
            delta,
            trilat_shifted_pts,     # full data set for distance look‑up
            tree,
            self_idx=None,
        )

        # Keep neighbours that are themselves in the overlap region
        overlap_neigh_mask = overlap_mask[nbr_global]
        nbr_global = nbr_global[overlap_neigh_mask]
        d2         = d2[overlap_neigh_mask]

        # Keep only neighbours that have not yet been assigned to a cluster
        unassigned_mask = cluster_mapping[nbr_global] < 0
        nbr_global = nbr_global[unassigned_mask]
        d2         = d2[unassigned_mask]

        if nbr_global.size == 0:
            raise RuntimeError("Point neighbourhood is unexpectedly empty!")

        # Gaussian weights based on squared distances
        w = np.exp(-d2 / (2.0 * sigma**2))
        w /= w.sum()

        merged_id = len(clustered_overlap_pnts)
        cluster_mapping[nbr_global] = merged_id

        # Weighted averages
        clustered_overlap_pnts.append(w @ trilat_shifted_pts[nbr_global])
        clustered_overlap_cols.append(w @ colours[nbr_global])

        merged_nrm = w @ normals[nbr_global]
        clustered_overlap_nrms.append(merged_nrm / np.linalg.norm(merged_nrm))

    return (
        cluster_mapping,
        np.vstack(clustered_overlap_pnts),
        np.vstack(clustered_overlap_cols),
        np.vstack(clustered_overlap_nrms),
    )