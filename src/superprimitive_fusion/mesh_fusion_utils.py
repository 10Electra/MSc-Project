import math
import numpy as np
import open3d as o3d # type: ignore
from typing import Tuple, Union
import copy
from collections import deque, defaultdict
from scipy.spatial import cKDTree # type: ignore
from superprimitive_fusion.utils import distinct_colours


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
        if r_k2 == 0:
            print('Warning: a neighbour distance was found to be 0')
            r_k2 += 1e-8
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
    """Returns list[ndarray] of neighbour indices for every point (excl. self idx)."""
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


def normal_shift_smooth(
    points:         np.ndarray,                 # (N,3)
    normals:        np.ndarray,                 # (N,3)
    weights:        np.ndarray,                 # (N,) precision for each point
    local_spacing:  np.ndarray,                 # (N,) local sampling spacing
    local_density:  np.ndarray,                 # (N,) local sampling density
    overlap_idx:    np.ndarray,                 # (L,) indices to be updated
    nbr_cache:      list[np.ndarray],           # (N,K)
    r_alpha:        float = 2.0,
    h_alpha:        float = 2.0,
    sigma_theta:    float = 0.2,                # Normal angle difference std (related to sharpness)
    normal_diff_thresh: float = 45.0,           # degrees; hard front-facing gate
    shift_all:      bool = False,               # All pts or just overlap
    rho_floor:      float = 1e-6,               # floor for density std; disables w_density if tiny
) -> np.ndarray:
    """
    One pass of along-normal shifting. Assumes:
      - normals are unit-length,
      - nbr_cache[i] already lies inside the cylinder (radius r_alpha*D_i, height 2*h_alpha*D_i),
      - weights are precisions (e.g. 1/covariance).
    Returns: updated (N,3) points array.
    """

    new_pts = points.copy()
    shift_idx = overlap_idx if not shift_all else range(len(points))

    cos_thr = np.cos(np.deg2rad(normal_diff_thresh))
    sigma_c = 0.5 * (sigma_theta ** 2)  # =~ mapping from angle sigma to cosine sigma
    
    for i in shift_idx:
        p = points[i]
        n = normals[i]
        Dpj = local_spacing[i]

        nbr = nbr_cache[i]
        if nbr.size == 0:
            continue

        pts_nbr = points[nbr]               # (k,3)
        weights_nbr = weights[nbr]          # (k,)
        normals_nbr = normals[nbr]          # (k,3)

        # Normal similarity
        cos_theta = np.einsum('ij,j->i', normals_nbr, n)
        mask_face = cos_theta >= cos_thr
        if not np.any(mask_face): 
            continue
        
        cos_theta   = cos_theta[mask_face]
        pts_nbr     = pts_nbr[mask_face]
        weights_nbr = weights_nbr[mask_face]
        normals_nbr = normals_nbr[mask_face]

        diff    = pts_nbr - p               # Vectors from p to neighbours
        h       = diff @ n                  # Signed axial offsets  (k,)
        h2      = h * h
        
        r2      = np.einsum('ij,ij->i', diff, diff) - h2
        r2      = np.maximum(r2, 0.0)       # Squared radial distance
        
        sigma_r     = r_alpha * Dpj         # Radius of cylinder neighbourhood
        sigma_h     = 2.0 * h_alpha * Dpj   # Height of cylinder neighbourhood
        
        # Spatial Gaussian (radius and height components); summed exponents for stability
        S = r2 / (2.0 * sigma_r * sigma_r) + h2 / (2.0 * sigma_h * sigma_h)
        w_spatial = np.exp(-S)
        
        w_normal = np.exp(-(1.0 - cos_theta) / (2.0 * sigma_c))
        
        
        # Using MAD as the std for local density
        d_rho = local_density[i] - local_density[nbr][mask_face]
        med         = np.median(d_rho)
        mad         = np.median(np.abs(d_rho - med)) # Median absolute deviation
        sigma_rho = 1.4826 * mad # =~ std if Gaussian
        if sigma_rho < rho_floor:
            w_density = 1.0
            print('[WARN; vertex smoother] Using w_density=1.0 as sigma_rho < rho_floor.')
        else:
            w_density = np.exp(-(d_rho * d_rho) / (2.0 * sigma_rho * sigma_rho))

        w = weights_nbr * w_spatial * w_density * w_normal

        w_sum = w.sum()
        if w_sum < 1e-12:
            continue

        delta_h = (w @ h) / w_sum
        new_pts[i] = p + delta_h * n # move along normal only

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
                     k: int = 1) -> o3d.geometry.TriangleMesh:
    """
    Remove faces that lie on the other side of one or more closed edge-loops
    and keep the k largest remaining components (by area).

    Parameters
    ----------
    mesh      : open3d.geometry.TriangleMesh (watertight or open)
    cut_edges : (N,2) int array of vertex indices forming one or more closed loops
    k         : number of largest components to keep (k <= total components)

    Returns
    -------
    A copy of 'mesh' with the unwanted triangles removed and vertices compacted.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

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

    verts = np.asarray(mesh.vertices)[tris] # (T,3,3)
    tri_area = 0.5 * np.linalg.norm(
        np.cross(verts[:, 1] - verts[:, 0],
                 verts[:, 2] - verts[:, 0]), axis=1)
    comp_area = np.bincount(comp, weights=tri_area, minlength=curr)
    
    if k >= curr:                          # nothing to discard
        tri_keep = np.arange(len(tris))
    else:
        keep_comps = np.argpartition(comp_area, -k)[-k:]  # unsorted k‑largest
        tri_keep   = np.flatnonzero(np.isin(comp, keep_comps))

    tri_drop = np.setdiff1d(np.arange(len(tris)), tri_keep, assume_unique=True)

    out = copy.copy(mesh)
    out.remove_triangles_by_index(tri_drop.tolist())
    return out


def merge_nearby_clusters(
    normal_shifted_points:  np.ndarray,
    normals:                np.ndarray,
    weights:                np.ndarray,
    colours:                np.ndarray,
    overlap_mask:           np.ndarray,
    overlap_idx:            np.ndarray,
    global_avg_spacing:     float,
    h_alpha:                float,
    tree,
    normal_diff_thresh:     float = 45.0  # degrees; front-facing gate
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vertex density reduction by weighted vertex merging.
    - Seed points are selected in order of descending precision
    - Cluster neighbourhood/membership == cylindrical neighbourhood
    around the seed; same-side-facing points only
    - Merge operator: sum of precisions; precision-weighted means.
    
    Args:
        points: (N, 3) array of original points.
        normal_shifted_points: (N, 3) array of shifted points.
        normals: (N, 3) array of normals.
        weights: (N,) array of weights representing point confidence.
        colours: (N, C) array of colours.
        overlap_mask: (N,) boolean array indicating overlap points.
        global_avg_spacing: float, global average spacing.
        h_alpha: float, parameter for find_cyl_neighbours.
        tree: kd-tree of (all) normal-shifted points
        normal_diff_thresh: normal difference angle threshold for same-way-facing

    Returns:
        cluster_mapping: (N,) assigned cluster id for each input point or -1 if unclustered.
        clustered_pnts: (K, 3)
        clustered_cols: (K, C)
        clustered_nrms: (K, 3)
        clustered_wts:  (K,) precision (sum of member precisions)
    """

    N = normal_shifted_points.shape[0]
    C = colours.shape[1]
    
    # Output lists
    out_pts:   list[np.ndarray] = []
    out_cols:  list[np.ndarray] = []
    out_nrms:  list[np.ndarray] = []
    out_wts:   list[float]      = []
    
    # cluster mapping == -1 means 'not yet assigned to any merged cluster'
    cluster_mapping = -np.ones(N, dtype=int)
    visited         = np.zeros(N, dtype=bool)

    # Highest precision first among overlap points
    ordered_seeds = overlap_idx[np.argsort(weights[overlap_idx])[::-1]]

    cos_th = np.cos(np.deg2rad(normal_diff_thresh))

    for seed in ordered_seeds:
        if visited[seed]:
            continue
        
        p0  = normal_shifted_points[seed]
        n0  = normals[seed]

        # Find cylindrical neighbours around the seed (global indices)
        nbr_global, d2 = find_cyl_neighbours(
            point=p0,
            normal=n0,
            local_spacing=global_avg_spacing,
            h_alpha=h_alpha,
            r_alpha=np.sqrt(2.0) / 2.0,
            points=normal_shifted_points,
            tree=tree,
            self_idx=seed
        )

        if nbr_global.size == 0:
            # Lonely seed -> single-point cluster
            idxs = np.array([seed], dtype=int)
        else:
            # Keep only overlap, not yet visited
            m = overlap_mask[nbr_global] & (~visited[nbr_global])
            if not np.any(m):
                # Nothing to merge; make a singleton cluster for the seed
                idxs = np.array([seed], dtype=int)
            else:
                nbr_global = nbr_global[m]

                # Front-facing gate
                cos = (normals[nbr_global] @ n0)
                m2  = cos >= cos_th
                idxs = nbr_global[m2]
                if idxs.size == 0:
                    idxs = np.array([seed], dtype=int)

        # Bayesian merge on idxs
        tau = weights[idxs]     # (K,)
        T   = float(tau.sum())  # scalar precision

        # Positions & colours: precision-weighted means
        x_merge = (tau[:, None] * normal_shifted_points[idxs]).sum(axis=0) / T
        c_merge = (tau[:, None] * colours[idxs]).sum(axis=0) / T

        # Normal: principal eigenvector of sum tau * n n^T
        n_i   = normals[idxs]   # (K,3)
        M     = (tau[:, None, None] * (n_i[:, :, None] * n_i[:, None, :])).sum(axis=0)  # 3x3
        eigvals, eigvecs = np.linalg.eigh(M)
        n_merge = eigvecs[:, np.argmax(eigvals)]
        # Align merged normal with seed normal for consistency
        if float(n_merge @ n0) < 0.0:
            n_merge = -n_merge

        # Record
        cid = len(out_pts)
        cluster_mapping[idxs] = cid
        visited[idxs] = True

        out_pts.append(x_merge.astype(normal_shifted_points.dtype, copy=False))
        out_cols.append(c_merge.astype(colours.dtype, copy=False))
        out_nrms.append(n_merge.astype(normals.dtype, copy=False))
        out_wts.append(T)

    # Stack outputs
    clustered_overlap_pnts = np.vstack(out_pts)     if out_pts else  np.zeros((0,3), dtype=normal_shifted_points.dtype)
    clustered_overlap_cols = np.vstack(out_cols)    if out_cols else np.zeros((0,C), dtype=colours.dtype)
    clustered_overlap_nrms = np.vstack(out_nrms)    if out_nrms else np.zeros((0,3), dtype=normals.dtype)
    clustered_overlap_wts  = np.asarray(out_wts, dtype=weights.dtype)

    return (
        cluster_mapping,
        clustered_overlap_pnts,
        clustered_overlap_cols,
        clustered_overlap_nrms,
        clustered_overlap_wts,
    )

def sanitise_mesh(V, F):
    # shapes
    assert V.ndim == 2 and V.shape[1] == 3, f"V shape {V.shape}"
    assert F.ndim == 2 and F.shape[1] == 3, f"F shape {F.shape}"

    # dtypes / contiguity
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)

    # finite vertices
    finite = np.isfinite(V).all(axis=1)
    if not finite.all():
        bad = np.where(~finite)[0]
        print(f"Removing {len(bad)} non-finite verts")

    # old -> new index map  (-1 for removed)
    Vmap = -np.ones(len(V), dtype=np.int32)
    Vmap[finite] = np.arange(finite.sum(), dtype=np.int32)

    # drop faces touching removed verts
    keep_faces = (Vmap[F] >= 0).all(axis=1)
    F = F[keep_faces]

    # remap faces so they reference the *same* vertices (in geometry) after compaction
    F = Vmap[F]

    # compact vertices
    V = V[finite]

    # remove degenerate/zero-area faces
    if F.size:
        same = (F[:,0] == F[:,1]) | (F[:,1] == F[:,2]) | (F[:,2] == F[:,0])
        if same.any():
            F = F[~same]

    if F.size:
        n = np.cross(V[F[:,1]] - V[F[:,0]], V[F[:,2]] - V[F[:,0]])
        a = np.linalg.norm(n, axis=1)
        F = F[a > 1e-18]

    # final dtypes / contiguity
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)
    return V, F

def colour_transfer(V0, C0, V1):
    tree = cKDTree(V0)
    dist, idx = tree.query(V1, k=1)
    C1 = C0[idx]

    mask = dist > 1e-6
    if mask.any():
        dist3, idx3 = tree.query(V1[mask], k=3)
        w = 1.0 / (dist3 + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        C1[mask] = (C0[idx3] * w[..., None]).sum(axis=1)
    return C1

def get_mesh_components(mesh:o3d.geometry.TriangleMesh, show=True):
    out = mesh.cluster_connected_triangles()

    clust_ids = np.asarray(out[0])
    unique_clusters = np.unique(clust_ids)
    print(f'There are {len(unique_clusters)} clusters of connected components')

    colours = distinct_colours(len(unique_clusters))
    colours = (colours * 255).astype(np.uint8)

    components = []
    for i in unique_clusters:
        component = copy.deepcopy(mesh)
        tris = np.asarray(component.triangles)
        tris_clust = tris[clust_ids==i]
        component.triangles = o3d.utility.Vector3iVector(tris_clust)
        component.paint_uniform_color(colours[i].astype(float) / 255)
        components.append(component)

    if show:
        o3d.visualization.draw_geometries(
            components
        )
    return components

def count_inconsistent_normal_pairs(mesh: o3d.geometry.TriangleMesh,
                                    dot_threshold: float = 0.0,
                                    include_nonmanifold_pairs: bool = False,
                                    show: bool = False) -> int:
    if len(mesh.triangles) == 0:
        if show:
            print("Found 0 inconsistent pairs across 0 triangles")
        return 0

    if len(mesh.triangle_normals) != len(mesh.triangles):
        mesh.compute_triangle_normals(normalized=True)

    normals = np.asarray(mesh.triangle_normals, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    # Build list of adjacent triangle index pairs
    edge_to_tris = defaultdict(list)
    for t_idx, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_tris[e].append(t_idx)

    pairs = []
    for tris in edge_to_tris.values():
        k = len(tris)
        if k == 2:
            pairs.append(tris)
        elif include_nonmanifold_pairs and k > 2:
            for i in range(k):
                for j in range(i + 1, k):
                    pairs.append((tris[i], tris[j]))

    if not pairs:
        if show:
            print("Found 0 inconsistent pairs across 0 triangles")
        return 0

    pairs = np.asarray(pairs, dtype=np.int64)
    n1 = normals[pairs[:, 0]]
    n2 = normals[pairs[:, 1]]
    dots = np.einsum("ij,ij->i", n1, n2)
    bad_pair_mask = dots < dot_threshold
    count = int(np.count_nonzero(bad_pair_mask))

    if show:
        bad_tris_idx = np.unique(pairs[bad_pair_mask].ravel())
        bad_mask = np.zeros(len(triangles), dtype=bool)
        bad_mask[bad_tris_idx] = True

        bg = copy.deepcopy(mesh)
        fg = copy.deepcopy(mesh)

        bg.triangles = o3d.utility.Vector3iVector(triangles[~bad_mask])
        fg.triangles = o3d.utility.Vector3iVector(triangles[bad_mask])

        bg.paint_uniform_color([0.8, 0.8, 0.8])
        fg.paint_uniform_color([1.0, 0.0, 0.0])

        print(f"Found {count} inconsistent pairs across {bad_mask.sum()} triangles")
        o3d.visualization.draw_geometries([bg, fg])

    return count


def show_mesh_boundaries(mesh: o3d.geometry.TriangleMesh, show: bool = True, edges: bool = True, base_mesh:o3d.geometry.TriangleMesh|None=None):
    """
    Display boundary edges (preferred) or boundary vertices of a triangle mesh.
    Returns the list of Open3D geometries shown.
    """
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if tris.size == 0:
        if show:
            print("No triangles.")
        return []

    edge_counts = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_counts[e] = edge_counts.get(e, 0) + 1

    bnd_edges = np.array([e for e, cnt in edge_counts.items() if cnt == 1], dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    if base_mesh is None:
        bg = copy.deepcopy(mesh)
    else:
        bg = copy.deepcopy(base_mesh)
    bg.paint_uniform_color([0.8, 0.8, 0.8])

    geoms = [bg]
    if edges and len(bnd_edges) > 0:
        uniq = np.unique(bnd_edges.ravel())
        idx = {v: i for i, v in enumerate(uniq)}
        lines = np.array([(idx[i], idx[j]) for i, j in bnd_edges], dtype=np.int32)
        ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(verts[uniq]),
                                  lines=o3d.utility.Vector2iVector(lines))
        ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(lines), 1)))
        geoms.append(ls)
    else:
        uniq = np.unique(bnd_edges.ravel()) if len(bnd_edges) > 0 else np.array([], dtype=np.int64)
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts[uniq]))
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(pcd)

    print(f"Found {len(bnd_edges)} boundary edges and {len(uniq)} boundary vertices.")
    if show:
        o3d.visualization.draw_geometries(geoms)
    return geoms