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


def trilateral_shift_cached(
    points:         np.ndarray,                 # (N,3)
    normals:        np.ndarray,                 # (N,3)
    weights:        np.ndarray,                 # (N,) weights for each point
    local_spacing:  np.ndarray,                 # (N,) local sampling spacing
    local_density:  np.ndarray,                 # (N,) local sampling density
    overlap_idx:    np.ndarray,                 # (L,) indices to be updated
    nbr_cache:      list[np.ndarray],           # (N,K)
    r_alpha: float = 2.0,
    h_alpha: float = 2.0,
    sigma_theta: float = 0.2,                   # normal angle difference std (related to sharpness)
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
        weights_nbr = weights[nbr]          # (k,)
        normals_nbr = normals[nbr]          # (k,3)

        diff    = pts_nbr - p               # vectors from p to neighbours
        h_signed= diff @ n                  # signed axial offsets  (k,)
        h_abs   = np.abs(h_signed)

        r2      = np.square(diff).sum(axis=1) - h_abs*h_abs
        r2      = np.maximum(r2, 0.0)
        r_abs   = np.sqrt(r2)               # radial distance

        cos_theta   = normals_nbr @ n       # (k,) cosine of angle between normals

        rho_i       = local_density[i]
        rho_nbr     = local_density[nbr]
        d_rho       = rho_i - rho_nbr       # sign irrelevant after square

        sigma_r     = r_alpha * Dpj
        sigma_h     = 2.0 * h_alpha * Dpj
        sigma_rho   = np.max(np.abs(d_rho)) + 1e-12

        w_radial    = np.exp(-(r_abs**2) / (2*sigma_r**2))
        w_vertical  = np.exp(-(h_abs**2) / (2*sigma_h**2))
        w_normal    = np.exp(-((1.0 - cos_theta)**2) / (2 * sigma_theta**2))
        w_density   = np.exp(-(d_rho**2) / (2*sigma_rho**2))

        w           = w_radial * w_vertical * w_density * w_normal

        w *= weights_nbr                    # incorporate vertex confidence


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
    trilat_shifted_pts: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    colours: np.ndarray,
    overlap_mask: np.ndarray,
    overlap_idx: np.ndarray,
    global_avg_spacing: float,
    h_alpha: float,
    tree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge nearby clusters of points in the overlap region.

    Args:
        points: (N, 3) array of original points.
        trilat_shifted_pts: (N, 3) array of shifted points.
        normals: (N, 3) array of normals.
        weights: (N,) array of weights representing point confidence.
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
    clustered_overlap_wts: list = []

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

        conf_nbr = weights[nbr_global]  # shape (K,)

        w *= conf_nbr                   # modulate spatial weights by confidence
        w_sum = w.sum()

        if w_sum < 1e-12:
            continue                    # avoid divide-by-zero

        w /= w_sum

        merged_id = len(clustered_overlap_pnts)
        cluster_mapping[nbr_global] = merged_id

        # Weighted averages
        clustered_overlap_pnts.append(w @ trilat_shifted_pts[nbr_global])
        clustered_overlap_cols.append(w @ colours[nbr_global])
        clustered_overlap_wts.append(conf_nbr.sum())

        merged_nrm = w @ normals[nbr_global]
        clustered_overlap_nrms.append(merged_nrm / np.linalg.norm(merged_nrm))

    return (
        cluster_mapping,
        np.vstack(clustered_overlap_pnts),
        np.vstack(clustered_overlap_cols),
        np.vstack(clustered_overlap_nrms),
        np.vstack(clustered_overlap_wts),
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