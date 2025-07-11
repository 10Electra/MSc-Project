import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d
from trimesh import Trimesh
from typing import Tuple, Union
import copy
from collections import deque, defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_integer_segments(sp_regions):
    integer_segments = np.zeros([sp_regions.shape[1], sp_regions.shape[2]])
    for i in range(sp_regions.shape[0]):
        integer_segments[sp_regions[i,:,:]] = i + 1
    return integer_segments

# def plot_region_numbers(integer_segments:np.ndarray,text_size=8,figsize=(10,8)):
#     plt.figure(figsize=figsize)
#     plt.imshow(integer_segments, cmap='nipy_spectral')

#     # Calculate center of mass for each region and add text labels
#     for region_id in range(1, int(np.max(integer_segments)) + 1):
#         region_mask = (integer_segments == region_id)
        
#         # Calculate center of mass
#         if np.any(region_mask):  # Check if region exists
#             y_coords, x_coords = np.where(region_mask)
#             com_y = float(np.mean(y_coords))
#             com_x = float(np.mean(x_coords))
            
#             # Add text at center of mass
#             plt.text(com_x, com_y, str(region_id), 
#                     color='white', fontweight='bold', fontsize=text_size,
#                     ha='center', va='center',
#                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

#     plt.colorbar(label='Region ID')
#     plt.title('Segmented Regions with Labels')
#     plt.axis('off')
#     plt.show()

def plot_region_numbers(integer_segments: np.ndarray,
                        text_size: int = 8,
                        figsize: tuple = (10, 8),
                        title: str = 'Labelled Segments'):
    # --- figure & axes -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(integer_segments, cmap='nipy_spectral')

    # --- label every region --------------------------------------------------
    n_regions = int(np.max(integer_segments))
    for region_id in range(1, n_regions + 1):
        region_mask = (integer_segments == region_id)
        if np.any(region_mask):
            y, x = np.where(region_mask)
            ax.text(x.mean(), y.mean(), str(region_id),
                    color='white', fontweight='bold', fontsize=text_size,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.7))

    # --- colour-bar, same height --------------------------------------------
    divider = make_axes_locatable(ax)                     # split the axes
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 5 % as wide
    cbar = fig.colorbar(im, cax=cax)                      # attach here
    cbar.set_label('Region ID')

    # --- cosmetics -----------------------------------------------------------
    ax.set_title(title)
    ax.axis('off')                                        # hide ticks/box
    plt.show()

def quadto2tris(idx: tuple, verts: np.ndarray) -> tuple[list[int], list[int]]:
    """Decides which two triangles to construct from the quadrilateral provided"""
    
    diag1_len_sq = np.sum((verts[0] - verts[2])**2)
    diag2_len_sq = np.sum((verts[1] - verts[3])**2)
    
    if diag1_len_sq <= diag2_len_sq:
        tri1 = [idx[0], idx[2], idx[1]]
        tri2 = [idx[0], idx[3], idx[2]]
    else:
        tri1 = [idx[0], idx[3], idx[1]]
        tri2 = [idx[1], idx[3], idx[2]]
    
    return (tri1, tri2)

def triangulate_segments(verts, integer_segments):
    tris = [[] for _ in range(int(np.max(integer_segments))+1)]
    H, W = integer_segments.shape[0], integer_segments.shape[1]
    for v in range(H - 1):
        for u in range(W - 1):
            id_tl = v * W + u
            id_tr = id_tl + 1
            id_bl = id_tl + W
            id_br = id_bl + 1

            quad_idx = (id_bl, id_tl, id_tr, id_br)
            quad_verts = np.array([verts[id] for id in quad_idx])

            sps = [int(integer_segments.flatten()[id]) for id in quad_idx]
            sp_counts = {}
            for sp in sps:
                sp_counts[sp] = 1 + sp_counts.get(sp, 0)

            if len(sp_counts) == 1 and sps[0] != 0: # All verts from one sp
                tri1, tri2 = quadto2tris(quad_idx, quad_verts)
                tris[sps[0]-1].extend([tri1, tri2])
            elif len(sp_counts) == 2 and len(set(sp_counts.values())) == 2: # Three verts from one sp, one from another
                aclock_quad_idx = list(quad_idx)
                for i in range(4):
                    if sp_counts[sps[i]] == 1:
                        del aclock_quad_idx[i]
                sample_sp = int(integer_segments.flatten()[aclock_quad_idx[0]])
                if sample_sp == 0:
                    continue # Skip this triangle if it is not in a superprimitive
                aclock_quad_idx.reverse()
                tris[sample_sp-1].append(aclock_quad_idx)
    return tris

def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh, copy: bool = False) -> Trimesh:
    """Convert Open3D TriangleMesh -> Trimesh, preserving vertex colours.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        The source mesh.
    copy : bool, default False
        If True, numpy arrays are deep-copied; otherwise views are used.

    Returns
    -------
    Trimesh
        A Trimesh with vertices, faces and (if present) vertex colours.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if copy:
        verts = verts.copy()
        faces = faces.copy()

    tm = Trimesh(vertices=verts, faces=faces, process=False)

    # Copy across vertex colours
    if mesh.has_vertex_colors():
        # Open3D stores floats in [0,1]; Trimesh expects uint8 in [0,255]
        cols = np.asarray(mesh.vertex_colors)
        if copy:
            cols = cols.copy()

        if cols.max() <= 1.0:
            cols = (cols * 255).astype(np.uint8)

        # Ensure Trimesh's 4-channel RGBA
        if cols.shape[1] == 3:
            alpha = np.full((cols.shape[0], 1), 255, dtype=np.uint8)
            cols = np.hstack([cols, alpha])

        tm.visual.vertex_colors = cols

    return tm

def trimesh_to_o3d(mesh: Trimesh, copy: bool = False) -> o3d.geometry.TriangleMesh:
    """Convert Trimesh -> Open3D TriangleMesh, preserving vertex colours.

    Parameters
    ----------
    mesh : Trimesh
        The source mesh.
    copy : bool, default False
        If True, numpy arrays are deep-copied before passing to Open3D.

    Returns
    -------
    o3d.geometry.TriangleMesh
        An Open3D mesh with vertices, faces and (if present) vertex colours.
    """
    verts = mesh.vertices.copy() if copy else mesh.vertices
    faces = mesh.faces.copy()     if copy else mesh.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices  = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Copy across vertex colours
    if mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors.size:
        cols = mesh.visual.vertex_colors
        if copy:
            cols = cols.copy()

        # Trimesh stores uint8 [0,255]; Open3D wants float [0,1] and RGB only
        if cols.dtype == np.uint8:
            cols = cols.astype(np.float64) / 255.0
        if cols.shape[1] == 4:        # drop alpha channel for Open3D
            cols = cols[:, :3]

        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    return o3d_mesh

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

        delta_h = (w @ h_signed) / w_sum   # minus sign pulls toward denser side
        new_pts[i] = p + delta_h * n        # move along normal only

    return new_pts

def get_o3d_colours_from_trimesh(colours:np.ndarray) -> o3d.utility.Vector3dVector:
    if colours.shape[1] == 4:
        no_alpha = colours[:,:3]
    else:
        no_alpha = colours
    
    return o3d.utility.Vector3dVector((no_alpha / 255).astype('float64'))

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
    global_avg_spacing: float,
    h_alpha: float,
    find_cyl_neighbours,
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
        find_cyl_neighbours: function to find cylindrical neighbours.

    Returns:
        cluster_mapping: (N,) array mapping each point to its cluster id or -1.
        clustered_overlap_pnts: (K, 3) array of merged overlap points.
        clustered_overlap_cols: (K, C) array of merged overlap colours.
        clustered_overlap_nrms: (K, 3) array of merged overlap normals.
    """
    import open3d as o3d

    overlap_idx = np.flatnonzero(overlap_mask)                            # (M,)
    trilat_shifted_overlap_pts = trilat_shifted_pts[overlap_idx]          # (M, 3)
    trilat_shifted_overlap_tree = o3d.geometry.KDTreeFlann(               # KD‑tree
        trilat_shifted_overlap_pts.T)

    delta = np.sqrt(2) / 2
    sigma = delta * global_avg_spacing

    cluster_mapping = -1 * np.ones(len(trilat_shifted_pts), dtype=int)

    clustered_overlap_pnts: list = []
    clustered_overlap_cols: list = []
    clustered_overlap_nrms: list = []

    while (cluster_mapping[overlap_idx] < 0).any():

        free_local_idx = np.flatnonzero(cluster_mapping[overlap_idx] < 0)  # local indices
        id_local = np.random.choice(free_local_idx)                      # local id in [0, M)

        point  = trilat_shifted_overlap_pts[id_local]                    # (3,)
        normal = normals[overlap_idx[id_local]]                          # (3,)

        nbr_local, d2 = find_cyl_neighbours(
            point,
            normal,
            global_avg_spacing,
            h_alpha,
            delta,
            trilat_shifted_overlap_pts,
            trilat_shifted_overlap_tree,
            self_idx=None)

        nbr_global = overlap_idx[nbr_local]

        mask = cluster_mapping[nbr_global] < 0
        nbr_global = nbr_global[mask]
        d2 = d2[mask]

        if len(nbr_global) == 0:
            raise RuntimeError("Point neighbourhood is unexpectedly empty!")

        w = np.exp(-d2 / (2 * sigma ** 2))
        w /= w.sum()

        merged_id = len(clustered_overlap_pnts)
        cluster_mapping[nbr_global] = merged_id

        clustered_overlap_pnts.append(w @ trilat_shifted_pts[nbr_global])
        clustered_overlap_cols.append(w @ colours[nbr_global])

        merged_nrm = w @ normals[nbr_global]
        clustered_overlap_nrms.append(merged_nrm / np.linalg.norm(merged_nrm))

    clustered_overlap_pnts_np = np.vstack(clustered_overlap_pnts)
    clustered_overlap_cols_np = np.vstack(clustered_overlap_cols)
    clustered_overlap_nrms_np = np.vstack(clustered_overlap_nrms)

    return cluster_mapping, clustered_overlap_pnts_np, clustered_overlap_cols_np, clustered_overlap_nrms_np

def smooth_mask(
    mask: np.ndarray,
    radius_erode: int = 2,
    radius_dilate: int | None = 2,
) -> np.ndarray:
    """Smooths a binary mask by conservative morphological erosion.

    Args:
        mask (np.ndarray): Boolean or 0/1 binary array where `True`/1 denotes
            the foreground region to be smoothed.
        radius_erode (int): Radius (in pixels) of the circular structuring
            element used for erosion. Larger values remove larger “teeth” and
            thin the mask more.
        radius_dilate (int | None): If an integer, the eroded mask is dilated
            with this (usually smaller) radius and then intersected with the
            original mask to regain a little thickness while **guaranteeing**
            that no new foreground pixels are added. If ``None`` (default),
            only erosion is applied.

    Returns:
        np.ndarray: A boolean array of the same shape as ``mask``, containing
            the smoothed (and still conservative) mask.
    """
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise TypeError("`mask` must be of dtype bool or uint8/0-1")

    # Ensure uint8 for OpenCV, but keep a bool copy for the final AND.
    m_uint8: np.ndarray = mask.astype(np.uint8, copy=False)

    # --- build circular structuring element --------------------------------
    def _disk(r: int) -> np.ndarray:
        """Return a (2r+1)×(2r+1) uint8 disk for cv::morphologyEx kernels."""
        if r < 1:
            raise ValueError("radius must be ≥ 1")
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    # --- 1. Erosion ---------------------------------------------------------
    eroded = cv.erode(m_uint8, _disk(radius_erode))

    if radius_dilate is None:
        return eroded.astype(bool, copy=False)

    # --- 2. Optional dilation, then clip to original ------------------------
    reopened = cv.dilate(eroded, _disk(radius_dilate))
    conservative = np.logical_and(reopened.astype(bool, copy=False), mask.astype(bool, copy=False))

    return conservative

def fill_ring_holes(
    mask: np.ndarray,
    radius: int = 3,
    max_iterations: int = 10,
) -> np.ndarray:
    """Fills background pixels whose surrounding ring is entirely foreground.

    Args:
        mask (np.ndarray): Binary image where `True` (or 1) denotes the
            foreground region.
        radius (int): Radius (in pixels) of the hollow circle that must be
            completely inside the foreground for the centre pixel to be filled.
        max_iterations (int): The operation is applied iteratively because
            filling one pixel may enable its neighbour on the next pass.  Set
            this to a small number (e.g. 10) or -1 for “until convergence”.

    Returns:
        np.ndarray: A boolean array where all pixels that satisfied the ring
            condition (possibly over several passes) have been converted to
            foreground.
    """
    if radius < 1:
        raise ValueError("'radius' must be >= 1")
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise TypeError("'mask' must be of dtype bool or uint8/0-1")

    # Build the hit-or-miss kernel
    side = 2 * radius + 1
    kernel = np.zeros((side, side), dtype=np.int8)

    yx = np.indices(kernel.shape) - radius
    dist = np.hypot(yx[0], yx[1])
    kernel[(dist >= radius - 0.5) & (dist <= radius + 0.5)] = 1   # ring = "should be 1"
    kernel[radius, radius] = -1                                   # centre = "should be 0"
    # 0  "don't care"

    # Iteratively apply hit-or-miss until no more pixels are filled
    work = mask.astype(bool, copy=True)
    it = 0
    while True:
        hit = cv.morphologyEx(work.astype(np.uint8), cv.MORPH_HITMISS, kernel)
        if not hit.any() or (max_iterations != -1 and it >= max_iterations):
            break
        work |= hit.astype(bool)
        it += 1

    return work

def crop_by_SP(segment_id:int,
               integer_segments:np.ndarray,
               image:np.ndarray, points:np.ndarray,
               border=0,
               make_binary=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crops the segmentation, colour and, 3D point arrays to the bounding rectangle around a particular segment.

    Args:
        segment_id (int): the integer related to the superprimitive to be cropped around
        image (np.ndarray): original colour image to be correspondingly cropped
        integer_segments (np.ndarray): segmentation masks in integer format (each segment made of a different integer)
        points (np.ndarray): 3D points in (H,W,3) format

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: cropped segmentation, image, and points arrays
    """
    if integer_segments.ndim != 2:
        raise ValueError("integer_segments must be 2-D (H, W).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be shape (H, W, 3)")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must be shape (H, W, 3)")
    if points.shape != image.shape:
        raise ValueError("points and image shapes must match")
    if points.shape[:2] != integer_segments.shape:
        raise ValueError("integer_segments shape must match image and points shapes")

    temp_segs = integer_segments.copy()
    
    # Locate all pixels in the desired segment
    mask = temp_segs == segment_id
    if not mask.any():
        raise ValueError(f"segment_id {segment_id} not found in integer_segments.")
    
    if make_binary:
        temp_segs[~mask] = 0
        temp_segs[mask] = 1
        temp_segs = temp_segs.astype(bool)
    
    # Bounding rows and columns
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]

    # Slice both arrays
    seg_crop = temp_segs[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1]
    img_crop =     image[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1]
    pts_crop =    points[r_min - border : r_max + border + 1, c_min - border : c_max + border + 1, :]

    return seg_crop, img_crop, pts_crop

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)