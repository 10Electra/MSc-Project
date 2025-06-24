import math
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from trimesh import Trimesh
from typing import Tuple, Union

def get_integer_segments(sp_regions):
    integer_segments = np.zeros([sp_regions.shape[1], sp_regions.shape[2]])
    for i in range(sp_regions.shape[0]):
        integer_segments[sp_regions[i,:,:]] = i + 1
    return integer_segments

def plot_region_numbers(integer_segments:np.ndarray,text_size=8):
    plt.figure(figsize=(10, 8))
    plt.imshow(integer_segments, cmap='nipy_spectral')

    # Calculate center of mass for each region and add text labels
    for region_id in range(1, int(np.max(integer_segments)) + 1):
        region_mask = (integer_segments == region_id)
        
        # Calculate center of mass
        if np.any(region_mask):  # Check if region exists
            y_coords, x_coords = np.where(region_mask)
            com_y = float(np.mean(y_coords))
            com_x = float(np.mean(x_coords))
            
            # Add text at center of mass
            plt.text(com_x, com_y, str(region_id), 
                    color='white', fontweight='bold', fontsize=text_size,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    plt.colorbar(label='Region ID')
    plt.title('Segmented Regions with Labels')
    plt.axis('off')
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

def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> Trimesh:
    trimesh_mesh = Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False)
    return trimesh_mesh

def trimesh_to_o3d(mesh: Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    
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

def calc_local_sampling_spacing(
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

def find_cyl_neighbours(point, normal, local_sampling_spacing, alpha, points, tree):
    """
    Find neighbouring points within a cylindrical region around a point.

    Args:
        point (np.ndarray): The query point (3,).
        normal (np.ndarray): The normal at the query point (3,).
        local_sampling_spacing (float): Local sampling spacing at the query point.
        alpha (float): Scaling factor for the cylinder; r = h/2 = alpha*LSS
        points (np.ndarray): All points in the cloud (N, 3).
        tree (o3d.geometry.KDTreeFlann): KDTree for the point cloud.

    Returns:
        np.ndarray: Indices of neighbouring points.
    """
    r_c = alpha * local_sampling_spacing
    h_half = alpha * local_sampling_spacing  # 0.5·h_c
    r2 = r_c * r_c
    h2 = h_half * h_half
    R = math.sqrt(r2 + h2)

    N, idx, d2 = tree.search_radius_vector_3d(point, R)
    idx = np.asarray(idx, dtype=np.int32)
    pts = points[idx]
    diff = pts - point

    h = np.abs(diff @ normal)                         # (N,)
    rad2 = np.maximum(d2 - h * h, 0.0)               # (N,)
    # Exclude the query point itself
    mask = (h <= h_half) & (rad2 <= r2) & (idx != np.where((points == point).all(axis=1))[0][0])

    neighbour_idx = idx[mask]
    return neighbour_idx