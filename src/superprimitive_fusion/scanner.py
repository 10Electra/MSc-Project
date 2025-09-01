import math
import numpy as np
import open3d as o3d  # type: ignore
from typing import Tuple, Dict, List
from perlin_noise import PerlinNoise

from superprimitive_fusion.utils import (
    polar2cartesian,
)

Vec3 = Tuple[float, float, float]

def _interpolate_vertex_colors(
        mesh: o3d.geometry.TriangleMesh,
        primitive_ids: np.ndarray,
        bary_uv: np.ndarray
    ) -> np.ndarray:
    """Return RGB at hit points: sample texture if available, else barycentric vertex colors."""
    assert primitive_ids.ndim == 1 and bary_uv.ndim == 2, "Must not be img shape"

    # Texture path
    has_uvs = getattr(mesh, "has_triangle_uvs", lambda: False)()
    has_textures = hasattr(mesh, "textures") and len(mesh.textures) > 0
    if has_uvs and has_textures:
        # (3 * n_tris, 2) -> (n_tris, 3, 2)
        tri_uvs_all = np.asarray(mesh.triangle_uvs, dtype=np.float32).reshape(-1, 3, 2)
        tri_uvs = tri_uvs_all[primitive_ids]  # (N, 3, 2) for the hit triangles

        # Barycentric -> UV on the triangle
        u, v = bary_uv[:, 0], bary_uv[:, 1]
        w = 1.0 - u - v
        hit_uv = (tri_uvs[:, 0, :] * w[:, None] +
                  tri_uvs[:, 1, :] * u[:, None] +
                  tri_uvs[:, 2, :] * v[:, None]).astype(np.float32)

        # Choose texture per triangle if available; else use texture 0 for all.
        if hasattr(mesh, "triangle_material_ids") and len(mesh.textures) > 1:
            tri_mats_all = np.asarray(mesh.triangle_material_ids, dtype=np.int32)
            mat_ids = tri_mats_all[primitive_ids]
        else:
            mat_ids = np.zeros(len(hit_uv), dtype=np.int32)

        out = np.zeros((len(hit_uv), 3), dtype=np.float32)

        # Sample each texture group (bilinear)
        unique_tex = np.unique(mat_ids)
        for tex_id in unique_tex:
            mask = (mat_ids == tex_id)
            if not np.any(mask):
                continue

            img = np.asarray(mesh.textures[int(tex_id)])
            # Ensure H * W * C
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            if img.shape[2] > 3:
                img = img[..., :3]  # drop alpha

            H, W = img.shape[0], img.shape[1]
            uv = np.clip(hit_uv[mask], 0.0, 1.0)
            x = uv[:, 0] * (W - 1)
            y = uv[:, 1] * (H - 1)

            x0 = np.floor(x).astype(np.int32)
            y0 = np.floor(y).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, W - 1)
            y1 = np.clip(y0 + 1, 0, H - 1)

            wa = (x1 - x) * (y1 - y)
            wb = (x - x0) * (y1 - y)
            wc = (x1 - x) * (y - y0)
            wd = (x - x0) * (y - y0)

            Ia = img[y0, x0, :].astype(np.float32)
            Ib = img[y0, x1, :].astype(np.float32)
            Ic = img[y1, x0, :].astype(np.float32)
            Id = img[y1, x1, :].astype(np.float32)

            col = (Ia * wa[:, None] + Ib * wb[:, None] +
                   Ic * wc[:, None] + Id * wd[:, None]) / 255.0
            out[mask] = col

        return out

    # --- Fallback: per-vertex color interpolation ---
    vcols = np.asarray(mesh.vertex_colors, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.int32)[primitive_ids]
    c0, c1, c2 = vcols[tris[:, 0]], vcols[tris[:, 1]], vcols[tris[:, 2]]
    u, v = bary_uv[:, 0], bary_uv[:, 1]
    w = 1.0 - u - v
    return w[:, None] * c0 + u[:, None] * c1 + v[:, None] * c2

def virtual_scan(
        meshlist:   list[o3d.geometry.TriangleMesh],
        cam_centre: np.ndarray | tuple,
        look_at:    np.ndarray | tuple,
        width_px:   int = 360,
        height_px:  int = 240,
        fov:        float = 70.0,
        up:         list = [0,0,1],
    ) -> dict:

    cam_centre_lst  = list(cam_centre) if isinstance(cam_centre, tuple) else cam_centre.tolist()
    look_at_lst     = list(look_at)    if isinstance(look_at, tuple)    else look_at.tolist()

    scene = o3d.t.geometry.RaycastingScene()

    o3d_geom_id_list = []
    for mesh in meshlist:
        id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        o3d_geom_id_list.append(id)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov,
        center=look_at_lst,
        eye=cam_centre_lst,
        up=up,
        width_px=width_px,
        height_px=height_px,
    )

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    # Intersection metadata (gometry id + triangle id + barycentric uv + normals)
    geom_ids_img = ans["geometry_ids"].numpy().astype(np.int16)
    prim_ids = ans["primitive_ids"].numpy().astype(np.int32)
    bary_uv = ans.get("primitive_uvs", None).numpy()
    normals = ans["primitive_normals"].numpy()

    # IDs in the same order as 'meshlist'
    o3d_geom_ids = np.asarray(o3d_geom_id_list, dtype=np.uint32)   # [id_A, id_B, id_C, ...]

    # Lookup: geometry_id -> meshlist index
    id_to_idx = np.full(int(o3d_geom_ids.max()) + 1, -1, dtype=np.int32)
    id_to_idx[o3d_geom_ids] = np.arange(o3d_geom_ids.size, dtype=np.int32)

    # Remap the (H, W) image of geometry_ids to [0..len(meshlist)-1], with -1 for misses
    invalid = o3d.t.geometry.RaycastingScene.INVALID_ID
    hit_mask = geom_ids_img != invalid

    mesh_idx_img = np.full(geom_ids_img.shape, -1, dtype=np.int32)
    mesh_idx_img[hit_mask] = id_to_idx[geom_ids_img[hit_mask]]
    
    rays_np = rays.numpy()  # (H*W,6)
    origins = rays_np[..., :3]
    dirs = rays_np[..., 3:]
    verts = origins + dirs * t_hit[..., None]
    

    # Assertions for bugfixing
    assert bary_uv is not None
    for mesh in meshlist:
        assert mesh.has_vertex_colors()

    vcols = np.full((*t_hit.shape, 3), 0.5)
    for id in range(len(meshlist)):
        rel_prim_ids = prim_ids[mesh_idx_img==id]
        rel_bary_uv  =  bary_uv[mesh_idx_img==id]
        vcols[mesh_idx_img==id] = _interpolate_vertex_colors(meshlist[id], rel_prim_ids, rel_bary_uv)
    
    scan = dict()
    scan['verts'] = verts
    scan['vcols'] = vcols
    scan['norms'] = normals
    scan['segmt'] = mesh_idx_img

    return scan


def triangulate_rgbd_grid_grouped(verts, valid, z, obj_id,
                                  k=3.5, normals=None, max_normal_angle_deg=None):
    """
    Output: list of (Ni,3) int32 arrays. Index i corresponds to object_id == i.
            Unknown id == -1 is skipped.
    """
    # --- reshape ---
    if verts.ndim == 2:
        N = verts.shape[0]
        H = int(np.round(np.sqrt(N)))
        W = N // H
        P = verts.reshape(H, W, 3)
    else:
        H, W, _ = verts.shape
        P = verts
    valid_img = valid.reshape(H, W)
    z_img = z.reshape(H, W)
    obj_id = obj_id.reshape(H, W).astype(np.int32)

    # --- disparity + robust thresholds ---
    disp = np.zeros_like(z_img, dtype=np.float32)
    m = valid_img & np.isfinite(z_img) & (z_img > 0)
    disp[m] = 1.0 / z_img[m]

    dx  = np.abs(disp[:, 1:] - disp[:, :-1])
    dy  = np.abs(disp[1:, :] - disp[:-1, :])
    dd1 = np.abs(disp[:-1, :-1] - disp[1:, 1:])    # tl-br
    dd2 = np.abs(disp[:-1, 1:]  - disp[1:, :-1])   # tr-bl

    mask_x  = valid_img[:, 1:] & valid_img[:, :-1]
    mask_y  = valid_img[1:, :] & valid_img[:-1, :]
    mask_d1 = valid_img[:-1, :-1] & valid_img[1:, 1:]
    mask_d2 = valid_img[:-1, 1:]  & valid_img[1:, :-1]

    vals = np.concatenate([
        dx[mask_x].ravel(), dy[mask_y].ravel(),
        (dd1[mask_d1] / np.sqrt(2)).ravel(),
        (dd2[mask_d2] / np.sqrt(2)).ravel()
    ])
    if vals.size:
        med = np.median(vals)
        mad = 1.4826 * np.median(np.abs(vals - med))
        base_thr = med + k * mad if mad > 0 else med * 1.5
    else:
        base_thr = np.inf
    thr_x, thr_y, thr_d = base_thr, base_thr, base_thr * np.sqrt(2)

    # --- object gates (unknown == -1 → cut) ---
    same_x  = (obj_id[:, 1:] == obj_id[:, :-1]) & (obj_id[:, 1:] >= 0)
    same_y  = (obj_id[1:, :] == obj_id[:-1, :]) & (obj_id[1:, :] >= 0)
    same_d1 = (obj_id[:-1, :-1] == obj_id[1:, 1:]) & (obj_id[:-1, :-1] >= 0)
    same_d2 = (obj_id[:-1, 1:]  == obj_id[1:, :-1]) & (obj_id[:-1, 1:]  >= 0)

    # --- base edge validity ---
    good_x  = mask_x  & same_x  & (dx  <= thr_x)
    good_y  = mask_y  & same_y  & (dy  <= thr_y)
    good_d1 = mask_d1 & same_d1 & (dd1 <= thr_d)
    good_d2 = mask_d2 & same_d2 & (dd2 <= thr_d)

    # --- optional normal-angle gate ---
    if (normals is not None) and (max_normal_angle_deg is not None):
        n = normals.reshape(H, W, 3).astype(np.float32)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
        ok = n_norm[..., 0] > 0
        n = np.where(ok[..., None], n / np.clip(n_norm, 1e-12, None), 0.0)

        cos_max = np.cos(np.deg2rad(max_normal_angle_deg))
        dot_x  = (n[:, 1:, :] * n[:, :-1, :]).sum(-1)
        dot_y  = (n[1:, :, :] * n[:-1, :, :]).sum(-1)
        dot_d1 = (n[:-1, :-1, :] * n[1:, 1:, :]).sum(-1)
        dot_d2 = (n[:-1, 1:, :]  * n[1:, :-1, :]).sum(-1)

        n_ok_x  = ok[:, 1:]  & ok[:, :-1]
        n_ok_y  = ok[1:, :]  & ok[:-1, :]
        n_ok_d1 = ok[:-1, :-1] & ok[1:, 1:]
        n_ok_d2 = ok[:-1, 1:]  & ok[1:, :-1]

        good_x  &= n_ok_x  & (dot_x  >= cos_max)
        good_y  &= n_ok_y  & (dot_y  >= cos_max)
        good_d1 &= n_ok_d1 & (dot_d1 >= cos_max)
        good_d2 &= n_ok_d2 & (dot_d2 >= cos_max)

    # --- per-quad indices ---
    id_img = np.arange(H * W, dtype=np.int32).reshape(H, W)
    tl = id_img[:-1, :-1]; tr = id_img[:-1, 1:]
    bl = id_img[1:, :-1];  br = id_img[1:, 1:]

    # per-quad edges
    e_top    = good_x[:H-1, :W-1]
    e_bottom = good_x[1:,  :W-1]
    e_left   = good_y[:H-1, :W-1]
    e_right  = good_y[:H-1, 1:]
    d1, d2 = good_d1, good_d2

    # candidate triangles by diagonal
    A = e_top & e_right & d1            # (tl,tr,br)
    B = e_left & e_bottom & d1          # (tl,br,bl)
    C = e_top & e_left & d2             # (tl,tr,bl)
    D = e_right & e_bottom & d2         # (tr,br,bl)

    # orientation selection (prefer keeping more tris; tie → shorter 3D diagonal)
    count1 = A.astype(np.uint8) + B.astype(np.uint8)
    count2 = C.astype(np.uint8) + D.astype(np.uint8)

    d1_len2 = np.full((H-1, W-1), np.inf, dtype=P.dtype)
    d2_len2 = np.full((H-1, W-1), np.inf, dtype=P.dtype)
    fin_d1 = np.isfinite(P[:-1, :-1, :]).all(-1) & np.isfinite(P[1:, 1:, :]).all(-1)
    if fin_d1.any():
        diff = P[:-1, :-1, :][fin_d1] - P[1:, 1:, :][fin_d1]
        d1_len2[fin_d1] = (diff * diff).sum(-1)
    fin_d2 = np.isfinite(P[:-1, 1:, :]).all(-1) & np.isfinite(P[1:, :-1, :]).all(-1)
    if fin_d2.any():
        diff = P[:-1, 1:, :][fin_d2] - P[1:, :-1, :][fin_d2]
        d2_len2[fin_d2] = (diff * diff).sum(-1)

    prefer_d1 = (count1 > count2) | ((count1 == count2) & (d1_len2 <= d2_len2))
    A &= prefer_d1; B &= prefer_d1
    C &= ~prefer_d1; D &= ~prefer_d1

    # --- pack triangles + labels (label from any vertex in the tri; they’re equal by construction) ---
    lab_tl = obj_id[:-1, :-1]
    lab_tr = obj_id[:-1, 1:]
    # lab_bl = obj_id[1:, :-1]
    # lab_br = obj_id[1:, 1:]

    tris_list, labels_list = [], []

    def pack(mask, i0, i1, i2, lab_grid):
        if mask.any():
            tris_list.append(np.stack([i0[mask], i1[mask], i2[mask]], axis=1))
            labels_list.append(lab_grid[mask].astype(np.int32))

    pack(A, tl, br, tr, lab_tl)  # (tl,br,tr) → label from tl
    pack(B, tl, bl, br, lab_tl)  # (tl,bl,br) → label from tl
    pack(C, tl, bl, tr, lab_tl)  # (tl,bl,tr) → label from tl
    pack(D, tr, bl, br, lab_tr)  # (tr,bl,br) → label from tr

    if not tris_list:
        # no triangles at all; build empty list sized by max obj id
        max_obj = int(obj_id.max())
        return [] if max_obj < 0 else [np.empty((0, 3), dtype=np.int32) for _ in range(max_obj + 1)]

    tris_all   = np.concatenate(tris_list, axis=0).astype(np.int32)
    labels_all = np.concatenate(labels_list, axis=0)

    # drop unknown ids
    keep = labels_all >= 0
    tris_all = tris_all[keep]
    labels_all = labels_all[keep]

    # size output by maximum declared object id (so objects with 0 tris still get an empty array)
    max_obj = int(obj_id.max())
    if max_obj < 0:
        return []

    object_tris = [np.empty((0, 3), dtype=np.int32) for _ in range(max_obj + 1)]
    if labels_all.size == 0:
        return object_tris

    # group by label (vectorized; no per-triangle Python loops)
    order = np.argsort(labels_all, kind='mergesort')
    labels_sorted = labels_all[order]
    tris_sorted = tris_all[order]

    ids, starts, counts = np.unique(labels_sorted, return_index=True, return_counts=True)
    for obj, s, c in zip(ids, starts, counts):
        if obj >= 0:
            object_tris[int(obj)] = tris_sorted[s:s + c]

    return object_tris


def mesh_depth_image(
    points:         np.ndarray,                 # (H,W,3)
    weights:        np.ndarray,                 # (H,W) or (H*W,)
    vertex_colours: np.ndarray | None,          # (H,W,3) or (H*W,3) or None
    z:              np.ndarray,                 # (H,W) depth along camera look axis
    segmentation:   np.ndarray | None = None,   # (H,W) int; unknown == -1
    normals:        np.ndarray | None = None,   # (H,W,3) or None
    k:              float = 3.5,
    max_normal_angle_deg: float | None = None,
):
    """
    Returns:
        meshes:       list[o3d.geometry.TriangleMesh]  (index i == object_id i)
        weights_list: list[np.ndarray]                 (per-mesh compact vertex weights)
    Notes:
      - Triangles are built per object id via triangulate_rgbd_grid_grouped(...).
      - Inf/NaN pixels are ignored by the triangulator; they never appear in meshes.
      - Winding is left as produced by the triangulator; flip if your renderer requires it.
    """
    assert points.ndim == 3 and points.shape[2] == 3
    H, W, _ = points.shape
    N = H * W

    # Flatten colors to (N,3) if provided
    if vertex_colours is not None:
        if vertex_colours.ndim == 3:
            vertex_colours = vertex_colours.reshape(-1, 3)
        else:
            assert vertex_colours.shape == (N, 3)

    # Flatten weights to (N,)
    if weights.ndim == 2:
        weights = weights.reshape(-1)
    else:
        assert weights.shape == (N,)

    # Segmentation: default single object 0; drop unknown == -1
    if segmentation is None:
        segmentation = np.zeros((H, W), dtype=np.int32)
    else:
        assert segmentation.shape == (H, W)
        segmentation = segmentation.astype(np.int32, copy=False)

    # Normals (only used by triangulation if given)
    if normals is not None:
        assert normals.shape == (H, W, 3)

    # Validity mask (finite and not unknown)
    valid_img = np.isfinite(points).all(axis=2) & (segmentation != -1)

    # Per-object triangulation in grid index space (indices in [0..H*W))
    object_tris = triangulate_rgbd_grid_grouped(
        verts=points,
        valid=valid_img,
        z=z,
        obj_id=segmentation,
        k=k,
        normals=normals,
        max_normal_angle_deg=max_normal_angle_deg,
    )

    P_flat = points.reshape(-1, 3)
    W_flat = weights
    meshes: list[o3d.geometry.TriangleMesh] = []
    weights_list: list[np.ndarray] = []

    # Fast O(E) remap: global index_map reused per object
    index_map = np.full(N, -1, dtype=np.int64)

    for tris in object_tris:
        if tris.size == 0:
            meshes.append(o3d.geometry.TriangleMesh())
            weights_list.append(np.empty((0,), dtype=weights.dtype))
            continue

        tris = np.asarray(tris, dtype=np.int64, order='C')     # (M,3), grid indices

        # Unique referenced grid pixels (sorted)
        ref_idx = np.unique(tris.ravel())                      # (V_o,)

        # Compact vertices/weights/colors in the same order
        V = P_flat[ref_idx]
        w = W_flat[ref_idx]
        if vertex_colours is not None:
            C = vertex_colours[ref_idx]
        else:
            C = None

        # Old(grid)->new(compact) remap in O(1)
        index_map[ref_idx] = np.arange(ref_idx.size, dtype=np.int64)
        tris_compact = index_map[tris]                          # (M,3), compact indices
        index_map[ref_idx] = -1                                 # reset touched entries

        # Build mesh
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(V.astype(np.float64, copy=False))
        m.triangles = o3d.utility.Vector3iVector(tris_compact.astype(np.int32, copy=False))
        if C is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float64, copy=False))

        meshes.append(m)
        weights_list.append(w.astype(weights.dtype, copy=False))

    return meshes, weights_list


def clean_mesh_and_remap_weights(mesh, w):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    tris = np.asarray(mesh.triangles, dtype=np.int64)
    ref = np.zeros(len(mesh.vertices), dtype=bool)
    if tris.size:
        ref[tris.ravel()] = True

    mesh.remove_vertices_by_mask(~ref)
    w = w[ref]
    mesh.compute_vertex_normals()
    return mesh, w


def generate_rgbd_noise_moge_like(
    verts_img:  np.ndarray,               # (H,W,3) or (N,3); may contain inf/NaN
    cam_centre: np.ndarray | tuple,
    look_dir:   np.ndarray | tuple,
    normals_img:np.ndarray | None = None, # optional per-vertex normals (unit)
    *,
    # log-range noise: sigma_log = (a0 + a1 * (rho/median_rho)) * f_edge * f_grazing
    a0: float = 0.010,
    a1: float = 0.035,
    edge_mult: float = 3.0,
    grazing_lambda: float = 1.0,          # f_grazing = 1 + lambda(1 - cos(theta)), internally clamped
    corr_std_log: float = 0.015,          # low-frequency field std (log units)
    corr_kernel_size: int = 9,            # size of correlated noise kernel
    sigma_log_floor: float = 0.007,       # floor on sigma_log (~0.7% relative)
    tau_cap: float = 1.0 / (0.003**2),    # cap on precision
    bias_k1: float = 0.0,                 # optional small radial bow (bias only)
    fov_deg: float = 70.0,
    seed: int | None = None,
):
    """
    Returns:
      verts_noised : same shape as verts_img, with inf/NaN preserved
      weights      : (H,W) if input is (H,W,3), else (N,), tau = 1/((rho*sigma_log)^2); 0 for invalid
    """
    rng = np.random.default_rng(seed)

    V = np.asarray(verts_img)
    orig_dtype = V.dtype
    is_image = (V.ndim == 3)

    V_flat = V.reshape(-1, 3).astype(np.float64, copy=False)
    N = V_flat.shape[0]

    valid = np.isfinite(V_flat).all(axis=1)
    if not np.any(valid):
        weights = np.zeros(V.shape[:2] if is_image else (N,), dtype=np.float32)
        return V.copy(), weights

    Vv = V_flat[valid]

    # Camera geometry
    C = np.asarray(cam_centre, dtype=np.float64).reshape(1, 3)
    L = np.asarray(look_dir,   dtype=np.float64).reshape(3)
    L /= (np.linalg.norm(L) + 1e-12)

    R   = Vv - C
    rho = np.linalg.norm(R, axis=1)                   # range along ray
    ray = R / np.maximum(rho[:, None], 1e-12)         # ray directions
    d_ax = np.maximum(R @ L, 1e-12)                   # axial component for bias calc

    # Grazing factor from normals (preferred) or fallback to view dir
    if normals_img is not None:
        Nv = np.asarray(normals_img).reshape(-1, 3)[valid].astype(np.float64, copy=False)
        Nv /= (np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-12)
        cos_th = np.abs(np.sum(Nv * (-ray), axis=1))
    else:
        cos_th = np.abs(ray @ (-L))
    f_graz = 1.0 + grazing_lambda * (1.0 - np.clip(cos_th, 0.0, 1.0))
    f_graz = np.minimum(f_graz, 3.0)                  # clamp

    # Base sigma in log-range
    rho_med = np.median(rho) if rho.size else 1.0
    rho_til = rho / max(rho_med, 1e-12)
    sigma_log = (a0 + a1 * rho_til)

    # Structure: depth-edge inflation (image mode only)
    if is_image:
        H, W = V.shape[:2]
        rho_img = np.zeros((H, W), dtype=np.float64); rho_img.flat[valid] = rho

        # depth gradient magnitude (central differences)
        gx = np.zeros_like(rho_img); gy = np.zeros_like(rho_img)
        gx[:, 1:-1] = 0.5*(rho_img[:, 2:] - rho_img[:, :-2])
        gy[1:-1, :] = 0.5*(rho_img[2:, :] - rho_img[:-2, :])
        g = np.sqrt(gx*gx + gy*gy).reshape(-1)[valid]

        # normalise to [0,1] using robust scale (90th percentile)
        g90 = np.percentile(g, 90.0) if g.size else 1.0
        g_norm = np.clip(g / (g90 + 1e-12), 0.0, 1.0)

        f_edge = 1.0 + (edge_mult - 1.0) * g_norm
        sigma_log *= f_edge

        # Low-frequency correlated field (not reflected in tau)
        if corr_std_log > 0.0:
            lf = rng.normal(0.0, 1.0, size=(H, W))

            # Box blur (k odd), output SAME size using integral image
            k = corr_kernel_size
            pad = k // 2
            # Pad, build integral image
            lf_pad = np.pad(lf, pad_width=pad, mode='edge')
            ii = lf_pad.cumsum(0).cumsum(1)

            # For each (i,j) in original image, sum over the kxk window centred at (i,j)
            # Using integral image identities with proper offsets gives an HxW result.
            # Coordinates in ii are offset by +pad.
            sum_k = (
                ii[2*pad:2*pad+H, 2*pad:2*pad+W]
                - ii[:H,           2*pad:2*pad+W]
                - ii[2*pad:2*pad+H, :W]
                + ii[:H,           :W]
            )
            lf_blur = sum_k / (k * k)  # HxW, aligned with lf

            # Normalise std over valid pixels to 1, then scale to corr_std_log
            lf_v = lf_blur.reshape(-1)[valid]
            s = np.std(lf_v) if lf_v.size else 1.0

            corr_field_log = np.zeros(N, dtype=np.float64)
            corr_field_log[valid] = (lf_blur.reshape(-1)[valid] / (s + 1e-12)) * corr_std_log
        else:
            corr_field_log = np.zeros(N, dtype=np.float64)
    else:
        corr_field_log = np.zeros(N, dtype=np.float64)

    # Apply grazing and floor
    sigma_log *= f_graz
    sigma_log = np.maximum(sigma_log, sigma_log_floor)

    # Sample additive log-range noise (independent component)
    eta = np.random.default_rng(seed).normal(0.0, sigma_log)

    # New range with correlated field (bias added later)
    z = np.log(np.maximum(rho, 1e-12))
    z_noisy = z + eta + corr_field_log[valid]
    rho_noisy = np.exp(z_noisy)

    # Optional small radial bow bias along the ray
    if bias_k1 != 0.0:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(up_guess, L)) > 0.95:
            up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(L, up_guess); right /= (np.linalg.norm(right) + 1e-12)
        up    = np.cross(right, L)
        x = (R @ right) / d_ax; y = (R @ up) / d_ax
        scale = np.tan(np.deg2rad(fov_deg) * 0.5)
        r2 = (x/scale)**2 + (y/scale)**2
        rho_noisy += (bias_k1 * rho * r2)

    # Write back
    V_out = V_flat.copy()
    V_out[valid] = (C + ray * rho_noisy[:, None]).astype(orig_dtype, copy=False)
    V_out_img = V_out.reshape(V.shape)

    # Weights (exclude corr field; include edge + grazing) with global inflation and cap
    sigma_ax = rho * sigma_log              # per-pixel axial std in metres
    tau_ind_valid = 1.0 / (sigma_ax**2)     # m^-2

    weights_flat = np.zeros(N, dtype=np.float32)
    weights_flat[valid] = np.minimum(tau_ind_valid, tau_cap).astype(np.float32)

    # reshape to image if needed
    weights = weights_flat.reshape(V.shape[:2] if is_image else (-1,))
    return V_out_img, weights


def generate_rgbd_noise(
    verts_img:      np.ndarray,                 # (H,W,3); may contain inf/NaN
    cam_centre:     np.ndarray | tuple,
    look_dir:       np.ndarray | tuple,
    normals_img:    np.ndarray,                 # (H,W,3)
    segmt_img:      np.ndarray,                 # (H,W) integer labels
    *,
    con_unc:        float = 2e-4,   # uncertainty constant term
    lin_unc:        float = 2e-4,   # uncertainty linear term
    qdr_unc:        float = 2e-4,   # uncertainty quadratic term
    con_perlin:     float = 2e-4,   # constant perlin noise over depth
    lin_perlin:     float = 2e-4,   # perlin linear depth term
    qdr_perlin:     float = 2e-4,   # perlin quadratic depth term
    oct_perlin:     int = 3,        # perlin number of octaves
    seg_scale_std:  float = 0.1,    # per-segment scale noise std (scalefac = 1 + N(0,std))
    rot_std:        float = 0.1,    # global rotation noise std (radians, per-axis small-angle)
    trn_std:        float = 0.1,    # global translation noise std (meters, per-axis)
    grazing_lambda: float = 1.0,    # noise factor for surfaces misaligned with camera (None=0)
    seed: int | None = None,
):
    """
    Returns:
      verts_noised : same shape as verts_img, with inf/NaN preserved
      weights      : (H,W) precision along ray; 0 for invalid
    """
    # --- Assumptions (simplify!) ---
    assert verts_img.ndim == 3 and verts_img.shape[2] == 3
    assert normals_img.shape == verts_img.shape
    assert segmt_img.shape[:2] == verts_img.shape[:2]

    rng = np.random.default_rng(seed)
    H, W, _ = verts_img.shape

    # Flatten
    V = np.asarray(verts_img, dtype=np.float64)
    N = np.asarray(normals_img, dtype=np.float64)
    S = np.asarray(segmt_img)
    V_flat = V.reshape(-1, 3)
    N_flat = N.reshape(-1, 3)
    S_flat = S.reshape(-1)

    # Valid mask (all coords finite)
    valid = np.isfinite(V_flat).all(axis=1)
    if not np.any(valid):
        return verts_img.copy(), np.zeros((H, W), dtype=np.float32)

    # Camera vectors
    C = np.asarray(cam_centre, dtype=np.float64).reshape(1, 3)
    L = np.asarray(look_dir,   dtype=np.float64).reshape(3)
    L /= (np.linalg.norm(L) + 1e-12)

    # Geometry for valid points
    Vv = V_flat[valid]
    Nv = N_flat[valid]
    Sv = S_flat[valid]

    R_cam = Vv - C                                # camera -> point
    d = (R_cam @ L)                               # depth along camera forward
    d = np.maximum(d, 0.0)
    Rn = np.linalg.norm(R_cam, axis=1, keepdims=True)
    ray_dir = R_cam / np.maximum(Rn, 1e-12)       # unit viewing ray

    # Grazing term
    Nv = Nv / (np.linalg.norm(Nv, axis=1, keepdims=True) + 1e-12)
    cos_th = np.abs(np.sum(Nv * (-ray_dir), axis=1))
    grazing_boost = 1.0 + grazing_lambda * (1.0 - cos_th)

    # Axial std dev (meters)
    sigma_z = (con_perlin + lin_perlin * d + qdr_perlin * (d ** 2)) * grazing_boost

    uncertainty = (con_unc + lin_unc * d + qdr_unc * (d**2)) * grazing_boost
    uncertainty = np.clip(uncertainty, a_min=1e-8, a_max=None)

    # Perlin axial noise (image grid -> valid indices)
    P = PerlinNoise(octaves=oct_perlin, seed=int(rng.integers(0, 2**31 - 1)))
    p = np.empty((H, W), dtype=np.float64)
    # x = columns/W, y = rows/H
    for i in range(H):
        for j in range(W):
            p[i, j] = P([j / float(W), i / float(H)])

    p_valid = p.reshape(-1)[valid]
    mu, sd = p_valid.mean(), p_valid.std()
    z = np.zeros_like(p_valid) if sd < 1e-12 else (p_valid - mu) / sd

    axial_disp = (z * sigma_z)[:, None] * ray_dir   # (M,3)

    # Per-segment isotropic scaling about camera centre
    if seg_scale_std > 0.0:
        labels, inv = np.unique(Sv, return_inverse=True)
        scalefacs = 1.0 + rng.normal(0.0, seg_scale_std, size=labels.shape[0])
        s = scalefacs[inv]  # (M,)
        scale_disp = ((s - 1.0)[:, None]) * R_cam   # scale about C
    else:
        scale_disp = 0.0

    Vv_noised = Vv + axial_disp + scale_disp

    # Global rigid perturbation (single R,t for the whole image)
    if (rot_std > 0.0) or (trn_std > 0.0):
        r = rng.normal(0.0, rot_std, size=3)
        theta = np.linalg.norm(r)
        if theta < 1e-12:
            Rmat = np.eye(3, dtype=np.float64)
        else:
            k = r / theta
            K = np.array([[0.0,   -k[2],  k[1]],
                          [k[2],   0.0,  -k[0]],
                          [-k[1],  k[0],  0.0]], dtype=np.float64)
            Rmat = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        t = rng.normal(0.0, trn_std, size=3).astype(np.float64)
        Vv_noised = C + (Vv_noised - C) @ Rmat.T + t

    # Write back, preserve invalids
    V_out = V_flat.copy()
    V_out[valid] = Vv_noised
    V_out_img = V_out.reshape(H, W, 3).astype(verts_img.dtype, copy=False)

    # Precision weights from axial noise only; 0 for invalid
    w = np.zeros(V_flat.shape[0], dtype=np.float32)
    w[valid] = (1.0 / (uncertainty**2)).astype(np.float32)
    weights = w.reshape(H, W)

    return V_out_img, weights



def virtual_mesh_scan(
    meshlist:               list[o3d.geometry.TriangleMesh],
    cam_centre:             np.ndarray | tuple,
    look_at:                np.ndarray | tuple,
    k:                      float,
    max_normal_angle_deg:   float|None,
    width_px:               int=360,
    height_px:              int=240,
    fov:                    float=70,
    constant_uncertainty:   float=2e-4,
    linear_uncertainty:     float=3e-3,     # rate of uncertainty increase with depth
    quadrt_uncertainty:     float=1e-3,     # quadratic uncertainty coefficient
    constant_perlin_sigma:  float=2e-4,     # constant perlin noise term
    linear_perlin_sigma:    float=3e-3,     # linear depth term
    quadrt_perlin_sigma:    float=1e-3,     # quadratic depth term
    perlin_octaves:         int=3,
    seg_scale_std:          float=1e-4,      # std of per-segment scale noise
    rot_std:                float=1e-4,      # std of global rotation noise
    trn_std:                float=1e-3,      # std of global translation noise
    grazing_lambda:         float=1.0,      # sigma multiplier at grazing angles; 0 disables
    seed = None,
    include_depth_image:    bool=False,
) -> tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray]]|tuple[list[o3d.geometry.TriangleMesh], list[np.ndarray], dict]:
    """Easy call to virtual_scan() and mesh_depth_mage()"""

    cam_centre_np = np.asarray(cam_centre) if isinstance(cam_centre, tuple) else cam_centre
    look_at_np = np.asarray(look_at) if isinstance(look_at, tuple) else look_at
    up = [0, 0, 1]
    
    scan_result = virtual_scan(
        meshlist,
        cam_centre_np,
        look_at_np,
        width_px=width_px,
        height_px=height_px,
        fov=fov,
        up=up,
    )

    look_dir = look_at_np - cam_centre_np
    look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-12)

    K, E = build_K_E_from_lookat_and_rays(
        cam_centre=cam_centre_np,
        look_at=look_at_np,
        width_px=width_px,
        height_px=height_px,
        hfov_deg=fov,
        up_world=up,
    )
    
    verts_noised, weights = generate_rgbd_noise(
        verts_img=scan_result['verts'],
        cam_centre=cam_centre,
        look_dir=look_dir,
        normals_img=scan_result['norms'],
        segmt_img=scan_result['segmt'],
        
        con_unc=constant_uncertainty,
        lin_unc=linear_uncertainty,
        qdr_unc=quadrt_uncertainty,
        con_perlin=constant_perlin_sigma,
        lin_perlin=linear_perlin_sigma,
        qdr_perlin=quadrt_perlin_sigma,
        oct_perlin=perlin_octaves,
        seg_scale_std=seg_scale_std,
        rot_std=rot_std,
        trn_std=trn_std,
        grazing_lambda=grazing_lambda,
        seed=seed,
    )

    L = look_dir / np.linalg.norm(look_dir)
    depth = ((verts_noised - cam_centre_np) @ L).clip(min=0.0)
    depth_image = {'depth':depth, 'rgb':scan_result['vcols'], 'E':E, 'K_t':K}

    object_meshes, object_weights = mesh_depth_image(
        points=verts_noised,
        weights=weights,
        vertex_colours=scan_result['vcols'],
        z=depth,
        segmentation=scan_result['segmt'],
        normals=scan_result['norms'],
        k=k,
        max_normal_angle_deg=max_normal_angle_deg,
    )
    if include_depth_image:
        return object_meshes, object_weights, depth_image
    return object_meshes, object_weights


def fibonacci_sphere_points(n: int, radius: float) -> np.ndarray:
    """Roughly even points on a sphere using the Fibonacci lattice."""
    pts = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - (2*i + 1)/n   # y in (-1,1)
        r_xy = math.sqrt(max(0.0, 1 - y*y))
        theta = i * golden_angle
        x = math.cos(theta) * r_xy
        z = math.sin(theta) * r_xy
        pts.append((radius*x, radius*y, radius*z))
    return np.asarray(pts)

def capture_spherical_scans(
    meshlist:               list[o3d.geometry.TriangleMesh],
    num_views:              int = 6,
    radius:                 float = 0.3,
    look_at:                Vec3 = (0.0, 0.0, 0.0),
    cam_offset:             np.ndarray = np.array([0.0, 0.0, 0.0]),
    width_px:               int = 180,
    height_px:              int = 120,
    fov:                    float = 70.0,
    k:                      float = 3.5,
    max_normal_angle_deg:   float | None = None,
    sampler:                str = "fibonacci",  # or "latlong"
    constant_uncertainty:   float=2e-4,
    linear_uncertainty:     float=3e-3,
    quadrt_uncertainty:     float=1e-3,
    constant_perlin_sigma:  float=2e-4,         # constant perlin noise
    linear_perlin_sigma:    float=3e-3,         # linear depth term
    quadrt_perlin_sigma:    float=1e-3,         # quadratic depth term
    perlin_octaves:         int=3,              # octaves =~ scale of perlin noise
    seg_scale_std:          float=0.1,          # per segment scale noise std
    rot_std:                float=0.1,          # global rotation noise std
    trn_std:                float=0.1,          # global translation noise std
    sigma_floor:            float=0.00015,      # prevents infinite weights
    grazing_lambda:         float=1.0,          # sigma multiplier at grazing angles; 0 disables
    include_depth_images:   bool=False,
    scan_lat:               float | None = None # latitude for 'latlong' sampling
) -> List[Dict[str, object]]:
    """
    Capture RGB-D scans from evenly spaced viewpoints on a sphere.

    Returns a list of dicts with keys:
      - 'mesh': mesh_scan
      - 'pcd': pcd_scan (if return_pcd=True)
      - 'cam_centre': Vec3
      - 'look_at': Vec3
    """
    # Sample viewpoints
    if sampler == "fibonacci":
        cam_centres = list(cam_offset + fibonacci_sphere_points(num_views, radius))
    elif sampler == "latlong":
        cam_centres = []
        for i in range(num_views):
            lat = scan_lat if scan_lat is not None else 90.0
            lon = (360/num_views) * i
            cam_centres.append(cam_offset + polar2cartesian(r=radius, lat=lat, long=lon))
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")

    scans = []
    for c in cam_centres:
        scan = virtual_mesh_scan(
            meshlist=meshlist,
            cam_centre=c,
            look_at=look_at,
            k=k,
            max_normal_angle_deg=max_normal_angle_deg,
            width_px=width_px,
            height_px=height_px,
            fov=fov,
            constant_uncertainty=constant_uncertainty,
            linear_uncertainty=linear_uncertainty,
            quadrt_uncertainty=quadrt_uncertainty,
            constant_perlin_sigma=constant_perlin_sigma,
            linear_perlin_sigma=linear_perlin_sigma,
            quadrt_perlin_sigma=quadrt_perlin_sigma,
            perlin_octaves=perlin_octaves,
            seg_scale_std=seg_scale_std,
            rot_std=rot_std,
            trn_std=trn_std,
            grazing_lambda=grazing_lambda,
            seed=None,
            include_depth_image=include_depth_images,
        )
        record = {
            "mesh": scan,
            "cam_centre": c,
            "look_at": look_at,
        }
        scans.append(record)
    return scans

def render_rgb_view(
    meshes: list[o3d.geometry.TriangleMesh],
    cam_centre=(0.3, 0.3, 0.0),
    look_at=(0.0, 0.0, 0.0),
    *,
    width_px: int = 360,
    height_px: int = 240,
    fov: float = 70.0,
    bg_rgba=(1.0, 1.0, 1.0, 1.0),  # white background
) -> np.ndarray:
    """Render an RGB image of a scene defined by `meshes` from a given camera pose."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width_px, height_px)
    renderer.scene.set_background(bg_rgba)

    # Add all meshes with simple per-mesh materials
    for i, mesh in enumerate(meshes):
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = "defaultUnlit" if mesh.has_vertex_colors() else "defaultLit"
        renderer.scene.add_geometry(f"mesh_{i}", mesh, mtl)

    # Camera: vertical FOV in degrees, look-at with Z-up
    aspect = width_px / height_px
    cam = renderer.scene.camera
    cam.set_projection(
        fov, aspect, 0.01, 1000.0,
        o3d.visualization.rendering.Camera.FovType.Vertical
    )
    cam.look_at(look_at, cam_centre, [0, 0, 1])

    # Render to image and return RGB
    img = renderer.render_to_image()
    rgb = np.asarray(img)
    if rgb.ndim == 3 and rgb.shape[2] == 4:  # drop alpha if present
        rgb = rgb[:, :, :3]
    return rgb

def build_K_E_from_lookat_and_rays(
    cam_centre: np.ndarray,
    look_at:    np.ndarray,
    width_px:   int,
    height_px:  int,
    hfov_deg:   float,
    up_world=(0.0, 0.0, 1.0),
):
    """
    Returns:
      K : (3,3) intrinsics (pixel units). Assumes square pixels; fx=fy from H-FOV.
      E : (4,4) world->camera extrinsic. Camera axes: +X=right, +Y=down, +Z=forward.

    Conventions:
      - Horizontal FOV given (hfov_deg); pixel center at ((w-1)/2, (h-1)/2).
      - 'up_world' is only used by the ray generator (Open3D) for disambiguation.
      - Extrinsics are derived from the actual ray field to avoid convention drift.
    """
    cam_centre = np.asarray(cam_centre, dtype=np.float64).reshape(3)
    look_at    = np.asarray(look_at,    dtype=np.float64).reshape(3)

    # --- Intrinsics (square pixels; H-FOV given) ---
    fx = (width_px / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx  # square pixels
    cx = (width_px  - 1) / 2.0
    cy = (height_px - 1) / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # --- Rays (match Open3D raster convention exactly) ---
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=hfov_deg,
        center=look_at.tolist(),
        eye=cam_centre.tolist(),
        up=up_world,
        width_px=width_px,
        height_px=height_px,
    ).numpy().reshape(height_px, width_px, 6)

    dirs = rays[..., 3:]  # (H,W,3)
    cyi, cxi = (height_px - 1) // 2, (width_px - 1) // 2

    # Center ray = +Z (forward)
    f = dirs[cyi, cxi]
    f /= (np.linalg.norm(f) + 1e-12)

    # Image gradients → basis directions
    dx = dirs[cyi, min(cxi+1, width_px-1)] - dirs[cyi, max(cxi-1, 0)]
    dy = dirs[min(cyi+1, height_px-1), cxi] - dirs[max(cyi-1, 0), cxi]  # "down" in image

    r = dx / (np.linalg.norm(dx) + 1e-12)  # +X = right
    d = dy / (np.linalg.norm(dy) + 1e-12)  # +Y = down

    # Orthonormalize and enforce right-handedness with r × d = f
    d = d - np.dot(d, r) * r
    d /= (np.linalg.norm(d) + 1e-12)
    f = np.cross(r, d)
    f /= (np.linalg.norm(f) + 1e-12)

    # Safety: ensure proper rotation (no reflection)
    R_wc = np.stack([r, d, f], axis=1)  # columns: right, down, forward (camera->world)
    if np.linalg.det(R_wc) < 0:
        r = -r
        R_wc = np.stack([r, d, f], axis=1)

    # world->camera
    R_cw = R_wc.T
    t = -R_cw @ cam_centre
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R_cw
    E[:3,  3] = t

    return K, E