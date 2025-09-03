import os
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d # type: ignore
import pymeshlab as ml # type: ignore
import trimesh


def tsdf_fuse(
        depth_images, rgb_images, K, E_wc_list,
        voxel_length=1.0/1024.0, trunc_voxels=4, depth_trunc=1.0
    ):
    """Extrinsics MUST be world->camera (E_wc)."""
    assert len(depth_images)==len(rgb_images)==len(E_wc_list)
    h, w = depth_images[0].shape[:2]
    K = np.asarray(K, float)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
    sdf_trunc = trunc_voxels * voxel_length

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for depth, rgb, E_wc in zip(depth_images, rgb_images, E_wc_list):
        
        d = np.asarray(depth)
        d[~np.isfinite(d)] = 0
        dimg = o3d.geometry.Image(d.astype(np.float32))
        
        arr = np.asarray(rgb)
        if arr.dtype.kind in "fc":
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        cimg = o3d.geometry.Image(np.ascontiguousarray(arr))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            cimg, dimg, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        vol.integrate(rgbd, intr, np.asarray(E_wc, dtype=np.float64))

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def tsdf_fuse_from_depth_data(depth_data, voxel_length=1./1024, trunc_voxels=5):
    depth_images = [d['depth'].copy() for d in depth_data]
    for d in depth_images:
        d[~np.isfinite(d)] = 0
    rgb_images   = [d['rgb'].copy() for d in depth_data]
    K            = depth_data[0]['K_t']

    T_wc = [d['E'] for d in depth_data]

    mesh = tsdf_fuse(
        depth_images, rgb_images, K, T_wc,
        voxel_length=voxel_length,
        trunc_voxels=trunc_voxels,
    )
    return mesh




def _ensure_path(obj, suffix=".obj"):
    if isinstance(obj, (str, Path)):
        return str(obj), None
    if isinstance(obj, o3d.geometry.TriangleMesh):
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False); tmp.close()
        ok = o3d.io.write_triangle_mesh(tmp.name, obj, write_ascii=False, compressed=False)
        if not ok: raise RuntimeError("Open3D write failed.")
        return tmp.name, tmp.name
    raise TypeError("gt/recon must be a filepath or open3d TriangleMesh")

def _cleanup_tmp(p):
    if p and os.path.isfile(p):
        try: os.remove(p)
        except OSError: pass

def _list_ids(ms):
    try: return list(ms.mesh_id_list())
    except Exception: return [i for i in range(ms.number_meshes())]

def _dist_array_from_mesh(ms, mesh_id):
    m = ms.mesh(mesh_id)
    if hasattr(m, "vertex_quality_array"):
        q = np.asarray(m.vertex_quality_array()).ravel()
        if q.size and np.any(np.isfinite(q)): return q
    if hasattr(m, "vertex_scalar_array"):
        s = np.asarray(m.vertex_scalar_array()).ravel()
        if s.size and np.any(np.isfinite(s)): return s
    raise RuntimeError("No per-vertex quality/scalar on mesh id=%s." % mesh_id)

def _summ(d):
    d = np.asarray(d); d = d[np.isfinite(d)]
    return dict(
        mean=float(d.mean()),
        median=float(np.median(d)),
        rms=float(np.sqrt((d**2).mean())),
        p95=float(np.quantile(d, 0.95)),
        p99=float(np.quantile(d, 0.99)),
        hausdorff=float(d.max()),
        trimmed_hausdorff_99=float(np.quantile(d, 0.99)),
        count=int(d.size),
    )

def _fscore(d_re2gt, d_gt2re, taus_abs):
    out = {}
    assert isinstance(d_re2gt, np.ndarray) and isinstance(d_gt2re, np.ndarray)
    for t in taus_abs:
        P = float((d_re2gt <= t).mean()) if d_re2gt.size else 0.0   # how much of recon is within t of GT
        R = float((d_gt2re <= t).mean()) if d_gt2re.size else 0.0   # how much of GT is within t of recon
        F = 0.0 if (P+R)==0 else 2*P*R/(P+R)
        out[float(t)] = {"tau_m": float(t), "precision": P, "recall": R, "fscore": F}
    return out

rng = np.random.default_rng(42)  # reproducible

def _load_trimesh(path):
    # process=False preserves original vertices/faces (no repair/merge)
    return trimesh.load_mesh(path, process=False)

def _sample_on_faces(mesh_tm: trimesh.Trimesh, n_samples: int):
    """
    Importance-sample points by triangle area.
    Returns: (points Nx3, face_idx Nx1)
    """
    areas = mesh_tm.area_faces
    probs = areas / areas.sum()
    face_idx = rng.choice(len(areas), size=n_samples, p=probs)

    tri = mesh_tm.triangles[face_idx]  # (N, 3, 3)
    # Uniform barycentric
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v
    pts = (tri[:, 0] * w[:, None] +
           tri[:, 1] * u[:, None] +
           tri[:, 2] * v[:, None])  # (N,3)
    return pts, face_idx

def _nearest_faces(target_tm: trimesh.Trimesh, query_pts: np.ndarray):
    """
    For each query point, get nearest point ON the target surface, distance, and face index.
    Uses trimesh's proximity queries (BVH-backed).
    """
    # ProximityQuery is faster on large meshes
    pq = trimesh.proximity.ProximityQuery(target_tm)
    closest_pts, dists, face_ids = pq.on_surface(query_pts)  # (N,3), (N,), (N,)
    return closest_pts, dists, face_ids

def _face_normals(mesh_tm: trimesh.Trimesh):
    # Trimesh keeps face_normals unit-length; guard just in case
    N = np.asarray(mesh_tm.face_normals, dtype=np.float64)
    nb = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return N / nb

def _angles_from_faces(src_tm, tgt_tm, n_samples, max_pair_dist=None):
    """
    Sample on src triangles; pair to nearest tgt triangles; return angular errors (deg).
    """
    P_src, F_src = _sample_on_faces(src_tm, n_samples)
    N_src_faces = _face_normals(src_tm)
    n_src = N_src_faces[F_src]  # (N,3)

    _, dists, tgt_face = _nearest_faces(tgt_tm, P_src)
    N_tgt_faces = _face_normals(tgt_tm)
    n_tgt = N_tgt_faces[tgt_face]  # (N,3)

    # Optionally gate by distance (metres) to avoid nonsense pairings across gaps
    if max_pair_dist is not None:
        mask = dists <= float(max_pair_dist)
        n_src = n_src[mask]; n_tgt = n_tgt[mask]

    # Sign-agnostic angular error
    dots = np.einsum('ij,ij->i', n_src, n_tgt)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return ang

def _summ_angle(deg):
    deg = np.asarray(deg); deg = deg[np.isfinite(deg)]
    if deg.size == 0:
        return dict(mean=0.0, median=0.0, p90=0.0, p95=0.0, max=0.0, count=0)
    return dict(
        mean=float(deg.mean()),
        median=float(np.median(deg)),
        p90=float(np.quantile(deg, 0.90)),
        p95=float(np.quantile(deg, 0.95)),
        max=float(deg.max()),
        count=int(deg.size),
    )


def evaluate_mesh_pair(
    gt, recon,
    samples_per_mesh=200_000,
    taus_abs=(2.5e-4, 5e-4, 1e-3, 2e-3),# 0.25, 0.5, 1, 2 mm
    include_topology=True,
    return_distance_arrays=False,
    debug=False,
    include_normal_error=True,
    normal_from='triangle',             # 'triangle' or 'pca'
    normal_gate_mult=2.0,
):
    gt_path, tmp_gt = _ensure_path(gt, suffix=".obj")
    re_path, tmp_re = _ensure_path(recon, suffix=".obj")
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(gt_path);    gt_id    = ms.current_mesh_id()
        ms.load_new_mesh(re_path);    recon_id = ms.current_mesh_id()

        # recon -> GT  (savesample=True and capture the new layer id)
        before = set(_list_ids(ms))
        ms.set_current_mesh(recon_id)
        ms.apply_filter('get_hausdorff_distance',
                        targetmesh=gt_id,
                        samplevert=False, sampleedge=False, sampleface=True,
                        samplenum=samples_per_mesh, savesample=True)
        rg_sample_id = sorted(list(set(_list_ids(ms)) - before))[-1]
        d_re2gt = _dist_array_from_mesh(ms, rg_sample_id)

        # GT -> recon
        before = set(_list_ids(ms))
        ms.set_current_mesh(gt_id)
        ms.apply_filter('get_hausdorff_distance',
                        targetmesh=recon_id,
                        samplevert=False, sampleedge=False, sampleface=True,
                        samplenum=samples_per_mesh, savesample=True)
        gr_sample_id = sorted(list(set(_list_ids(ms)) - before))[-1]
        d_gt2re = _dist_array_from_mesh(ms, gr_sample_id)

        if debug:
            print("re->gt min/med/max (m):", float(d_re2gt.min()), float(np.median(d_re2gt)), float(d_re2gt.max()))
            print("gt->re min/med/max (m):", float(d_gt2re.min()), float(np.median(d_gt2re)), float(d_gt2re.max()))
            print("sample sizes:", d_re2gt.size, d_gt2re.size)

        stats = {
            "recon_to_gt": _summ(d_re2gt),
            "gt_to_recon": _summ(d_gt2re),
            "fscore": _fscore(d_re2gt, d_gt2re, taus_abs=taus_abs),
        }


        if include_normal_error and normal_from == 'triangle':
            # Load as trimesh
            tm_gt = _load_trimesh(gt_path)
            tm_re = _load_trimesh(re_path)

            tau_max = max(taus_abs) if len(taus_abs) else None
            gate = (normal_gate_mult * tau_max) if tau_max is not None else None

            ang_re2gt = _angles_from_faces(tm_re, tm_gt, samples_per_mesh, max_pair_dist=gate)
            ang_gt2re = _angles_from_faces(tm_gt, tm_re, samples_per_mesh, max_pair_dist=gate)

            stats["normal_error_deg_triangle"] = {
                "recon_to_gt": _summ_angle(ang_re2gt),
                "gt_to_recon": _summ_angle(ang_gt2re),
                "symmetric": _summ_angle(np.concatenate([ang_re2gt, ang_gt2re])),
            }
            if return_distance_arrays:
                stats["_normal_angles_deg_triangle"] = {
                    "recon_to_gt": ang_re2gt,
                    "gt_to_recon": ang_gt2re,
                }


        if include_topology:
            ms.set_current_mesh(recon_id)
            stats["recon_topology"] = ms.apply_filter('get_topological_measures')
            ms.set_current_mesh(gt_id)
            stats["gt_topology"] = ms.apply_filter('get_topological_measures')

        if return_distance_arrays:
            stats["_distances"] = {"recon_to_gt": d_re2gt, "gt_to_recon": d_gt2re}

        return stats
    finally:
        _cleanup_tmp(tmp_gt); _cleanup_tmp(tmp_re)


def mse_and_iou(depth_data_gt, depth_data_recon):
    """
    depth_data_*: iterable of dicts with keys:
        - 'rgb':   HxWx3 (float [0..1])
        - 'depth': HxW float (0 or NaN means invalid)
    Returns one MSE and IoU per view, plus simple means over views.
    """
    per_mse, per_iou = [], []

    for gt, re in zip(depth_data_gt, depth_data_recon):
        rgb_gt = np.asarray(gt['rgb'])[..., :3]
        rgb_re = np.asarray(re['rgb'])[..., :3]
        dep_gt = np.asarray(gt['depth'])
        dep_re = np.asarray(re['depth'])

        # shape checks
        if rgb_gt.shape[:2] != rgb_re.shape[:2] or dep_gt.shape != dep_re.shape \
           or rgb_gt.shape[:2] != dep_gt.shape:
            raise ValueError("RGB/Depth shapes must match per pair.")

        # valid-depth masks (silhouettes)
        m_gt = np.isfinite(dep_gt) & (dep_gt > 0)
        m_re = np.isfinite(dep_re) & (dep_re > 0)

        # IoU of silhouettes
        inter = int(np.logical_and(m_gt, m_re).sum())
        union = int(np.logical_or(m_gt, m_re).sum())
        iou = (inter / union) if union > 0 else np.nan
        per_iou.append(iou)

        # MSE on overlap of valid pixels only
        both = m_gt & m_re
        if np.any(both):
            diff2 = (rgb_gt - rgb_re) ** 2
            mse = float(diff2[both].mean())
        else:
            mse = np.nan

        per_mse.append(mse)

    overall_mse = float(np.nanmean(per_mse)) if per_mse else float('nan')
    overall_iou = float(np.nanmean(per_iou)) if per_iou else float('nan')

    return {
        "per_view": [{"mse": float(m), "iou": float(i)} for m, i in zip(per_mse, per_iou)],
        "overall": {"mse": overall_mse, "iou": overall_iou},
    }
