import math
import numpy as np
import open3d as o3d  # type: ignore
from typing import Tuple, Dict, List

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
        meshlist: list[o3d.geometry.TriangleMesh],
        cam_centre: tuple,
        look_at:   tuple,
        width_px:   int = 360,
        height_px:  int = 240,
        fov:        float = 70.0,
    ) -> dict:

    scene = o3d.t.geometry.RaycastingScene()

    geom_id_list = []
    for mesh in meshlist:
        id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        geom_id_list.append(id)
    geom_ids = np.asarray(geom_id_list)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov,
        center=list(look_at),
        eye=list(cam_centre),
        up=[0, 0, 1],
        width_px=width_px,
        height_px=height_px,
    )

    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()

    # Intersection metadata (gometry id + triangle id + barycentric uv + normals)
    geom_ids = ans["geometry_ids"].numpy().astype(np.int16)
    geom_ids[geom_ids > 1e2] = -1
    prim_ids = ans["primitive_ids"].numpy().astype(np.int32)
    bary_uv = ans.get("primitive_uvs", None).numpy()
    normals = ans["primitive_normals"].numpy()
        
    rays_np = rays.numpy()  # (H*W,6)
    origins = rays_np[..., :3]
    dirs = rays_np[..., 3:]
    verts = origins + dirs * t_hit[..., None]
    
    vcols = np.full((*t_hit.shape, 3), 0.5)

    # Assertions for bugfixing
    assert bary_uv is not None
    for mesh in meshlist:
        assert mesh.has_vertex_colors()

    for id in range(len(meshlist)):
        rel_prim_ids = prim_ids[geom_ids==id]
        rel_bary_uv  =  bary_uv[geom_ids==id]
        vcols[geom_ids==id] = _interpolate_vertex_colors(meshlist[id], rel_prim_ids, rel_bary_uv)
    
    scan = dict()
    scan['verts'] = verts
    scan['vcols'] = vcols
    scan['norms'] = normals
    scan['segmt'] = geom_ids
    
    return scan

def triangulate_rgbd_grid(
    verts:  np.ndarray,
    valid:  np.ndarray,
    z:      np.ndarray,
    obj_id: np.ndarray | None = None,
    k:      float = 3.5,        
    normals: np.ndarray | None = None,
    max_normal_angle_deg: float | None = None
    ) -> np.ndarray:
    """
    verts: (H*W,3) or (H,W,3)
    valid: (H*W,) or (H,W) bool
    z    : (H*W,) or (H,W) depth
    obj_id: (H*W,) or (H,W) int; unknown == -1
    normals: optional (H,W,3); if set, edges with normal jump > max_normal_angle_deg are cut
    """
    # reshape
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

    # disparity (robust jumps)
    disp = np.zeros_like(z_img, dtype=np.float32)
    m = valid_img & np.isfinite(z_img) & (z_img > 0)
    disp[m] = 1.0 / z_img[m]

    # object ids
    if obj_id is None:
        obj_id = np.full((H, W), -1, dtype=np.int32)
    else:
        obj_id = obj_id.reshape(H, W)

    # neighbor diffs
    dx = np.abs(disp[:, 1:] - disp[:, :-1])
    dy = np.abs(disp[1:, :] - disp[:-1, :])
    dd1 = np.abs(disp[:-1, :-1] - disp[1:, 1:])   # tl-br
    dd2 = np.abs(disp[:-1, 1:] - disp[1:, :-1])   # tr-bl

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

    # same-object gates (unknown == -1 → cut)
    same_x  = (obj_id[:, 1:] == obj_id[:, :-1]) & (obj_id[:, 1:] >= 0)
    same_y  = (obj_id[1:, :] == obj_id[:-1, :]) & (obj_id[1:, :] >= 0)
    same_d1 = (obj_id[:-1, :-1] == obj_id[1:, 1:]) & (obj_id[:-1, :-1] >= 0)
    same_d2 = (obj_id[:-1, 1:]  == obj_id[1:, :-1]) & (obj_id[:-1, 1:]  >= 0)

    # base edge validity: depth, object, disparity
    good_x  = mask_x  & same_x  & (dx  <= thr_x)
    good_y  = mask_y  & same_y  & (dy  <= thr_y)
    good_d1 = mask_d1 & same_d1 & (dd1 <= thr_d)
    good_d2 = mask_d2 & same_d2 & (dd2 <= thr_d)

    # optional: normal-angle gate
    if (normals is not None) and (max_normal_angle_deg is not None):
        n = normals.reshape(H, W, 3).astype(np.float32)
        # normalize (cheap and safer; Open3D normals are ~unit but don’t rely on it)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
        ok = n_norm[..., 0] > 0
        n = np.where(ok[..., None], n / np.clip(n_norm, 1e-12, None), 0)

        cos_max = np.cos(np.deg2rad(max_normal_angle_deg))

        dot_x  = (n[:, 1:, :] * n[:, :-1, :]).sum(-1)
        dot_y  = (n[1:, :, :] * n[:-1, :, :]).sum(-1)
        dot_d1 = (n[:-1, :-1, :] * n[1:, 1:, :]).sum(-1)
        dot_d2 = (n[:-1, 1:, :]  * n[1:, :-1, :]).sum(-1)

        # require normals to be finite on both sides
        n_ok_x  = ok[:, 1:]  & ok[:, :-1]
        n_ok_y  = ok[1:, :]  & ok[:-1, :]
        n_ok_d1 = ok[:-1, :-1] & ok[1:, 1:]
        n_ok_d2 = ok[:-1, 1:]  & ok[1:, :-1]

        good_x  &= n_ok_x  & (dot_x  >= cos_max)
        good_y  &= n_ok_y  & (dot_y  >= cos_max)
        good_d1 &= n_ok_d1 & (dot_d1 >= cos_max)
        good_d2 &= n_ok_d2 & (dot_d2 >= cos_max)

    # indices per-quad
    id_img = np.arange(H * W, dtype=np.int32).reshape(H, W)
    tl = id_img[:-1, :-1]; tr = id_img[:-1, 1:]
    bl = id_img[1:, :-1];  br = id_img[1:, 1:]

    # per-quad edge masks
    e_top    = good_x[:H-1, :W-1]
    e_bottom = good_x[1:,  :W-1]
    e_left   = good_y[:H-1, :W-1]
    e_right  = good_y[:H-1, 1:]
    d1 = good_d1
    d2 = good_d2

    # triangles for each diagonal
    A = e_top & e_right & d1            # (tl,tr,br)
    B = e_left & e_bottom & d1          # (tl,br,bl)
    C = e_top & e_left & d2             # (tl,tr,bl)
    D = e_right & e_bottom & d2         # (tr,br,bl)

    # counts per orientation
    count1 = A.astype(np.uint8) + B.astype(np.uint8)
    count2 = C.astype(np.uint8) + D.astype(np.uint8)

    # tie-break by shorter 3D diagonal; compute only where endpoints are finite (no warnings)
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

    A &= prefer_d1
    B &= prefer_d1
    C &= ~prefer_d1
    D &= ~prefer_d1

    # pack tris
    tris = []
    def add(mask, i0, i1, i2):
        if mask.any():
            tris.append(np.stack([i0[mask], i1[mask], i2[mask]], axis=1))
    add(A, tl, tr, br)
    add(B, tl, br, bl)
    add(C, tl, tr, bl)
    add(D, tr, br, bl)

    if not tris:
        return np.empty((0, 3), dtype=np.int32)
    return np.concatenate(tris, axis=0).astype(np.int32)


def mesh_depth_image(
        points:         np.ndarray,
        vertex_colours: np.ndarray,
        cam_centre:     np.ndarray | tuple,
        segmentation:   np.ndarray | None = None,
        normals:        np.ndarray | None = None,
        k = 3.5,
        max_normal_angle_deg: float | None = None,
    ) -> o3d.geometry.TriangleMesh:
    assert points.ndim == 3
    if vertex_colours.ndim == 3:
        vertex_colours = vertex_colours.reshape(-1, 3)
    if segmentation is not None:
        assert segmentation.ndim == 2
    if normals is not None:
        assert normals.ndim == 3
        
    z = np.linalg.norm((points - cam_centre), axis=2)
        
    valid_img = np.ones_like(segmentation, dtype=bool)
    valid_img[segmentation==-1] = 0
    
    tris = triangulate_rgbd_grid(
        verts=points,
        valid=valid_img,
        z=z,
        obj_id=segmentation,
        k=k,
        normals=normals,
        max_normal_angle_deg=max_normal_angle_deg,
    ).astype(np.int32)
    
    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(points.reshape(-1,3))
    mesh_out.triangles = o3d.utility.Vector3iVector(tris[:, [0,2,1]])

    if vertex_colours is not None:
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(vertex_colours.reshape(-1,3))

    #############################
    # Clean‑up / post‑processing#
    #############################
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()
    mesh_out.remove_duplicated_triangles()
    mesh_out.remove_non_manifold_edges()
    mesh_out.compute_vertex_normals()

    return mesh_out


def virtual_mesh_scan(
    meshlist:               list[o3d.geometry.TriangleMesh],
    cam_centre:             tuple,
    look_at:               tuple,
    k:                      float,
    max_normal_angle_deg:   float|None,
    width_px:               int=360,
    height_px:              int=240,
    fov:                    float=70,
) -> o3d.geometry.TriangleMesh:
    """Easy call to virtual_scan() and mesh_depth_mage()"""
    
    result = virtual_scan(
        meshlist,
        cam_centre,
        look_at,
        width_px=width_px,
        height_px=height_px,
        fov=fov,
    )

    mesh = mesh_depth_image(
        points=result['verts'],
        vertex_colours=result['vcols'],
        cam_centre=cam_centre,
        segmentation=result['segmt'],
        normals=result['norms'],
        k=3.5,
        max_normal_angle_deg=max_normal_angle_deg,
    )
    
    return mesh


def fibonacci_sphere_points(n: int, radius: float) -> List[Vec3]:
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
    return pts

def capture_spherical_scans(
    meshlist:               list[o3d.geometry.TriangleMesh],
    num_views:              int = 6,
    radius:                 float = 0.3,
    look_at:                Vec3 = (0.0, 0.0, 0.0),
    width_px:               int = 180,
    height_px:              int = 120,
    fov:                    float = 70.0,
    k:                      float = 3.5,
    max_normal_angle_deg:   float|None=None,
    sampler:                str = "fibonacci",  # or "latlong"
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
        cam_centres = fibonacci_sphere_points(num_views, radius)
    elif sampler == "latlong":
        cam_centres = []
        for i in range(num_views):
            lat = 60  # fix or change to your needs
            lon = (360/num_views) * i
            cam_centres.append(polar2cartesian(r=radius, lat=lat, long=lon))
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")

    scans = []
    for c in cam_centres:
        mesh = virtual_mesh_scan(
            meshlist=meshlist,
            cam_centre=c,
            look_at=look_at,
            k=k,
            max_normal_angle_deg=max_normal_angle_deg,
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
        record = {
            "mesh": mesh,
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