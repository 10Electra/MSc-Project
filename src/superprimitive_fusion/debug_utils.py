import copy
from collections import defaultdict
import numpy as np
import open3d as o3d

# ---------------------------
# Basic utilities
# ---------------------------

def _as_arrays(o3d_mesh):
    V = np.asarray(o3d_mesh.vertices)
    F = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    return V, F

def _face_subset_mesh(V, F, face_idx):
    """Create a compact mesh consisting only of faces[face_idx]."""
    face_idx = np.asarray(face_idx, dtype=np.int64)
    if face_idx.size == 0:
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        m.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
        return m
    subF = F[face_idx]
    vids = np.unique(subF.reshape(-1))
    vid_map = -np.ones(len(F.reshape(-1)) if F.size else 0, dtype=np.int64)  # not used, kept for clarity

    vid_map = -np.ones(int(np.max(vids)) + 1, dtype=np.int64)
    vid_map[vids] = np.arange(vids.size, dtype=np.int64)

    subV = V[vids]
    subF = vid_map[subF]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(subV)
    m.triangles = o3d.utility.Vector3iVector(subF.astype(np.int32))
    m.compute_vertex_normals()
    return m

def _edge_index(F):
    """Return edge->faces dict for sorted vertex-pair edges."""
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    face_ids = np.repeat(np.arange(F.shape[0], dtype=np.int64), 3)
    edict = defaultdict(list)
    for (i, j), fid in zip(map(tuple, edges), face_ids):
        edict[(i, j)].append(fid)
    return edict

def _edge_lineset(V, edict, color=(1.0, 0.8, 0.0)):
    """Build a LineSet from an edge dict (keys are (i,j) vertex pairs)."""
    if not edict:
        return o3d.geometry.LineSet()
    pts = []
    segs = []
    for (i, j) in edict.keys():
        segs.append([len(pts), len(pts) + 1])
        pts.append(V[i])
        pts.append(V[j])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(segs, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(segs), 1)))
    return ls

# ---------------------------
# Bow-tie (vertex non-manifold) detection
# ---------------------------

def _vertex_face_incidence(F, nV):
    inc = [[] for _ in range(nV)]
    for f_id, (a, b, c) in enumerate(F):
        inc[a].append(f_id); inc[b].append(f_id); inc[c].append(f_id)
    return inc

def _face_adj_from_F(F):
    """Face adjacency by shared edge (unordered). Returns list of sets: f -> {adjacent f}."""
    edict = _edge_index(F)
    adj = [set() for _ in range(F.shape[0])]
    for fids in edict.values():
        if len(fids) >= 2:
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    a, b = fids[i], fids[j]
                    adj[a].add(b); adj[b].add(a)
    return adj

import numpy as np

import numpy as np
from collections import defaultdict

def _bow_tie_vertices(V, F):
    """
    Return indices of vertices whose incident faces split into >1 connected fan
    (vertex non-manifold 'bow-tie').

    Implementation notes:
      - No dependency on external face adjacency.
      - For each vertex v, connect incident faces that share an edge (v, u).
      - Works for triangles and polygons; ignores degenerate (u == v) edges.
    """
    V = np.asarray(V)
    F = np.asarray(F)
    nV = V.shape[0]

    # --- Build vertex -> incident face indices (no aliasing!) ---
    inc = [[] for _ in range(nV)]
    for fi, fa in enumerate(F):
        # Ensure iterable row
        fa = list(fa)
        for v in set(fa):  # set() tolerates duplicate verts inside a face
            if 0 <= v < nV:
                inc[v].append(fi)

    bow = []
    for v in range(nV):
        faces = inc[v]
        if len(faces) <= 1:
            continue

        # Map each edge that includes v to its incident faces
        edge2faces = defaultdict(list)  # key: (min(a,b), max(a,b))
        for fi in faces:
            fa = list(F[fi])
            k = len(fa)
            if k < 2:
                continue
            for i in range(k):
                a, b = fa[i], fa[(i + 1) % k]
                if a == b:
                    continue
                if a == v or b == v:
                    e = (a, b) if a < b else (b, a)
                    edge2faces[e].append(fi)

        # Build local adjacency among 'faces' via these edges
        adj_local = {fi: set() for fi in faces}
        for flist in edge2faces.values():
            # All faces sharing the same edge (v, u) are mutually adjacent
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    a, b = flist[i], flist[j]
                    adj_local[a].add(b)
                    adj_local[b].add(a)

        # Count connected components among incident faces
        seen = set()
        comps = 0
        for f in faces:
            if f in seen:
                continue
            comps += 1
            if comps > 1:
                bow.append(v)
                break
            stack = [f]
            seen.add(f)
            while stack:
                cur = stack.pop()
                for nb in adj_local[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

    return np.asarray(bow, dtype=np.int64)

# ---------------------------
# Triangle-triangle intersection (fallback SAT)
# ---------------------------

def _tri_tri_intersect(a, b, c, d, e, f):
    """
    Conservative SAT-like test. Treats coplanar overlap as intersection.
    Skips exact shared edges/vertices in caller.
    """
    EPS = 1e-12
    T1 = np.array([a, b, c], dtype=float)
    T2 = np.array([d, e, f], dtype=float)

    def norm(p):
        return np.cross(p[1] - p[0], p[2] - p[0])

    N1 = norm(T1); N2 = norm(T2)
    # Plane-side check
    s2 = (T2 - T1[0]) @ N1
    if (np.max(s2) < -EPS) or (np.min(s2) > EPS):
        return False
    s1 = (T1 - T2[0]) @ N2
    if (np.max(s1) < -EPS) or (np.min(s1) > EPS):
        return False
    # Edge-edge axes
    E1 = [T1[1] - T1[0], T1[2] - T1[1], T1[0] - T1[2]]
    E2 = [T2[1] - T2[0], T2[2] - T2[1], T2[0] - T2[2]]
    for u in E1:
        for v in E2:
            ax = np.cross(u, v)
            ln = np.linalg.norm(ax)
            if ln < EPS:
                continue
            ax /= ln
            p1 = T1 @ ax; p2 = T2 @ ax
            if (np.max(p1) < np.min(p2) - EPS) or (np.max(p2) < np.min(p1) - EPS):
                return False
    return True

# ---------------------------
# Self-intersection detection
# ---------------------------

def _triangle_aabbs(V, F):
    tri = V[F]  # (M,3,3)
    lo = tri.min(axis=1); hi = tri.max(axis=1)
    return lo, hi

def _candidate_pairs_grid(V, F, lo, hi, max_bucket=64):
    """
    Uniform grid broad-phase for non-adjacent triangle pairs.
    """
    M = F.shape[0]
    bbox_min = lo.min(axis=0); bbox_max = hi.max(axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-9)
    res = int(np.clip(np.ceil(M ** (1.0 / 3.0)), 8, max_bucket))
    res = np.array([res, res, res], dtype=int)
    cell_size = span / res

    def cells_for_tri(i):
        cl = np.floor((lo[i] - bbox_min) / cell_size).astype(int)
        ch = np.floor((hi[i] - bbox_min) / cell_size).astype(int)
        cl = np.maximum(cl, 0); ch = np.minimum(ch, res - 1)
        cells = []
        for x in range(cl[0], ch[0] + 1):
            for y in range(cl[1], ch[1] + 1):
                for z in range(cl[2], ch[2] + 1):
                    cells.append((x, y, z))
        return cells

    from collections import defaultdict
    grid = defaultdict(list)
    for i in range(M):
        for cell in cells_for_tri(i):
            grid[cell].append(i)

    # Build candidates
    vsets = [set(f) for f in F]
    cand = set()
    for bucket in grid.values():
        if len(bucket) < 2:
            continue
        b = bucket
        for i in range(len(b)):
            fi = b[i]
            for j in range(i + 1, len(b)):
                fj = b[j]
                # skip adjacent (share vertex)
                if vsets[fi].intersection(vsets[fj]):
                    continue
                if fi < fj:
                    cand.add((fi, fj))
                else:
                    cand.add((fj, fi))
    return cand

def _self_intersecting_faces(V, F, prefer_igl=True, max_pairs=None):
    """
    Return set of face indices that participate in self-intersections.
    Tries Open3D fast path, then libigl; falls back to grid + SAT.
    """
    # Open3D (if available in your build)
    try:
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(F.astype(np.int32))
        )
        if hasattr(o3d_mesh, "get_self_intersecting_triangles"):
            idx = o3d_mesh.get_self_intersecting_triangles()
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            return set(idx.tolist())
    except Exception:
        pass

    # libigl path
    if prefer_igl:
        try:
            import igl
            Vr, Fr, IF, J, IM = igl.remesh_self_intersections(V, F)
            IF = np.asarray(IF)
            J = np.asarray(J).reshape(-1)
            if IF.size > 0:
                # IF: intersecting face pairs in remeshed mesh; map back using J
                involved = np.unique(IF.reshape(-1))
                orig_faces = np.unique(J[Fr[involved]])
                return set(orig_faces.astype(int).tolist())
            return set()
        except Exception:
            pass

    # Fallback: grid + tri-tri
    lo, hi = _triangle_aabbs(V, F)
    cand = _candidate_pairs_grid(V, F, lo, hi)
    if (max_pairs is not None) and (len(cand) > max_pairs):
        cand = set(list(cand)[:max_pairs])
    bad_faces = set()
    for i, j in cand:
        if np.any(hi[i] < lo[j]) or np.any(hi[j] < lo[i]):
            continue
        a, b, c = V[F[i, 0]], V[F[i, 1]], V[F[i, 2]]
        d, e, f = V[F[j, 0]], V[F[j, 1]], V[F[j, 2]]
        if _tri_tri_intersect(a, b, c, d, e, f):
            bad_faces.add(i); bad_faces.add(j)
    return bad_faces

# ---------------------------
# Visualization helpers
# ---------------------------

def spheres_at_vertices(V, F, idx, radius=None, color=(0.0, 0.6, 1.0)):
    """Create small spheres at the specified vertex indices."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    if radius is None:
        # median triangle edge length as scale proxy
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        elen = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        med = np.median(elen[np.isfinite(elen)])
        radius = 0.25 * med if (np.isfinite(med) and med > 0) else 1e-3

    spheres = []
    for v in idx:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
        s.translate(V[int(v)])
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        spheres.append(s)
    return spheres

def _paint_uniform(mesh, color=(0.82, 0.82, 0.82)):
    m = copy.deepcopy(mesh)
    m.paint_uniform_color(color)
    return m

# ---------------------------
# Main debug entry point
# ---------------------------

def debug_mesh(mesh: o3d.geometry.TriangleMesh,
               prefer_igl: bool = True,
               show: bool = True,
               max_pairs: int = 200000):
    """
    Analyze 'mesh' and visualize offending primitives:
      - Self-intersecting faces (RED)
      - Bow-tie vertices (CYAN spheres)
      - Non-manifold edges (>2 faces) (YELLOW lines)

    Returns a dict with indices for programmatic use.
    """
    V, F = _as_arrays(mesh)

    # Health summary (Open3D)
    e_man = mesh.is_edge_manifold(allow_boundary_edges=True)
    v_man = mesh.is_vertex_manifold()
    self_x = mesh.is_self_intersecting()
    water  = mesh.is_watertight()

    # Offending primitives
    bow = _bow_tie_vertices(V, F)
    edict = _edge_index(F)
    nme = {e: fids for e, fids in edict.items() if len(fids) > 2}
    bad_faces = _self_intersecting_faces(V, F, prefer_igl=prefer_igl, max_pairs=max_pairs)

    # Optional Trimesh summary (if available)
    tm_info = {}
    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
        tm_info = dict(
            winding_consistent=bool(tm.is_winding_consistent),
            watertight=bool(tm.is_watertight),
            euler_number=int(tm.euler_number)
        )
    except Exception:
        pass

    # Print summary
    print("=== Mesh Health ===")
    print(f"[o3d] edge manifold: {e_man}")
    print(f"[o3d] vertex manifold: {v_man}")
    print(f"[o3d] self-intersecting: {self_x}")
    print(f"[o3d] watertight: {water}")
    if tm_info:
        print(f"[tm]  winding consistent: {tm_info['winding_consistent']}")
        print(f"[tm]  watertight: {tm_info['watertight']}")
        print(f"[tm]  euler number: {tm_info['euler_number']}")
    print(f"bow-tie vertices: {len(bow)}")
    print(f"non-manifold edges (>2 incident faces): {len(nme)}")
    print(f"self-intersecting faces: {len(bad_faces)}")

    # Build visualization
    geoms = []
    base = _paint_uniform(mesh, (0.82, 0.82, 0.82))
    base.compute_vertex_normals()
    geoms.append(base)

    if len(bad_faces) > 0:
        sub = _face_subset_mesh(V, F, sorted(bad_faces))
        sub.paint_uniform_color((1.0, 0.0, 0.0))
        geoms.append(sub)

    if len(bow) > 0:
        geoms.extend(_spheres_at_vertices(V, F, bow, color=(0.0, 0.7, 1.0)))

    if len(nme) > 0:
        geoms.append(_edge_lineset(V, nme, color=(1.0, 0.85, 0.2)))

    if show:
        o3d.visualization.draw_geometries(geoms)

    return {
        "bow_tie_vertices": np.asarray(bow, dtype=np.int64),
        "non_manifold_edges": list(nme.keys()),
        "self_intersecting_faces": np.asarray(sorted(list(bad_faces)), dtype=np.int64),
        "trimesh_info": tm_info
    }
