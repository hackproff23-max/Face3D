"""Pipeline: 3 fotos -> landmarks 3D (MediaPipe) -> malha fundida -> GLB com skeleton."""
import json
import struct
import threading

import numpy as np
from scipy.spatial import Delaunay

ANCHORS = [33, 133, 362, 263, 1, 61, 291, 168, 152, 10]
_mesh = None
_lock = threading.Lock()


def detect(img_bgr):
    """Retorna (pontos 3D Nx3 com Y para cima, uv Nx2) ou None."""
    global _mesh
    import cv2
    import mediapipe as mp

    with _lock:
        if _mesh is None:
            _mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5)
        h, w = img_bgr.shape[:2]
        res = _mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    raw = np.array([[p.x * w, -p.y * h, -p.z * w] for p in lm], dtype=np.float64)
    uv = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
    return raw, uv


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    E = np.diag([1, 1, d])
    R = U @ E @ Vt
    s = (sig * np.array([1, 1, d])).sum() / ((S ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def _anchors_for(v):
    """Foto de perfil: só usa pontos do meio do rosto + o lado que aparece."""
    mid = [1, 168, 6, 10, 152, 13, 14, 0, 17]
    if v[33][2] > v[263][2]:
        return mid + [33, 133, 61]
    return mid + [263, 362, 291]


def triangulate(P):
    tri = Delaunay(P[:, :2]).simplices.copy()
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = cross < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    e = np.stack([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1),
                  np.linalg.norm(c - a, axis=1)], 1).max(1)
    return tri[e < 3.0 * np.median(e)].astype(np.uint32)


def vertex_normals(P, tri):
    n = np.zeros_like(P)
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    for i in range(3):
        np.add.at(n, tri[:, i], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    return n / l


def fuse(views):
    """views[0] = frente. Alinha as outras e mistura por quão de frente cada ponto está para a câmera."""
    front = views[0]
    n0 = vertex_normals(front, triangulate(front))
    num = front * (np.clip(n0[:, 2], 0, 1) ** 2 + 0.05)[:, None]
    den = np.clip(n0[:, 2], 0, 1) ** 2 + 0.05
    for v in views[1:]:
        s, R, t = umeyama(v[_anchors_for(v)], front[_anchors_for(v)])
        aligned = s * (v @ R.T) + t
        w = np.clip(n0 @ (R @ np.array([0, 0, 1.0])), 0, 1) ** 2
        num += aligned * w[:, None]
        den = den + w
    P = num / den[:, None]
    P *= 0.22 / np.linalg.norm(P[10] - P[152])  # altura do rosto ~22 cm
    P -= P.mean(0)
    return P


def build_rig(P):
    ear = (P[234] + P[454]) / 2
    chin = P[152]
    neck = np.array([ear[0], chin[1] - 0.05, ear[2] - 0.03])
    head = np.array([ear[0], ear[1] - 0.03, ear[2] - 0.02])
    jaw = np.array([ear[0], P[13][1] + 0.015, ear[2]])
    root = np.array([ear[0], neck[1] - 0.08, neck[2]])
    return {
        "names": ["root", "neck", "head", "jaw", "eye_R", "eye_L"],
        "pos": np.stack([root, neck, head, jaw, P[468], P[473]]),
        "parent": [-1, 0, 1, 2, 2, 2],
    }


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def skin_weights(P):
    n = len(P)
    y13, y152 = P[13][1], P[152][1]
    t = (y13 - P[:, 1]) / (y13 - y152 + 1e-9)
    wj = _smooth(t / 0.12)
    xm = (P[234][0] + P[454][0]) / 2
    hw = abs(P[234][0] - P[454][0]) / 2 + 1e-9
    wj *= 1 - _smooth((np.abs(P[:, 0] - xm) / hw - 0.65) / 0.35)
    we = np.zeros(n)
    eye_id = np.zeros(n, dtype=np.uint8)
    we[468:473], eye_id[468:473] = 1, 4
    we[473:478], eye_id[473:478] = 1, 5
    wj = wj * (1 - we)
    wh = np.clip(1 - wj - we, 0, 1)
    J = np.zeros((n, 4), dtype=np.uint8)
    W = np.zeros((n, 4), dtype=np.float32)
    J[:, 0], J[:, 1], J[:, 2] = 2, 3, np.where(eye_id > 0, eye_id, 0)
    W[:, 0], W[:, 1], W[:, 2] = wh, wj, we
    W /= W.sum(1, keepdims=True)
    return J, W


def build_glb(P, uv, tri, jpeg, rig):
    n, names, pos, parent = len(P), rig["names"], rig["pos"], rig["parent"]
    normals = vertex_normals(P, tri)
    J, W = skin_weights(P)
    blob, views, accs = bytearray(), [], []

    def view(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        blob.extend(data)
        views.append(v)
        return len(views) - 1

    def acc(data, ctype, count, atype, target=None, mm=None):
        a = {"bufferView": view(data, target), "componentType": ctype, "count": count, "type": atype}
        if mm:
            a["min"], a["max"] = mm
        accs.append(a)
        return len(accs) - 1

    f4 = lambda x: np.ascontiguousarray(x, dtype="<f4").tobytes()
    a_pos = acc(f4(P), 5126, n, "VEC3", 34962, (P.min(0).tolist(), P.max(0).tolist()))
    a_nrm = acc(f4(normals), 5126, n, "VEC3", 34962)
    a_uv = acc(f4(uv), 5126, n, "VEC2", 34962)
    a_j = acc(J.tobytes(), 5121, n, "VEC4", 34962)
    a_w = acc(f4(W), 5126, n, "VEC4", 34962)
    a_i = acc(np.ascontiguousarray(tri, dtype="<u4").tobytes(), 5125, int(tri.size), "SCALAR", 34963)
    ibm = np.zeros((len(names), 16), dtype="<f4")
    for k, p in enumerate(pos):
        ibm[k] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -p[0], -p[1], -p[2], 1]
    a_ibm = acc(ibm.tobytes(), 5126, len(names), "MAT4")
    img_view = view(jpeg)

    nodes = [{"name": "FaceMesh", "mesh": 0, "skin": 0}]
    for k, name in enumerate(names):
        local = pos[k] - (pos[parent[k]] if parent[k] >= 0 else 0)
        nd = {"name": name, "translation": [float(x) for x in local]}
        kids = [1 + c for c in range(len(names)) if parent[c] == k]
        if kids:
            nd["children"] = kids
        nodes.append(nd)

    gltf = {
        "asset": {"version": "2.0", "generator": "Face3D"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [{"name": "Face", "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                           "JOINTS_0": a_j, "WEIGHTS_0": a_w},
            "indices": a_i, "material": 0}]}],
        "skins": [{"joints": list(range(1, 1 + len(names))), "skeleton": 1, "inverseBindMatrices": a_ibm}],
        "materials": [{"name": "Skin", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.85}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": img_view, "mimeType": "image/jpeg"}],
        "accessors": accs, "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<III", 0x46546C67, 2, total)
            + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def generate(front_img, left_img, right_img, front_jpeg):
    dets = [detect(i) for i in (front_img, left_img, right_img)]
    if dets[0] is None:
        raise ValueError("Não encontrei rosto na foto da frente.")
    ok = [d for d in dets if d is not None]
    P = fuse([d[0] for d in ok])
    tri = triangulate(P)
    return build_glb(P, dets[0][1], tri, front_jpeg, build_rig(P)), len(ok)
