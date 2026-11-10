"""Ajusta a cabeça-molde (GLB exportado do mod) ao rosto das 3 fotos.

Mantém topologia, UV, pesos, ossos e nomes do molde: só muda a forma (malha + ossos)
e pinta o atlas de textura com a pele das fotos.
"""
import json
import struct

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

import face3d

# landmarks MediaPipe usados no alinhamento
LM_IRIS_R, LM_IRIS_L = 468, 473  # íris do olho direito do rosto (esquerda da imagem) e esquerdo
LM_NOSE, LM_CHIN, LM_NASION, LM_MOUTH = 1, 152, 168, (13, 14)


# ---------------------------------------------------------------- molde
class Template:
    def __init__(self, path):
        d = open(path, "rb").read()
        jl, _ = struct.unpack("<II", d[12:20])
        self.js = json.loads(d[20:20 + jl])
        bl, _ = struct.unpack("<II", d[20 + jl:28 + jl])
        self.bin = bytearray(d[28 + jl:28 + jl + bl])
        j, bv = self.js, self.js["bufferViews"]

        def acc(i):
            a = j["accessors"][i]
            v = bv[a["bufferView"]]
            n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
            dt = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[a["componentType"]]
            off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
            return np.frombuffer(bytes(self.bin), dtype=dt, count=a["count"] * n, offset=off).reshape(a["count"], n), off

        prim = j["meshes"][0]["primitives"][0]
        at = prim["attributes"]
        self.acc_pos, self.acc_nrm, self.acc_ibm = at["POSITION"], at["NORMAL"], j["skins"][0]["inverseBindMatrices"]
        self.P, self.off_pos = acc(self.acc_pos)
        self.N, self.off_nrm = acc(self.acc_nrm)
        self.UV, _ = acc(at["TEXCOORD_0"])
        self.tri = acc(prim["indices"])[0].reshape(-1, 3).astype(np.int64)
        self.P = self.P.astype(np.float64)
        self.UV = self.UV.astype(np.float64)
        _, self.off_ibm = acc(self.acc_ibm)
        self.img_view = j["images"][0]["bufferView"]
        iv = bv[self.img_view]
        self.img_off, self.img_len = iv.get("byteOffset", 0), iv["byteLength"]
        if self.img_off + self.img_len < len(self.bin) - 4 or self.img_view != len(bv) - 1:
            raise ValueError("Molde com layout inesperado (textura não é o último bloco).")
        png = bytes(self.bin[self.img_off:self.img_off + self.img_len])
        self.atlas = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)  # BGRA
        if self.atlas is None or self.atlas.ndim != 3 or self.atlas.shape[2] != 4:
            self.atlas = cv2.cvtColor(self.atlas, cv2.COLOR_BGR2BGRA)
        # grupos por região do atlas
        u, v = self.UV[:, 0], self.UV[:, 1]
        self.skin = (u < 0.5) & (v < 0.5)
        self.eye = (u >= 0.5) & (v < 0.5)
        # nós / ossos
        self.nodes = j["nodes"]
        self.joints = j["skins"][0]["joints"]
        self.parent = {}
        for i, nd in enumerate(self.nodes):
            for c in nd.get("children", []):
                self.parent[c] = i
        self.node_world = self._world()
        self.names = {nd.get("name"): i for i, nd in enumerate(self.nodes)}

    def _world(self):
        w = {}

        def go(i):
            if i in w:
                return w[i]
            t = np.array(self.nodes[i].get("translation", [0, 0, 0]), dtype=np.float64)
            w[i] = t + (go(self.parent[i]) if i in self.parent else 0)
            return w[i]

        for i in range(len(self.nodes)):
            go(i)
        return w

    def eye_front_points(self):
        out = {}
        for name, mask in (("l", self.eye & (self.P[:, 0] > -0.005)), ("r", self.eye & (self.P[:, 0] <= -0.005))):
            pts = self.P[mask]
            c = pts[:, :2].mean(0)
            near = pts[np.linalg.norm(pts[:, :2] - c, axis=1) < 0.006]
            out[name] = near[near[:, 2].argmax()]
        return out


# ---------------------------------------------------------------- normais
def normals_merged(P, tri):
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    key = np.round(P * 1e5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[tri[:, k]], fn)
    n = acc[inv]
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    return n / l


# ---------------------------------------------------------------- ajuste
def similarity_to_template(tpl, L):
    """Umeyama dos pontos-chave dos landmarks (brutos, referencial da frente) para o molde."""
    sk = tpl.P[tpl.skin]
    eyes = tpl.eye_front_points()
    eye_y = (eyes["l"][1] + eyes["r"][1]) / 2
    cost = np.abs(sk[:, 0]) * 3 + np.abs(sk[:, 1] - eye_y)
    mid = sk[np.argsort(cost)[:20]]
    nasion = mid[mid[:, 2].argmax()]
    nose = sk[sk[:, 2].argmax()]
    ch = sk[(np.abs(sk[:, 0]) < 0.015) & (sk[:, 2] > 0.05)]
    chin = ch[ch[:, 1].argmin()]
    mouth = tpl.node_world[tpl.names["skf_lip_volume"]]
    dst = np.stack([eyes["r"], eyes["l"], nasion, nose, chin, mouth])
    src = np.stack([L[LM_IRIS_R], L[LM_IRIS_L], L[LM_NASION], L[LM_NOSE], L[LM_CHIN],
                    (L[LM_MOUTH[0]] + L[LM_MOUTH[1]]) / 2])
    s, R, t = face3d.umeyama(src, dst)
    return s, R, t, chin[1]


def fit_shape(tpl, L, rounds=(1e-4, 1e-5, 2e-6)):
    """Retorna função campo(pontos)->deslocamento e pontos de L no espaço do molde."""
    a, R, b, chin_y = similarity_to_template(tpl, L)
    Lt = a * (L[:468] @ R.T) + b
    V0 = tpl.P
    skin_idx = np.where(tpl.skin)[0]
    nrm0 = tpl.N.astype(np.float64)
    front = skin_idx[nrm0[skin_idx, 2] > 0.25]
    # âncoras com deslocamento zero: nuca e pescoço
    fore_y = Lt[10][1]  # topo da testa (landmark 10) no espaço do molde
    anch = np.where((V0[:, 2] < -0.005) | (V0[:, 1] < chin_y - 0.015) | (V0[:, 1] > fore_y + 0.008))[0]
    rng = np.random.default_rng(0)
    if len(anch) > 500:
        anch = rng.choice(anch, 500, replace=False)
    cur = V0.copy()
    field = None
    for sm in rounds:
        tree = cKDTree(cur[front])
        _, nn = tree.query(Lt)
        vid = front[nn]
        # média por vértice
        uniq, inv = np.unique(vid, return_inverse=True)
        tgt = np.zeros((len(uniq), 3))
        cnt = np.zeros(len(uniq))
        np.add.at(tgt, inv, Lt)
        np.add.at(cnt, inv, 1)
        tgt /= cnt[:, None]
        src = np.vstack([V0[uniq], V0[anch]])
        disp = np.vstack([tgt - V0[uniq], np.zeros((len(anch), 3))])
        field = RBFInterpolator(src, disp, kernel="thin_plate_spline", smoothing=sm, degree=1)
        cur = V0 + _eval(field, V0)
    return field, (a, R, b), Lt


def _eval(field, pts, chunk=2000):
    return np.vstack([field(pts[i:i + chunk]) for i in range(0, len(pts), chunk)])


def deform_all(tpl, field):
    P = tpl.P.copy()
    d = _eval(field, P)
    newP = P + d
    # olhos: deslocamento rígido pelo centro de cada globo
    for mask in (tpl.eye & (P[:, 0] > -0.005), tpl.eye & (P[:, 0] <= -0.005)):
        c = P[mask].mean(0)
        newP[mask] = P[mask] + field(c[None])[0]
    return newP


def new_joint_world(tpl, field):
    out = {}
    for i, w in tpl.node_world.items():
        name = tpl.nodes[i].get("name", "")
        move = name.startswith("skf_") or name == "sk_head"
        out[i] = w + field(w[None])[0] if move else w.copy()
    return out


# ---------------------------------------------------------------- textura
def _raster_skin(tpl, newP, newN, A):
    pos = np.zeros((A, A, 3), np.float32)
    nrm = np.zeros((A, A, 3), np.float32)
    mask = np.zeros((A, A), bool)
    uvp = tpl.UV * A
    for t in tpl.tri:
        if not tpl.skin[t].all():
            continue
        q = uvp[t]
        x0, y0 = np.floor(q.min(0)).astype(int)
        x1, y1 = np.ceil(q.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, A - 1), min(y1, A - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (q[1, 1] - q[2, 1]) * (q[0, 0] - q[2, 0]) + (q[2, 0] - q[1, 0]) * (q[0, 1] - q[2, 1])
        if abs(d) < 1e-9:
            continue
        w0 = ((q[1, 1] - q[2, 1]) * (xs - q[2, 0]) + (q[2, 0] - q[1, 0]) * (ys - q[2, 1])) / d
        w1 = ((q[2, 1] - q[0, 1]) * (xs - q[2, 0]) + (q[0, 0] - q[2, 0]) * (ys - q[2, 1])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if not m.any():
            continue
        iy, ix = np.nonzero(m)
        W = np.stack([w0[m], w1[m], w2[m]], 1)
        pos[y0 + iy, x0 + ix] = W @ newP[t]
        nrm[y0 + iy, x0 + ix] = W @ newN[t]
        mask[y0 + iy, x0 + ix] = True
    return pos, nrm, mask


def bake(tpl, newP, newN, views, sims, sim_t, hair_mode="tint"):
    """views: lista de (img_bgr, raw_landmarks); sims: de fuse_full; sim_t: (a,R,b) front-raw -> molde."""
    A = tpl.atlas.shape[0]
    pos, nrm, mask = _raster_skin(tpl, newP, newN, A)
    ys, xs = np.nonzero(mask)
    p_t, n_t = pos[ys, xs].astype(np.float64), nrm[ys, xs].astype(np.float64)
    n_t /= np.maximum(np.linalg.norm(n_t, axis=1, keepdims=True), 1e-9)
    a, Rt, b = sim_t
    p_front = (p_t - b) @ Rt / a  # R^T (p-b)/a  (linhas)
    cols, wts = [], []
    for (img, _), (s, R, t, cam) in zip(views, sims):
        h, w = img.shape[:2]
        p_v = ((p_front - t) @ R) / s
        px, py = p_v[:, 0], -p_v[:, 1]
        d_t = Rt @ cam
        wv = np.clip(n_t @ d_t, 0, 1) ** 2
        inside = (px > 1) & (px < w - 2) & (py > 1) & (py < h - 2)
        edge = np.clip(np.minimum.reduce([px, w - 1 - px, py, h - 1 - py]) / (0.04 * min(h, w)), 0, 1)
        wv = wv * inside * edge
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        c = np.stack([map_coordinates(rgb[..., k], [np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)], order=1)
                      for k in range(3)], 1)
        cols.append(c)
        wts.append(wv)
    # ganho por vista para casar exposição com a frente
    for i in range(1, len(cols)):
        ov = (wts[0] > 0.3) & (wts[i] > 0.3)
        if ov.sum() > 200:
            g = np.clip(np.median(cols[0][ov], 0) / np.maximum(np.median(cols[i][ov], 0), 1), 0.7, 1.4)
            cols[i] = np.clip(cols[i] * g, 0, 255)
    W = np.stack(wts, 1)
    C = np.stack(cols, 1)
    tot = W.sum(1)
    col = (C * W[..., None]).sum(1) / np.maximum(tot, 1e-6)[:, None]
    cover = np.clip(tot / 0.4, 0, 1)
    good = cover > 0.9
    skin_col = np.median(col[good], 0) if good.any() else np.array([190, 150, 130.0])
    # cor do cabelo: faixa acima da testa na foto da frente
    hair_col = estimate_hair(views[0][0], views[0][1], skin_col)
    # pontos acima da linha da testa no espaço do molde recebem cor do cabelo
    fore_y = newP[tpl.skin][:, 1].max() - 0.06
    fill = np.where((p_t[:, 1] > fore_y)[:, None], hair_col[None], skin_col[None])
    col = col * cover[:, None] + fill * (1 - cover[:, None])
    atlas = tpl.atlas.copy()
    rgb_out = col[:, ::-1].clip(0, 255).astype(np.uint8)  # RGB -> BGR
    # fundo do quadrante da pele (fora das ilhas UV) com a cor da pele
    q = atlas[:A // 2, :A // 2]
    q[..., :3] = np.array(skin_col[::-1], np.uint8)
    q[..., 3] = 255
    atlas[ys, xs, :3] = rgb_out
    atlas[ys, xs, 3] = 255
    # cabelo / barba do molde
    reg_stub = (slice(A // 2, int(A * 0.76)), slice(0, A // 2))
    reg_hair = (slice(A // 2, int(A * 0.82)), slice(A // 2, A))
    atlas[reg_stub][..., 3] = 0  # sem barba/penugem do molde
    sub = atlas[reg_hair]
    if hair_mode == "none":
        sub[..., 3] = 0
    else:
        lum = cv2.cvtColor(sub[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        opq = sub[..., 3] > 20
        ref = max(float(lum[opq].mean()), 1.0) if opq.any() else 1.0
        k = np.clip(lum / ref, 0.3, 2.0)[..., None]
        sub[..., :3] = np.clip(hair_col[::-1][None, None, :] * k, 0, 255).astype(np.uint8)
    return atlas, skin_col, hair_col


def estimate_hair(img, L, skin_col):
    h, w = img.shape[:2]
    px = lambda i: (L[i][0], -L[i][1])
    x1, y1 = px(54)
    x2, _ = px(284)
    xt, yt = px(10)
    hf = np.linalg.norm(np.array(px(10)) - np.array(px(152)))
    ya, yb = int(max(yt - 0.14 * hf, 0)), int(max(yt - 0.03 * hf, 1))
    xa, xb = int(max(min(x1, x2), 0)), int(min(max(x1, x2), w - 1))
    if yb <= ya or xb <= xa:
        return skin_col * 0.35
    patch = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    return np.median(patch, 0)


# ---------------------------------------------------------------- saída
def write_glb(tpl, newP, newN, jworld, atlas):
    js = json.loads(json.dumps(tpl.js))
    binb = bytearray(tpl.bin)
    binb[tpl.off_pos:tpl.off_pos + newP.size * 4] = newP.astype("<f4").tobytes()
    binb[tpl.off_nrm:tpl.off_nrm + newN.size * 4] = newN.astype("<f4").tobytes()
    js["accessors"][tpl.acc_pos]["min"] = newP.min(0).tolist()
    js["accessors"][tpl.acc_pos]["max"] = newP.max(0).tolist()
    for i, nd in enumerate(js["nodes"]):
        if "translation" in nd or i in jworld:
            par = tpl.parent.get(i)
            loc = jworld[i] - (jworld[par] if par is not None else 0)
            if "translation" in nd or np.linalg.norm(loc) > 0:
                nd["translation"] = [float(x) for x in loc]
    ibm = np.zeros((len(tpl.joints), 16), "<f4")
    for k, n in enumerate(tpl.joints):
        p = jworld[n]
        ibm[k] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -p[0], -p[1], -p[2], 1]
    binb[tpl.off_ibm:tpl.off_ibm + ibm.size * 4] = ibm.tobytes()
    ok, png = cv2.imencode(".png", atlas)
    png = png.tobytes()
    binb = binb[:tpl.img_off] + png
    js["bufferViews"][tpl.img_view]["byteLength"] = len(png)
    binb.extend(b"\0" * (-len(binb) % 4))
    js["buffers"][0]["byteLength"] = len(binb)
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(binb)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(j), 0x4E4F534A) + j
            + struct.pack("<II", len(binb), 0x004E4942) + bytes(binb))


# ---------------------------------------------------------------- pipeline
def fit_from_views(tpl, imgs, raws, hair_mode="tint"):
    """imgs/raws: listas [frente, outras...] já com detecção feita (sem None)."""
    P_blend, sims = face3d.fuse_full(raws)
    field, sim_t, _ = fit_shape(tpl, P_blend)
    newP = deform_all(tpl, field)
    newN = normals_merged(newP, tpl.tri)
    atlas, skin, hair = bake(tpl, newP, newN, list(zip(imgs, raws)), sims, sim_t, hair_mode)
    glb = write_glb(tpl, newP, newN, new_joint_world(tpl, field), atlas)
    return glb, {"skin": skin.tolist(), "hair": hair.tolist()}


def generate_from_template(tpl, imgs, hair_mode="tint"):
    """imgs: [frente, lado, lado] (BGR). Usa as vistas em que o rosto for detectado (frente obrigatória)."""
    dets = [face3d.detect(i) for i in imgs]
    if dets[0] is None:
        raise ValueError("Não encontrei rosto na foto da frente.")
    ok = [(im, d[0]) for im, d in zip(imgs, dets) if d is not None]
    glb, info = fit_from_views(tpl, [o[0] for o in ok], [o[1] for o in ok], hair_mode)
    return glb, len(ok), info
