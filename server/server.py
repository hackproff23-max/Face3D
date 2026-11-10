import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

import face3d

app = FastAPI(title="Face3D")
TEMPLATE_PATH = os.environ.get("TEMPLATE_GLB", os.path.join(os.path.dirname(__file__), "template.glb"))
_tpl = None


def template():
    """Carrega o molde (cabeça exportada do seu mod) se existir."""
    global _tpl
    if _tpl is None and os.path.exists(TEMPLATE_PATH):
        import fit
        _tpl = fit.Template(TEMPLATE_PATH)
    return _tpl


def _decode(data: bytes, name: str):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, f"Imagem '{name}' inválida.")
    return img


@app.get("/")
def health():
    return {"ok": True, "template": os.path.exists(TEMPLATE_PATH)}


@app.post("/generate")
async def generate(front: UploadFile = File(...), left: UploadFile = File(...), right: UploadFile = File(...),
                   hair: str = "tint"):
    fb, lb, rb = await front.read(), await left.read(), await right.read()
    imgs = [_decode(fb, "frente"), _decode(lb, "esquerda"), _decode(rb, "direita")]
    try:
        tpl = template()
        if tpl is not None:
            import fit
            glb, used, _ = fit.generate_from_template(tpl, imgs, hair if hair in ("tint", "none") else "tint")
            mode = "template"
        else:
            glb, used = face3d.generate(imgs[0], imgs[1], imgs[2], fb)
            mode = "generico"
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(glb, media_type="model/gltf-binary", headers={"X-Views-Used": str(used), "X-Mode": mode})
