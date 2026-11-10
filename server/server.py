import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

import face3d

app = FastAPI(title="Face3D")


def _decode(data: bytes, name: str):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, f"Imagem '{name}' inválida.")
    return img


@app.get("/")
def health():
    return {"ok": True}


@app.post("/generate")
async def generate(front: UploadFile = File(...), left: UploadFile = File(...), right: UploadFile = File(...)):
    fb, lb, rb = await front.read(), await left.read(), await right.read()
    imgs = [_decode(fb, "frente"), _decode(lb, "esquerda"), _decode(rb, "direita")]
    try:
        glb, used = face3d.generate(imgs[0], imgs[1], imgs[2], fb)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(glb, media_type="model/gltf-binary", headers={"X-Views-Used": str(used)})
