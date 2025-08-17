# backend/main.py
import io
import os
from typing import Literal, List, Dict, Any

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .engines import ModelHub
from .utils import draw_bboxes, encode_image_base64

app = FastAPI(title="Fire Detection Demo", version="1.0.1")

FRONT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")

hub: ModelHub | None = None


@app.on_event("startup")
def _load_models():
    global hub
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    yolo_path = os.path.join(models_dir, "model_yolo8n.pt")
    unet_path = os.path.join(models_dir, "model_unet_anchor_free.pth")
    hub = ModelHub(yolo_path=yolo_path, unet_path=unet_path)


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(FRONT_DIR, "index.html"))


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    model: Literal["yolo", "unet"] = Form("yolo"),
    conf: float = Form(0.25)
) -> JSONResponse:
    global hub
    if hub is None:
        raise HTTPException(status_code=500, detail="Models are not initialized")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Декодируем изображение (OpenCV -> Pillow fallback)
    import cv2
    img = None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception:
        try:
            from PIL import Image
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported image format")

    H, W = img.shape[:2]

    # Инференс
    try:
        if model == "yolo":
            dets = hub.infer_yolo(img, conf=conf)
            color = (255, 194, 0)
        else:
            dets = hub.infer_unet(img, conf=conf)
            color = (255, 77, 124)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    vis = draw_bboxes(img, dets, color=color)
    data_url = encode_image_base64(vis, ext="jpg")

    bboxes: List[Dict[str, Any]] = []
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        w, h = x2 - x1, y2 - y1
        bboxes.append({
            "id": i,
            "label": d.get("label", "Fire"),
            "conf": float(d.get("conf", 0.0)),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": w, "h": h
        })

    return JSONResponse({
        "image": {"data": data_url, "width": W, "height": H},
        "bboxes": bboxes,
        "model": model,
        "conf": conf
    })


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
