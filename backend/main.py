# backend/main.py
import io
import os
from typing import Literal, List, Dict, Any

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageOps

import math, time
import cv2

# --- Torch threading: единоразовая настройка ---
import torch, os  # <= важно: импортируем torch прежде чем использовать
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
# set_num_interop_threads разрешён только один раз; вызываем, только если явно задан
if os.getenv("TORCH_NUM_INTEROP_THREADS"):
    torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS")))
# --- конец блока настройки ---

from .engines import ModelHub
from .utils import draw_bboxes, encode_image_base64

# Цвета моделей (BGR для OpenCV!)
MODEL_COLORS_BGR = {
    "yolo": (255, 194, 0),
    "unet": (255, 77, 124),
}
MODEL_COLORS_HEX = {
    "yolo": "#00c2ff",
    "unet": "#7c4dff",
}

FD_MAX_UPLOAD_BYTES   = int(os.getenv("FD_MAX_UPLOAD_BYTES", str(128 * 1024 * 1024)))  # 128 MB
# ЕДИНЫЕ лимиты (используются и в раннем thumbnail, и в _downscale_if_needed):
FD_MAX_SIDE           = int(os.getenv("FD_MAX_SIDE", "1536"))
FD_MAX_PIXELS         = int(os.getenv("FD_MAX_PIXELS", "3500000"))
# Лимиты для превью (картинка, которая уходит в base64 клиенту)
FD_PREVIEW_MAX_SIDE   = int(os.getenv("FD_PREVIEW_MAX_SIDE", "1280"))
FD_PREVIEW_JPEG_QUAL  = int(os.getenv("FD_PREVIEW_JPEG_QUAL", "85"))

def _downscale_if_needed(img_bgr):
    h, w = img_bgr.shape[:2]
    scale_side   = FD_MAX_SIDE / max(h, w)
    scale_pixels = math.sqrt(FD_MAX_PIXELS / (h * w))
    scale = min(1.0, scale_side, scale_pixels)
    if scale >= 1.0:
        return img_bgr, 1.0, 1.0
    new_w = max(32, int((w * scale) // 32 * 32))
    new_h = max(32, int((h * scale) // 32 * 32))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # коэффициенты для обратного масштабирования предсказаний
    sx, sy = w / new_w, h / new_h
    return resized, sx, sy

app = FastAPI(title="Fire Detection Demo", version="1.0.1")

FRONT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")

hub: ModelHub | None = None

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    msg = str(exc)
    if "DefaultCPUAllocator" in msg or "can't allocate memory" in msg:
        return JSONResponse(status_code=413, content={"detail": "Image too large for U-Net on this server. Try YOLO or upload a smaller image."})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(MemoryError)
async def mem_error_handler(request: Request, exc: MemoryError):
    return JSONResponse(status_code=413, content={"detail": "Out of memory during inference"})

@app.on_event("startup")
def _load_models():
    global hub
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    yolo_path = os.path.join(models_dir, "model_yolo8n.pt")
    unet_path = os.path.join(models_dir, "model_unet_anchor_free.pth")
    hub = ModelHub(yolo_path=yolo_path, unet_path=unet_path)

@app.on_event("startup")
def _warmup():
    import numpy as np, torch
    dummy = np.zeros((128,128,3), np.uint8)
    with torch.inference_mode():
        _ = hub.infer_yolo(dummy, conf=0.25)
        _ = hub.infer_unet(dummy, conf=0.25)

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(FRONT_DIR, "index.html"))


@app.post("/api/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    model: Literal["yolo", "unet"] = Form("yolo"),
    conf: float = Form(0.25)
):
    # Если сервер знает фактический размер временного файла — отсекаем сразу
    try:
        import os, stat
        fileno = file.file.fileno()
        size_hint = os.fstat(fileno).st_size
        max_bytes = int(os.getenv("FD_MAX_UPLOAD_BYTES", "41943040"))
        if size_hint and size_hint > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
    except Exception:
        pass  # не везде доступен fileno

    # Без лишней копии в память: читаем прямо из spooled temp file
    try:
        file.file.seek(0)
        pil = Image.open(file.file)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception:
        raise HTTPException(status_code=415, detail="Unsupported or broken image")

    # РАННИЙ даунскейл перед переводом в NumPy (экономит RAM) — используем ЕДИНЫЕ лимиты
    W0, H0 = pil.size
    if max(W0, H0) > FD_MAX_SIDE or (W0 * H0) > FD_MAX_PIXELS:
        pil.thumbnail((FD_MAX_SIDE, FD_MAX_SIDE), Image.Resampling.LANCZOS)

    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    H, W = img.shape[:2]
    t0 = time.perf_counter()

    # --- безопасный режим для U-Net ---
    if model == "unet":
        img_in, sx, sy = _downscale_if_needed(img)
        try:
            dets = hub.infer_unet(img_in, conf=conf)
        except RuntimeError as e:
            # превратим OOM в понятную 413 JSON-ошибку
            if "DefaultCPUAllocator" in str(e) or "can't allocate memory" in str(e):
                raise HTTPException(status_code=413, detail="Image too large for U-Net on this server")
            raise
        # масштабируем боксы обратно к исходному размеру
        if (sx, sy) != (1.0, 1.0):
            for d in dets:
                d["x1"], d["x2"] = d["x1"] * sx, d["x2"] * sx
                d["y1"], d["y2"] = d["y1"] * sy, d["y2"] * sy
    else:
        dets = hub.infer_yolo(img, conf=conf)

    # сгенерируем таблицу боксов для фронта
    bboxes: List[Dict[str, Any]] = []
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = map(float, (d["x1"], d["y1"], d["x2"], d["y2"]))
        w, h = float(x2 - x1), float(y2 - y1)
        bboxes.append({
            "id": i,                                # ← добавили номер
            "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
            "w": round(w), "h": round(h),
            "conf": float(d.get("conf", 0.0)),
            "label": d.get("label", "Fire"),
            "model": model,                         # ← на всякий случай тоже кладём
            "color": MODEL_COLORS_BGR.get(model, (255, 77, 124)),  # для сервера (BGR)
            "color_hex": MODEL_COLORS_HEX.get(model, "#00cc66"),  # для UI-таблицы
    })

    vis = draw_bboxes(img, bboxes)
    data_url = encode_image_base64(vis, ext="jpg")
    dt = (time.perf_counter() - t0) * 1000.0

    return JSONResponse({
        "image": {"data": data_url, "width": W, "height": H},
        "bboxes": bboxes,
        "model": model,
        "conf": conf,
        "time_ms": round(dt, 1)
    })


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
