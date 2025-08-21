from __future__ import annotations
from typing import List, Dict, Any
import os
import cv2
import base64
import numpy as np

def draw_bboxes(img_bgr: np.ndarray, dets: List[Dict[str, Any]], color=(255, 77, 124)) -> np.ndarray:
    vis = img_bgr.copy()
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        color = tuple(map(int, d.get("color", color)))
        conf = float(d.get("conf", 0.0))
        label = d.get("label", "Fire")
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{i}. {label} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1 - 6, th + 6)
        cv2.rectangle(vis, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
        cv2.putText(vis, text, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

def encode_image_base64(img_bgr: np.ndarray, ext="jpg") -> str:
    ok, buf = cv2.imencode(f".{ext}", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if ext.lower() in ("jpg", "jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"
