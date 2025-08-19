# backend/engines.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Tuple

import torch, cv2
import torch.nn.functional as F
import torch.jit
import numpy as np

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _to_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _nms_xyxy(boxes: List[Tuple[float,float,float,float]], scores: List[float], iou_thr: float = 0.5) -> List[int]:
    """Простой NMS без torchvision (для малых чисел детекций)."""
    if not boxes:
        return []
    if tv_nms is not None:
        b = torch.tensor(boxes, dtype=torch.float32)
        s = torch.tensor(scores, dtype=torch.float32)
        return tv_nms(b, s, iou_thr).tolist()
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter == 0: return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def _decode_anchor_free(heat: torch.Tensor,
                        wh: torch.Tensor,
                        offset: torch.Tensor,
                        img_h: int, img_w: int,
                        conf_thr: float = 0.30,
                        topk: Optional[int] = None,
                        nms_iou: float = 0.5) -> List[Dict[str, Any]]:
    """
    Декодирует головы anchor-free вида (heatmap, wh, offset) в список боксов.
    Ожидаемые формы: heat=[B,C,Hh,Wh], wh=[B,2,Hh,Wh], offset=[B,2,Hh,Wh].
    """
    assert heat.ndim == 4 and wh.ndim == 4 and offset.ndim == 4, "Unexpected tensor dims"

    # Сигмоида к тепловой карте
    heat = heat.sigmoid()

    B, C, Hh, Wh = heat.shape
    assert B >= 1, "Batch must be >= 1"

    # Локальные максимумы + порог
    hmax = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    peaks = (heat == hmax) & (heat > conf_thr)

    # ВАЖНО: порядок индексов для 4D тензора → (b, c, y, x)
    b_idx, c_idx, y_idx, x_idx = torch.where(peaks)

    # Берём только batch==0 для демо (мы и так гоним по одному изображению)
    if b_idx.numel() == 0:
        return []

    mask_b0 = (b_idx == 0)
    c_idx = c_idx[mask_b0]
    y_idx = y_idx[mask_b0]
    x_idx = x_idx[mask_b0]

    if c_idx.numel() == 0:
        return []

    # Опционально top-k по score
    scores_all = heat[0, c_idx, y_idx, x_idx]
    if topk is not None and scores_all.numel() > topk:
        topk_sel = torch.topk(scores_all, k=topk).indices
        c_idx = c_idx[topk_sel]
        y_idx = y_idx[topk_sel]
        x_idx = x_idx[topk_sel]
        scores_all = scores_all[topk_sel]

    # Пересчёт в пиксели: stride как отношение исходного размера к размеру карты
    stride_x = img_w / float(Wh)
    stride_y = img_h / float(Hh)

    boxes_xyxy: List[Tuple[float,float,float,float]] = []
    scores: List[float] = []

    # Собираем боксы
    for c, y, x, s in zip(c_idx.tolist(), y_idx.tolist(), x_idx.tolist(), scores_all.tolist()):
        # Смещения и размеры в "клетках"
        off_x = float(offset[0, 0, y, x].item())
        off_y = float(offset[0, 1, y, x].item())
        w_cell = float(wh[0, 0, y, x].item())
        h_cell = float(wh[0, 1, y, x].item())

        # Центр и размеры в пикселях
        cx = (x + off_x) * stride_x
        cy = (y + off_y) * stride_y
        w_pix = max(0.0, w_cell * stride_x)
        h_pix = max(0.0, h_cell * stride_y)

        x1 = max(0.0, cx - w_pix / 2.0)
        y1 = max(0.0, cy - h_pix / 2.0)
        x2 = min(float(img_w - 1), cx + w_pix / 2.0)
        y2 = min(float(img_h - 1), cy + h_pix / 2.0)

        # Скипаем вырожденные
        if x2 <= x1 or y2 <= y1:
            continue

        boxes_xyxy.append((x1, y1, x2, y2))
        scores.append(float(s))

    if not boxes_xyxy:
        return []

    # NMS (простой, без torchvision)
    keep = _nms_xyxy(boxes_xyxy, scores, iou_thr=nms_iou)

    dets: List[Dict[str, Any]] = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": scores[i],
            "label": "Fire"  # один класс; при множественных добавьте маппер
        })
    return dets



class ModelHub:
    """Две модели: YOLOv8n и U-Net anchor-free (TorchScript или nn.Module)."""
    def __init__(self, yolo_path: str, unet_path: str):
        self.device = _to_device()
        self.yolo = self._load_yolo(yolo_path) if yolo_path and os.path.exists(yolo_path) else None
        self.unet = self._load_unet(unet_path) if unet_path and os.path.exists(unet_path) else None

    # ---------- YOLO ----------
    def _load_yolo(self, path: str):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed")
        try:
            model = YOLO(path)
            if hasattr(model, "fuse"):
                model.fuse()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def infer_yolo(self, img_bgr: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        if self.yolo is None:
            raise RuntimeError("YOLO model is not available. Put weights into models/model_yolo8n.pt")
        results = self.yolo.predict(source=img_bgr, conf=conf, iou=0.45, device=self.device, verbose=False)
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        boxes = r.boxes
        dets: List[Dict[str, Any]] = []
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(confs), int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            dets.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "conf": float(c),
                "label": names.get(k, "Fire")
            })
        return dets

    # ---------- U-NET ----------
    def _load_unet(self, path: str):
        """
        Порядок:
          1) torch.jit.load (TorchScript)
          2) torch.load (pickled nn.Module)
          Если это чекпойнт/стейт‑дикт — его нельзя восстановить без кода архитектуры → используйте экспорт.
        """
        # 1) TorchScript
        try:
            m = torch.jit.load(path, map_location=self.device)
            m.eval()
            return m
        except Exception:
            pass
        # 2) Полный nn.Module
        try:
            m = torch.load(path, map_location=self.device, weights_only=False)
            if hasattr(m, "eval"):
                m.eval()
                return m
            else:
                raise RuntimeError("Loaded object is not a torch.nn.Module")
        except Exception as e_full:
            raise RuntimeError(
                "Failed to load U-Net model. Provide TorchScript (.pth saved via torch.jit.save) "
                "or a fully pickled torch.nn.Module. "
                f"torch.load error: {e_full}"
            )

    def _infer_unet_single(self, img_bgr: np.ndarray, conf: float) -> List[Dict[str, Any]]:
        if self.unet is None:
            raise RuntimeError("U-Net model is not available. Put weights into models/model_unet_anchor_free.pth")

        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1].copy()

        ten = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ten = ten.to(self.device)

        # ---> NEW: паддинг до кратности 32 (право/низ), чтобы UNet не ломался на нечётных размерах
        pad_mult = 32
        pad_h = (-orig_h) % pad_mult
        pad_w = (-orig_w) % pad_mult
        if pad_h or pad_w:
            # F.pad: (left, right, top, bottom)
            ten = F.pad(ten, (0, pad_w, 0, pad_h), mode="replicate")
        pad_h_total = orig_h + pad_h
        pad_w_total = orig_w + pad_w
        # <---

        with torch.inference_mode():
            out = self.unet(ten)

        # Нормализация формата вывода
        if isinstance(out, (list, tuple)):
            if len(out) == 3:
                heat, wh, offset = out
                topk = int(os.getenv("UNET_TOPK", "512"))
                dets = _decode_anchor_free(
                    heat, wh, offset,
                    img_h=pad_h_total, img_w=pad_w_total,
                    conf_thr=conf, topk=topk, nms_iou=0.5
                )
            elif len(out) == 1:
                out = out[0]
                dets = []
            else:
                dets = []
        elif isinstance(out, dict):
            if all(k in out for k in ("heatmap", "wh", "offset")):
                heat, wh, offset = out["heatmap"], out["wh"], out["offset"]
                topk = int(os.getenv("UNET_TOPK", "512"))
                dets = _decode_anchor_free(
                    heat, wh, offset,
                    img_h=pad_h_total, img_w=pad_w_total,
                    conf_thr=conf, topk=topk, nms_iou=0.5
                )
            elif all(k in out for k in ("boxes", "scores")):
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                dets = []
                for (x1, y1, x2, y2), s in zip(boxes, scores):
                    if s >= conf:
                        dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                                    "conf": float(s), "label": "Fire"})
            else:
                dets = []
        elif torch.is_tensor(out):
            arr = out.detach().cpu().numpy()
            dets = []
            if arr.ndim == 2 and arr.shape[1] in (5, 6):
                for row in arr:
                    x1, y1, x2, y2, s = row[:5]
                    if s >= conf:
                        dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                                    "conf": float(s), "label": "Fire"})
        else:
            dets = []

        # ---> NEW: клиппинг в границы оригинального изображения (срезаем эффект паддинга)
        for d in dets:
            d["x1"] = float(max(0.0, min(d["x1"], orig_w - 1)))
            d["y1"] = float(max(0.0, min(d["y1"], orig_h - 1)))
            d["x2"] = float(max(0.0, min(d["x2"], orig_w - 1)))
            d["y2"] = float(max(0.0, min(d["y2"], orig_h - 1)))
        # <---

        return dets

    def infer_unet(self, img_bgr: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        # Тайловый режим включается переменными окружения (по умолчанию — выкл.)
        tile = int(os.getenv("FD_UNET_TILE", "0"))          # напр., 1024
        overlap = int(os.getenv("FD_UNET_OVERLAP", "128"))  # перекрытие
        if tile and max(img_bgr.shape[:2]) > tile:
            H, W = img_bgr.shape[:2]
            step = max(32, tile - overlap)                  # шаг сетки
            all_boxes: List[Dict[str, Any]] = []
            for y0 in range(0, H, step):
                for x0 in range(0, W, step):
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)
                    crop = img_bgr[y0:y1, x0:x1]
                    dets_t = self._infer_unet_single(crop, conf=conf)
                    # переносим координаты тайла в глобальные
                    for d in dets_t:
                        d["x1"] += x0; d["x2"] += x0
                        d["y1"] += y0; d["y2"] += y0
                    all_boxes.extend(dets_t)
            # финальный NMS по всем тайлам
            boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in all_boxes]
            scores = [float(d.get("conf", 0.0)) for d in all_boxes]
            keep = _nms_xyxy(boxes, scores, iou_thr=0.5)
            dets = [all_boxes[i] for i in keep]
            # финальный клиппинг
            for d in dets:
                d["x1"] = float(max(0.0, min(d["x1"], W - 1)))
                d["y1"] = float(max(0.0, min(d["y1"], H - 1)))
                d["x2"] = float(max(0.0, min(d["x2"], W - 1)))
                d["y2"] = float(max(0.0, min(d["y2"], H - 1)))
            return dets
        else:
            return self._infer_unet_single(img_bgr, conf=conf)
