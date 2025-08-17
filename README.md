# Fire Detection • YOLOv8n & U‑Net (anchor‑free)

Демо веб‑приложения на **FastAPI**:
- Drag‑and‑drop загрузка
- Переключатель **YOLOv8n / U‑Net**
- Нумерованные bbox‑ы с координатами и вероятностями
- Ограниченная область просмотра изображения (не «выпрыгивает» за экран)
- Адаптивный, кроссбраузерный, RU/EN

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# поместите веса:
#   models/model_yolo8n.pt
#   models/model_unet_anchor_free.pth
uvicorn backend.main:app --host 0.0.0.0 --port 8000
