const els = {
  // Загрузка
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  fileHint: document.getElementById('file-hint'),
  fileChip: document.getElementById('file-chip'),
  fileName: document.getElementById('file-name'),
  fileSize: document.getElementById('file-size'),
  clearFile: document.getElementById('clear-file'),

  // Кнопки
  runBtn: document.getElementById('run-btn'),

  // Параметры
  model: document.getElementById('model'),
  conf: document.getElementById('conf'),
  confVal: document.getElementById('conf-val'),
  confRange: document.getElementById('conf-range'),
  confBubble: document.getElementById('conf-bubble'),

  // Viewer
  zoomViewport: document.getElementById('zoom-viewport'),
  zoomCanvas: document.getElementById('zoom-canvas'),
  resultImg: document.getElementById('result-img'),
  zoomIn: document.getElementById('zoom-in'),
  zoomOut: document.getElementById('zoom-out'),
  zoomFit: document.getElementById('zoom-fit'),
  zoom100: document.getElementById('zoom-100'),
  zoomValue: document.getElementById('zoom-value'),
  skeleton: document.getElementById('skeleton'),

  // Мета/таблица/прочее
  metaModel: document.getElementById('meta-model'),
  metaSize: document.getElementById('meta-size'),
  metaConf: document.getElementById('meta-conf'),
  metaTime: document.getElementById('meta-time'),
  bboxesBody: document.getElementById('bboxes-body'),
  error: document.getElementById('error'),

  // i18n
  langRU: document.getElementById('lang-ru'),
  langEN: document.getElementById('lang-en'),
};

let currentFile = null;
let currentLang = localStorage.getItem('lang') || 'ru';
let i18n = {};
let zoomEnabled = false; // зум доступен только при наличии изображения

['dragstart','selectstart'].forEach(type => {
  els.zoomViewport.addEventListener(type, e => e.preventDefault());
  els.zoomCanvas.addEventListener(type, e => e.preventDefault());
  els.resultImg.addEventListener(type, e => e.preventDefault());
});

// ---------------- I18N ----------------
async function loadLocale(lang) {
  const res = await fetch(`/static/locales/${lang}.json`);
  i18n = await res.json();
  document.documentElement.lang = lang;
  applyI18n();
  (lang === 'ru' ? els.langRU : els.langEN).classList.add('active');
  (lang === 'ru' ? els.langEN : els.langRU).classList.remove('active');
}
function applyI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (i18n[k]) el.textContent = i18n[k];
  });
}

// ---------------- Helpers ----------------
function humanSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const u = ['B','KB','MB','GB']; let i = 0; let v = bytes;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
}

function setFile(file) {
  currentFile = file || null;

  if (currentFile) {
    els.fileName.textContent = currentFile.name;
    els.fileSize.textContent = humanSize(currentFile.size);

    // Показать статус‑пилюлю, скрыть подсказку
    els.fileChip.hidden = false;
    els.fileHint.hidden = true;

    els.runBtn.disabled = false;
  } else {
    // Полностью очистить UI статуса
    els.fileChip.hidden = true;
    els.fileName.textContent = '';
    els.fileSize.textContent = '';

    // Показать подсказку «Файл не выбран»
    els.fileHint.hidden = false;
    els.fileHint.setAttribute('data-i18n','no_file');
    els.fileHint.textContent = i18n['no_file'] || 'No file selected';

    // Очистить сам input[type=file]
    els.fileInput.value = ""; // надёжный кроссбраузерный сброс.

    // Заблокировать запуск
    els.runBtn.disabled = true;
  }
  els.error.hidden = true;
}

function setLoading(on) {
  els.runBtn.disabled = on || !currentFile;
  els.runBtn.textContent = on ? (i18n['running'] || 'Running...') : (i18n['run_btn'] || 'Run detection');
  els.skeleton.hidden = !on;
  els.zoomViewport.setAttribute('aria-busy', on ? 'true' : 'false');
  if (on) setZoomEnabled(false);
}

// -------- Очистка результата (viewer + мета + таблица) --------
function clearResults() {
  // стоп загрузки/скелета (если был)
  setLoading(false);

  // Очистить изображение и вернуть зум в исходное состояние
  els.resultImg.removeAttribute('src');
  Z.scale = 1; Z.fit = 1; Z.x = 0; Z.y = 0;
  els.zoomCanvas.style.transform = 'translate(0px, 0px) scale(1)';
  els.skeleton.hidden = true;
  els.zoomViewport.setAttribute('aria-busy', 'false');
  setZoomEnabled(false);

  // Сбросить мета‑инфо
  els.metaModel.textContent = '—';
  els.metaSize.textContent  = '—';
  els.metaConf.textContent  = '—';
  els.metaTime.textContent  = '—';

  // Таблица: показать «детекции отсутствуют»
  els.bboxesBody.innerHTML = '';
  const tr = document.createElement('tr');
  tr.className = 'muted';
  const td = document.createElement('td');
  td.colSpan = 9;
  td.textContent = i18n['no_detections'] || 'No detections';
  tr.appendChild(td);
  els.bboxesBody.appendChild(tr);

  // Ошибки — скрыть/очистить
  els.error.textContent = '';
  els.error.hidden = true;

  // Обновить визуал слайдера (пузырь/прогресс) на всякий случай
  if (typeof updateRangeProgress === 'function') updateRangeProgress();

  // Если внедряли «сворачивание таблицы», синхронизируем кнопку
  if (typeof updateTableToggle === 'function') {
    try { updateTableToggle(); } catch {}
  }
}

// ---------------- Drag & Drop / Keyboard ----------------
['dragenter','dragover'].forEach(evt => {
  els.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(evt => {
  els.dropzone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.remove('dragover');
  });
});
els.dropzone.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) setFile(file);
});
els.fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) setFile(file);
});
// Очистка по крестику: делаем фокус обратно на зону, чтобы UX был плавным
els.clearFile.addEventListener('click', (e) => {
  e.preventDefault();
  setFile(null);
  clearResults();
  els.conf.value = '0.25';
  updateRangeProgress();
  els.dropzone.focus();
});

// ---------------- Slider UI ----------------
// ---------------- Slider UI ----------------
function updateRangeProgress() {
  const min = parseFloat(els.conf.min || '0');
  const max = parseFloat(els.conf.max || '1');
  const val = parseFloat(els.conf.value);
  const t = (val - min) / (max - min); // 0..1

  const wrap = els.confRange || els.conf.parentElement;
  const wrapW = wrap.clientWidth;
  const cs = getComputedStyle(wrap);
  const thumb = parseFloat(cs.getPropertyValue('--thumb')) || 22;

  // Центр ползунка в пикселях относительно контейнера
  const pxCenter = (t * (wrapW - thumb)) + (thumb / 2);

  // Обновляем текст заранее, чтобы корректно измерить ширину пузыря
  if (els.confBubble) els.confBubble.textContent = val.toFixed(2);
  els.confVal.textContent = val.toFixed(2);

  // Кламп пузыря, чтобы он не выходил за пределы слайдера
  let bubbleLeft = pxCenter;
  if (els.confBubble) {
    const bubbleW = els.confBubble.offsetWidth || 0;
    const half = bubbleW / 2;
    // небольшие поля безопасности в 2px
    bubbleLeft = Math.min(Math.max(pxCenter, half + 2), Math.max(wrapW - half - 2, 0));
  }

  const progressW = Math.max(pxCenter - 1, 0);
  wrap.style.setProperty('--progress-w', `${progressW}px`);
  wrap.style.setProperty('--bubble-left', `${bubbleLeft}px`);
}

els.conf.addEventListener('input', updateRangeProgress);
els.conf.addEventListener('change', updateRangeProgress);
window.addEventListener('resize', () => {
  updateRangeProgress();
  fitIfLoaded();
});
updateRangeProgress();

// ---------------- Zoom Viewer ----------------
const Z = { scale: 1, fit: 1, x: 0, y: 0, minFactor: 0.5, maxFactor: 10 };
function hasImage() { return !!(els.resultImg.src && els.resultImg.naturalWidth && els.resultImg.naturalHeight); }
function setZoomEnabled(on) {
  zoomEnabled = !!on;
  els.zoomViewport.classList.toggle('zoom--disabled', !zoomEnabled);
  [els.zoomIn, els.zoomOut, els.zoomFit, els.zoom100].forEach(btn => (btn.disabled = !zoomEnabled));

  // Новое: затемняем плашку "100%" так же, как кнопки
  const wrap = els.zoomValue?.parentElement;
  if (wrap) wrap.setAttribute('aria-disabled', (!zoomEnabled).toString());
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function contentSize() {
  const iw = els.resultImg.naturalWidth || 1;
  const ih = els.resultImg.naturalHeight || 1;
  return { iw, ih, w: iw * Z.scale, h: ih * Z.scale };
}
function viewportSize() {
  const vw = els.zoomViewport.clientWidth || 1;
  const vh = els.zoomViewport.clientHeight || 1;
  return { vw, vh };
}
function applyTransform() {
  els.zoomCanvas.style.transform = `translate(${Z.x}px, ${Z.y}px) scale(${Z.scale})`;
  const perc = Math.round((Z.scale / Z.fit) * 100);
  els.zoomValue.textContent = `${perc}%`;
}
function computeFit() {
  const { vw, vh } = viewportSize();
  const iw = els.resultImg.naturalWidth || 1;
  const ih = els.resultImg.naturalHeight || 1;
  return Math.min(vw / iw, vh / ih);
}
function centerCanvas() {
  const { vw, vh } = viewportSize();
  const { w, h } = contentSize();
  Z.x = (vw - w) / 2;
  Z.y = (vh - h) / 2;
}
function clampPan() {
  const { vw, vh } = viewportSize();
  const { w, h } = contentSize();
  if (w <= vw) Z.x = (vw - w) / 2;
  else Z.x = clamp(Z.x, vw - w, 0);
  if (h <= vh) Z.y = (vh - h) / 2;
  else Z.y = clamp(Z.y, vh - h, 0);
}
function setScaleAt(newScale, clientX, clientY) {
  const minS = Z.fit * Z.minFactor;
  const maxS = Z.fit * Z.maxFactor;
  const prev = Z.scale;
  Z.scale = clamp(newScale, minS, maxS);

  const rect = els.zoomViewport.getBoundingClientRect();
  const cx = clientX - rect.left - Z.x;
  const cy = clientY - rect.top - Z.y;
  const k = Z.scale / prev;
  Z.x -= cx * (k - 1);
  Z.y -= cy * (k - 1);

  clampPan();
  applyTransform();
}
function zoomBy(factor, cx, cy) {
  const rect = els.zoomViewport.getBoundingClientRect();
  setScaleAt(Z.scale * factor, cx ?? (rect.left + rect.width / 2), cy ?? (rect.top + rect.height / 2));
}
function zoomToFit() {
  Z.fit = computeFit() || 1;
  Z.scale = Z.fit;
  centerCanvas();
  applyTransform();
}
function zoomTo1x() {
  const { vw, vh } = viewportSize();
  setScaleAt(1, els.zoomViewport.getBoundingClientRect().left + vw / 2, els.zoomViewport.getBoundingClientRect().top + vh / 2);
  clampPan();
  applyTransform();
}
function fitIfLoaded() {
  if (els.resultImg.naturalWidth && els.resultImg.naturalHeight) zoomToFit();
}
els.resultImg.addEventListener('load', () => {
  zoomToFit();
  // скрываем скелет только когда картинка готова
  setLoading(false);
  setZoomEnabled(true);
});
els.resultImg.addEventListener('error', () => {
  setZoomEnabled(false);
  setLoading(false);
});

// Колёсико — зум
els.zoomViewport.addEventListener('wheel', (e) => {
  if (!zoomEnabled) return;
  e.preventDefault();
  const factor = Math.pow(1.0015, -e.deltaY);
  zoomBy(factor, e.clientX, e.clientY);
}, { passive: false });

// Панорамирование и пинч
let isPanning = false,
  lastX = 0,
  lastY = 0;
const activePointers = new Map();
let pinchStart = null;

els.zoomViewport.addEventListener('pointerdown', (e) => {
  if (!zoomEnabled) return;
  e.preventDefault(); // не даём браузеру начинать выделение/drag
  els.zoomViewport.setPointerCapture(e.pointerId);
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (activePointers.size === 1) {
    isPanning = true;
    lastX = e.clientX;
    lastY = e.clientY;
    els.zoomViewport.classList.add('grabbing');
  } else if (activePointers.size === 2) {
    const pts = Array.from(activePointers.values());
    const dx = pts[0].x - pts[1].x,
      dy = pts[0].y - pts[1].y;
    pinchStart = { dist: Math.hypot(dx, dy), scale: Z.scale };
  }
});
els.zoomViewport.addEventListener('pointermove', (e) => {
  if (!zoomEnabled) return;
  if (!activePointers.has(e.pointerId)) return;
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (activePointers.size === 2 && pinchStart) {
    const pts = Array.from(activePointers.values());
    const dx = pts[0].x - pts[1].x,
      dy = pts[0].y - pts[1].y;
    const dist = Math.hypot(dx, dy);
    const midX = (pts[0].x + pts[1].x) / 2;
    const midY = (pts[0].y + pts[1].y) / 2;
    const factor = dist / (pinchStart.dist || 1);
    setScaleAt(pinchStart.scale * factor, midX, midY);
  } else if (isPanning) {
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    Z.x += dx;
    Z.y += dy;
    clampPan();
    applyTransform();
  }
});
function endPointer(e) {
  activePointers.delete(e.pointerId);
  if (activePointers.size < 2) pinchStart = null;
  if (activePointers.size === 0) {
    isPanning = false;
    els.zoomViewport.classList.remove('grabbing');
  }
}
['pointerup', 'pointercancel', 'pointerleave'].forEach(type =>
  els.zoomViewport.addEventListener(type, endPointer)
);

// dblclick: Fit <-> 1:1
els.zoomViewport.addEventListener('dblclick', (e) => {
  if (!zoomEnabled) return;
  const nearFit = Math.abs(Z.scale - Z.fit) / Z.fit < 0.05;
  if (nearFit) setScaleAt(1, e.clientX, e.clientY);
  else zoomToFit();
});

// Кнопки и клавиатура
els.zoomIn.addEventListener('click', () => {
  if (zoomEnabled) zoomBy(1.2);
});
els.zoomOut.addEventListener('click', () => {
  if (zoomEnabled) zoomBy(1 / 1.2);
});
els.zoomFit.addEventListener('click', () => {
  if (zoomEnabled) zoomToFit();
});
els.zoom100.addEventListener('click', () => {
  if (zoomEnabled) zoomTo1x();
});

window.addEventListener('keydown', (e) => {
  if (!zoomEnabled) return;
  if (e.key === '+' || e.key === '=') zoomBy(1.2);
  if (e.key === '-') zoomBy(1 / 1.2);
  if (e.key.toLowerCase() === 'f') zoomToFit();
  if (e.key === '0') zoomTo1x();
});

// ---------------- Run inference ----------------
els.runBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  // Показать skeleton до запроса
  setLoading(true);
  const t0 = performance.now();

  try {
    const fd = new FormData();
    fd.append('file', currentFile);
    fd.append('model', els.model.value);
    fd.append('conf', els.conf.value);

    const res = await fetch('/api/detect', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Request failed');

    // картинка
    els.resultImg.src = data.image?.data || '';
    if (!els.resultImg.src) {
      // бэкенд не прислал картинку
      setZoomEnabled(false);
      setLoading(false);
    }

    els.metaModel.textContent = data.model === 'yolo'
      ? (i18n['model_yolo'] || 'YOLOv8n')
      : (i18n['model_unet'] || 'U-Net');
    els.metaSize.textContent = `${data.image?.width} × ${data.image?.height}`;
    els.metaConf.textContent = (+data.conf).toFixed(2);

    // время (бэкенд приоритетен)
    const backendMs = Number(data.time_ms ?? data.infer_time_ms ?? data.ms ?? data.time);
    const elapsedMs = Number.isFinite(backendMs) ? backendMs : (performance.now() - t0);
    els.metaTime.textContent = elapsedMs < 1000
      ? `${Math.round(elapsedMs)} ms`
      : `${(elapsedMs / 1000).toFixed(2)} s`;

    // таблица bbox-ов
    els.bboxesBody.innerHTML = '';
    const dets = data.bboxes || [];
    if (!dets.length) {
      const tr = document.createElement('tr');
      tr.className = 'muted';
      const td = document.createElement('td');
      td.colSpan = 9;
      td.textContent = i18n['no_detections'] || 'No detections';
      tr.appendChild(td);
      els.bboxesBody.appendChild(tr);
    } else {
      for (const d of dets) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${d.id}</td>
          <td>${d.label || 'Fire'}</td>
          <td>${d.x1}</td><td>${d.y1}</td><td>${d.x2}</td><td>${d.y2}</td>
          <td>${d.w}</td><td>${d.h}</td>
          <td>${(d.conf * 100).toFixed(1)}%</td>
        `;
        els.bboxesBody.appendChild(tr);
      }
    }
    els.error.hidden = true;
    // (!) setLoading(false) произойдёт на load изображения, чтобы не мигал скелет
  } catch (e) {
    els.error.textContent = e.message || String(e);
    els.error.hidden = false;
    setLoading(false); // в случае ошибки прячем скелет немедленно
  }
});

// ---------------- Lang switch ----------------
els.langRU.addEventListener('click', () => {
  currentLang = 'ru';
  localStorage.setItem('lang', 'ru');
  loadLocale('ru');
});
els.langEN.addEventListener('click', () => {
  currentLang = 'en';
  localStorage.setItem('lang', 'en');
  loadLocale('en');
});

// init
loadLocale(currentLang).then(() => {
  els.confVal.textContent = (+els.conf.value).toFixed(2);
  initModelCombo('model'); // см. п.3
  setZoomEnabled(false); // на старте — без зума, пока нет изображения
});

// ---------------- Custom select (desktop only) ----------------
function initModelCombo(selectId) {
  const select = document.getElementById(selectId);
  const field = document.getElementById(`${selectId}-field`);
  const combo = document.getElementById(`${selectId}-combo`);
  const list = document.getElementById(`${selectId}-list`);
  const valueEl = field?.querySelector('.combo__value');
  if (!select || !field || !combo || !list || !valueEl) return;

  const useCustom = window.matchMedia('(pointer: fine)').matches;
  // (re)build список из <option>
  list.innerHTML = '';
  Array.from(select.options).forEach(opt => {
    const li = document.createElement('li');
    li.className = 'combo__option';
    li.setAttribute('role', 'option');
    li.tabIndex = -1;
    li.dataset.value = opt.value;
    li.textContent = opt.textContent;
    if (opt.selected) li.setAttribute('aria-selected', 'true');
    list.appendChild(li);
  });
  function updateFromSelect() {
    const opt = select.options[select.selectedIndex];
    valueEl.textContent = opt ? opt.textContent : '';
    Array.from(list.children).forEach(li => {
      li.setAttribute('aria-selected', li.dataset.value === select.value ? 'true' : 'false');
    });
  }
  updateFromSelect();

  if (!useCustom) {
    combo.setAttribute('hidden', '');
    list.hidden = true;
    return;
  } else {
    combo.removeAttribute('hidden');
  }

  if (!field.dataset.comboReady) {
    field.dataset.comboReady = '1';
    let open = false;
    function openList() {
      if (open) return;
      open = true;
      combo.setAttribute('aria-expanded', 'true');
      list.hidden = false;
      (list.querySelector('[aria-selected="true"]') || list.firstElementChild)?.focus();
    }
    function closeList() {
      if (!open) return;
      open = false;
      combo.setAttribute('aria-expanded', 'false');
      list.hidden = true;
    }
    combo.addEventListener('click', () => (open ? closeList() : openList()));
    // Закрываем по клику ВНЕ через единый метод, чтобы сбросить флаг `open`
    document.addEventListener('pointerdown', (e) => {
      if (!field.contains(e.target)) closeList();
    });
    list.addEventListener('click', (e) => {
      const li = e.target.closest('.combo__option');
      if (!li) return;
      select.value = li.dataset.value;
      updateFromSelect();
      closeList();
      select.dispatchEvent(new Event('change', { bubbles: true }));
      combo.focus();
    });
    list.addEventListener('keydown', (e) => {
      const items = Array.from(list.querySelectorAll('.combo__option'));
      const i = items.indexOf(document.activeElement);
      if (e.key === 'Escape') {
        e.preventDefault();
        closeList();
        combo.focus();
        return;
      }
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const li = document.activeElement.closest('.combo__option');
        if (li) {
          select.value = li.dataset.value;
          updateFromSelect();
          closeList();
          combo.focus();
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        (items[i + 1] || items[0]).focus();
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        (items[i - 1] || items[items.length - 1]).focus();
      }
      if (e.key === 'Home') {
        e.preventDefault();
        items[0]?.focus();
      }
      if (e.key === 'End') {
        e.preventDefault();
        items[items.length - 1]?.focus();
      }
    });
    document.addEventListener('click', (e) => {
      if (!field.contains(e.target)) {
        list.hidden = true;
        combo.setAttribute('aria-expanded', 'false');
      }
    });
    window.matchMedia('(pointer: fine)').addEventListener('change', () => initModelCombo(selectId));
    select.addEventListener('change', updateFromSelect);
  }
}
