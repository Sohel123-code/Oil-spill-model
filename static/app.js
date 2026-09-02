/**
 * app.js — OilSight frontend logic (v3.0)
 * Multi-select checkboxes, donut chart summary, result filtering
 */

// Use absolute path so requests resolve correctly inside HF Spaces iframe
const API = '/predict/batch';
const MAX_THUMBS = 12;

let selectedFiles = [];           // All loaded files
let checkedIndices = new Set();   // Indices of files the user has ticked
let lastResults = [];             // Stored for filtering

// ── DOM ELEMENTS ─────────────────────────────────────
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const uploadTitle    = document.getElementById('upload-title');
const uploadHint     = document.getElementById('upload-hint');
const previewRow     = document.getElementById('preview-row');
const selectToolbar  = document.getElementById('select-toolbar');
const btnSelectAll   = document.getElementById('btn-select-all');
const btnDeselectAll = document.getElementById('btn-deselect-all');
const selectCountEl  = document.getElementById('select-count');
const actionBar      = document.getElementById('action-bar');
const analyseBtn     = document.getElementById('analyse-btn');
const thresholdSldr  = document.getElementById('threshold-slider');
const thresholdDisp  = document.getElementById('threshold-display');
const loadingSection = document.getElementById('loading-section');
const loadingText    = document.getElementById('loading-text');
const summarySection = document.getElementById('summary-section');
const resultsGrid    = document.getElementById('results-grid');
const resetBtn       = document.getElementById('reset-btn');

// Lightbox Modal
const lightboxBackdrop = document.getElementById('lightbox-backdrop');
const lightboxImg      = document.getElementById('lightbox-img');
const lightboxFilename = document.getElementById('lightbox-filename');
const lightboxBadge    = document.getElementById('lightbox-badge');
const lightboxToggle   = document.getElementById('lightbox-toggle-btn');
const lightboxDlBtn    = document.getElementById('lightbox-dl-btn');
const lightboxClose    = document.getElementById('lightbox-close');
const lightboxMeta     = document.getElementById('lightbox-meta');

let currentLightboxData = null;
let currentLightboxMode = 'annotated';


// ── THRESHOLD SLIDER ─────────────────────────────────
thresholdSldr.addEventListener('input', () => {
  const v = parseFloat(thresholdSldr.value).toFixed(2);
  thresholdDisp.textContent = v;
  const pct = ((v - 0.5) / 0.49) * 100;
  thresholdSldr.style.background =
    `linear-gradient(90deg, var(--navy-600) ${pct}%, var(--navy-100) ${pct}%)`;
});


// ── FILE SELECTION ───────────────────────────────────
uploadZone.addEventListener('click', e => {
  if (e.target === fileInput) return;
  fileInput.click();
});
uploadZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
uploadZone.addEventListener('dragover', e => {
  e.preventDefault(); uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', e => {
  if (!uploadZone.contains(e.relatedTarget)) uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  acceptFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) acceptFiles([...fileInput.files]);
});


function acceptFiles(files) {
  const ok = files.filter(f => /\.(jpe?g|png|tiff?|bmp)$/i.test(f.name));
  if (!ok.length) return;
  selectedFiles = ok;

  // Default: all checked
  checkedIndices = new Set(ok.map((_, i) => i));

  // Update zone
  uploadZone.classList.add('has-files');
  uploadTitle.textContent = `${ok.length} image${ok.length > 1 ? 's' : ''} selected`;
  uploadHint.innerHTML = 'Click to change selection';

  // Build thumbnails with checkboxes
  buildThumbnails();

  // Show toolbar & action bar
  selectToolbar.classList.add('visible');
  actionBar.classList.add('visible');
  updateSelectCount();
}


function buildThumbnails() {
  previewRow.innerHTML = '';
  const showCount = Math.min(selectedFiles.length, MAX_THUMBS);

  selectedFiles.slice(0, showCount).forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'thumb' + (checkedIndices.has(i) ? ' checked' : '');
    div.dataset.index = i;

    const img = document.createElement('img');
    const u = URL.createObjectURL(f);
    img.src = u; img.alt = f.name; img.onload = () => URL.revokeObjectURL(u);

    const lbl = document.createElement('div');
    lbl.className = 'thumb-label';
    lbl.textContent = f.name;

    const checkbox = document.createElement('div');
    checkbox.className = 'thumb-check';
    checkbox.innerHTML = checkedIndices.has(i) ? '✓' : '';

    div.appendChild(img);
    div.appendChild(lbl);
    div.appendChild(checkbox);

    // Toggle on click
    div.addEventListener('click', () => {
      if (checkedIndices.has(i)) {
        checkedIndices.delete(i);
        div.classList.remove('checked');
        checkbox.innerHTML = '';
      } else {
        checkedIndices.add(i);
        div.classList.add('checked');
        checkbox.innerHTML = '✓';
      }
      updateSelectCount();
    });

    previewRow.appendChild(div);
  });

  // Handle overflow for thumbnails beyond MAX_THUMBS
  if (selectedFiles.length > MAX_THUMBS) {
    // For files beyond visible thumbnails, they stay checked by default
    const ex = document.createElement('div');
    ex.className = 'thumb-extra';
    ex.textContent = `+${selectedFiles.length - MAX_THUMBS}`;
    previewRow.appendChild(ex);
  }
}


function updateSelectCount() {
  const total = selectedFiles.length;
  const checked = checkedIndices.size;
  selectCountEl.textContent = `${checked} of ${total} selected`;
  analyseBtn.disabled = checked === 0;

  // Update the analyse button label
  if (checked > 0) {
    analyseBtn.querySelector('span').textContent = `Analyse ${checked} Image${checked > 1 ? 's' : ''}`;
  }
}


// ── SELECT ALL / DESELECT ALL ────────────────────────
btnSelectAll.addEventListener('click', () => {
  checkedIndices = new Set(selectedFiles.map((_, i) => i));
  // Update visible thumb checkboxes
  previewRow.querySelectorAll('.thumb').forEach(div => {
    div.classList.add('checked');
    div.querySelector('.thumb-check').innerHTML = '✓';
  });
  updateSelectCount();
});

btnDeselectAll.addEventListener('click', () => {
  checkedIndices.clear();
  previewRow.querySelectorAll('.thumb').forEach(div => {
    div.classList.remove('checked');
    div.querySelector('.thumb-check').innerHTML = '';
  });
  updateSelectCount();
});


// ── DOWNLOAD HELPER ──────────────────────────────────
function downloadImage(dataUrl, filename, prefix = 'detected') {
  const a = document.createElement('a');
  a.href = dataUrl;
  const baseName = filename.replace(/\.[^/.]+$/, "");
  a.download = `${prefix}_${baseName}.jpg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}


// ── ANALYSE ──────────────────────────────────────────
analyseBtn.addEventListener('click', analyse);

async function analyse() {
  if (!checkedIndices.size) return;

  // Collect only selected files
  const filesToSend = [...checkedIndices].sort((a,b) => a-b).map(i => selectedFiles[i]);

  // UI states
  analyseBtn.disabled = true;
  loadingSection.classList.add('visible');
  loadingText.textContent = `Analysing & mapping spill coordinates for ${filesToSend.length} image${filesToSend.length > 1 ? 's' : ''}…`;
  summarySection.classList.remove('visible');
  resultsGrid.innerHTML = '';

  const form = new FormData();
  filesToSend.forEach(f => form.append('files', f));
  form.append('threshold', parseFloat(thresholdSldr.value).toFixed(2));

  try {
    const resp = await fetch(API, { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`Server ${resp.status}`);
    const data = await resp.json();
    showResults(Array.isArray(data) ? data : [data]);
  } catch (err) {
    loadingSection.classList.remove('visible');
    analyseBtn.disabled = false;
    alert('❌ ' + err.message);
  }
}


// ── RESULTS RENDERING ────────────────────────────────
function showResults(results) {
  loadingSection.classList.remove('visible');
  analyseBtn.disabled = false;
  lastResults = results;

  // Summary statistics
  const n  = results.length;
  const nO = results.filter(r => r.pred_class === 1).length;
  const nC = n - nO;
  const avg = n ? (results.reduce((s,r) => s + (r.confidence||0), 0) / n * 100).toFixed(1) + '%' : '—';

  document.getElementById('stat-total').textContent = n;
  document.getElementById('stat-oil').textContent   = nO;
  document.getElementById('stat-clean').textContent = nC;
  document.getElementById('stat-avg').textContent   = avg;

  // Update donut chart
  updateDonut(nO, nC, n);

  // Update filter counts
  document.getElementById('filter-count-all').textContent   = n;
  document.getElementById('filter-count-oil').textContent   = nO;
  document.getElementById('filter-count-clean').textContent = nC;

  // Reset filter to "all"
  setActiveFilter('all');

  summarySection.classList.add('visible');

  // Build result cards
  renderResultCards(results);
}


function renderResultCards(results) {
  resultsGrid.innerHTML = '';
  results.forEach((r, i) => resultsGrid.appendChild(buildResultCard(r, i)));

  // Animate confidence progress bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.bar-fill').forEach(el => {
        el.style.width = el.dataset.w;
      });
    }, 60);
  });
}


// ── DONUT CHART ──────────────────────────────────────
function updateDonut(nOil, nClean, total) {
  const circumference = 2 * Math.PI * 52; // r=52
  const donutOil   = document.getElementById('donut-oil');
  const donutClean = document.getElementById('donut-clean');
  const centerNum  = document.getElementById('donut-center-num');

  centerNum.textContent = total;

  if (total === 0) {
    donutOil.style.strokeDasharray   = `0 ${circumference}`;
    donutClean.style.strokeDasharray = `0 ${circumference}`;
    return;
  }

  const oilLen   = (nOil / total) * circumference;
  const cleanLen = (nClean / total) * circumference;

  donutOil.style.strokeDasharray   = `${oilLen} ${circumference - oilLen}`;
  donutOil.style.strokeDashoffset  = `${circumference * 0.25}`;  // start from top

  donutClean.style.strokeDasharray  = `${cleanLen} ${circumference - cleanLen}`;
  donutClean.style.strokeDashoffset = `${circumference * 0.25 - oilLen}`;
}


// ── FILTER BUTTONS ───────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const filter = btn.dataset.filter;
    setActiveFilter(filter);
    applyFilter(filter);
  });
});

function setActiveFilter(filter) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.querySelector(`.filter-btn[data-filter="${filter}"]`);
  if (activeBtn) activeBtn.classList.add('active');
}

function applyFilter(filter) {
  let filtered;
  if (filter === 'oil') {
    filtered = lastResults.filter(r => r.pred_class === 1);
  } else if (filter === 'clean') {
    filtered = lastResults.filter(r => r.pred_class !== 1);
  } else {
    filtered = lastResults;
  }
  renderResultCards(filtered);
}


// ── BUILD HORIZONTAL SIDE-BY-SIDE RESULT CARD ─────────
function buildResultCard(r, idx) {
  const oil  = r.pred_class === 1;
  const file = selectedFiles.find(f => f.name === r.filename) || selectedFiles[idx];
  const conf = ((r.confidence || 0) * 100).toFixed(1);
  const pO   = ((r.prob_oil   || 0) * 100).toFixed(1);
  const pC   = ((r.prob_clean || 0) * 100).toFixed(1);
  const objs = Array.isArray(r.objects) ? r.objects : [];

  // Create persistent object URL for raw file
  const rawUrl = file ? URL.createObjectURL(file) : '';
  r.raw_image_url = rawUrl;

  const card = document.createElement('div');
  card.className = `result-card ${oil ? 'is-oil' : 'is-clean'}`;
  card.style.animationDelay = `${idx * 70}ms`;

  const displaySrc = r.annotated_image || rawUrl;
  const dtStr = (objs.length > 0 && objs[0].datetime) ? objs[0].datetime : 'SAR Imagery';

  // Build Tables HTML for the right side
  let tablesHtml = '';
  if (objs.length > 0) {
    objs.forEach((obj, i) => {
      tablesHtml += `
        <div class="table-section">
          <div class="table-section-head">
            <div class="table-title">
              <span>📍 Spill Object ${i + 1} of ${objs.length}</span>
              ${obj.datetime ? `<span style="font-size:0.75rem;color:var(--text-3);font-weight:600;">(Date: ${obj.datetime})</span>` : ''}
            </div>
            <div class="pixel-summary-pill">
              Pixel Box: [X: ${obj.xmin} → ${obj.xmax}, Y: ${obj.ymin} → ${obj.ymax}]
            </div>
          </div>
          <div class="ctable-wrap">
            <table class="ctable">
              <thead>
                <tr>
                  <th>Corner Position</th>
                  <th>Pixel Coordinates</th>
                  <th>Geo Longitude</th>
                  <th>Geo Latitude</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Upper-Left (UL)</td>
                  <td>(${obj.xmin}, ${obj.ymin})</td>
                  <td>${obj.obj_ul_lon}</td>
                  <td>${obj.obj_ul_lat}</td>
                </tr>
                <tr>
                  <td>Upper-Right (UR)</td>
                  <td>(${obj.xmax}, ${obj.ymin})</td>
                  <td>${obj.obj_ur_lon}</td>
                  <td>${obj.obj_ur_lat}</td>
                </tr>
                <tr>
                  <td>Bottom-Right (BR)</td>
                  <td>(${obj.xmax}, ${obj.ymax})</td>
                  <td>${obj.obj_br_lon}</td>
                  <td>${obj.obj_br_lat}</td>
                </tr>
                <tr>
                  <td>Bottom-Left (BL)</td>
                  <td>(${obj.xmin}, ${obj.ymax})</td>
                  <td>${obj.obj_bl_lon}</td>
                  <td>${obj.obj_bl_lat}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      `;
    });
  } else {
    tablesHtml = `
      <div class="no-coords-box">
        <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <span>No anomalous oil slick coordinates recorded for this image. Image verified as Clean SAR Imagery.</span>
      </div>
    `;
  }

  card.innerHTML = `
    <!-- LEFT SIDE: IMAGE PREVIEW & ACTIONS -->
    <div class="card-left">
      <div class="card-img-wrap" title="Click to view full high-res image">
        <img class="card-img" src="${displaySrc}" alt="${esc(r.filename)}" />
        <span class="card-img-badge">${objs.length ? `📍 ${objs.length} Spill Box${objs.length > 1 ? 'es' : ''}` : 'Verified Clean'}</span>
        <div class="card-img-overlay">
          <span class="card-img-overlay-btn">
            <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7"/></svg>
            View Full Image
          </span>
        </div>
      </div>

      <div class="card-left-btns">
        <button class="btn-fullview" type="button">
          <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
          <span>Full View</span>
        </button>
        ${r.annotated_image ? `
          <button class="btn-download" type="button">
            <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
            </svg>
            <span>Download</span>
          </button>
        ` : ''}
      </div>
    </div>

    <!-- RIGHT SIDE: DETAILS & COMPREHENSIVE TABLE -->
    <div class="card-right">
      <div class="card-header-row">
        <div class="card-title-group">
          <span class="card-fname">${esc(r.filename)}</span>
          <span class="card-date-badge">${dtStr}</span>
        </div>
        <div class="card-badge-wrap">
          <span class="badge ${oil ? 'badge-oil' : 'badge-clean'}">
            ${oil ? '🛢️ Oil Spill Detected' : '✅ No Oil Spill'}
          </span>
          <span class="badge-conf">${conf}%</span>
        </div>
      </div>

      <!-- Key Metadata Chips -->
      <div class="card-chips">
        <span class="chip">⏱ Inference: <strong>${r.inference_ms || '—'} ms</strong></span>
        <span class="chip">📐 Resolution: <strong>${r.image_size || '—'}</strong></span>
        <span class="chip">🎯 Spill Objects: <strong>${objs.length} detected</strong></span>
      </div>

      <!-- Probability Bars -->
      <div class="bars">
        <div class="bar-row">
          <div class="bar-label">
            <span>🛢️ Oil Spill Probability</span>
            <span>${pO}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill oil" data-w="${pO}%"></div></div>
        </div>
        <div class="bar-row">
          <div class="bar-label">
            <span>✅ Clean Sea Probability</span>
            <span>${pC}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill clean" data-w="${pC}%"></div></div>
        </div>
      </div>

      <!-- Coordinates & Box Details Table (Direct View) -->
      ${tablesHtml}
    </div>
  `;

  // Attach button events
  const imgWrap = card.querySelector('.card-img-wrap');
  if (imgWrap) imgWrap.onclick = () => openLightbox(r);

  const fullViewBtn = card.querySelector('.btn-fullview');
  if (fullViewBtn) fullViewBtn.onclick = (e) => {
    e.stopPropagation();
    openLightbox(r);
  };

  const dlBtn = card.querySelector('.btn-download');
  if (dlBtn && r.annotated_image) {
    dlBtn.onclick = (e) => {
      e.stopPropagation();
      downloadImage(r.annotated_image, r.filename);
    };
  }

  return card;
}


// ── FULL IMAGE LIGHTBOX ──────────────────────────────
function openLightbox(r) {
  currentLightboxData = r;
  currentLightboxMode = r.annotated_image ? 'annotated' : 'raw';

  lightboxFilename.textContent = r.filename;
  const isOil = r.pred_class === 1;
  lightboxBadge.textContent = `${isOil ? '🛢️ Oil Spill' : '✅ Clean'} (${((r.confidence || 0) * 100).toFixed(1)}%)`;
  lightboxBadge.className = `lightbox-badge ${isOil ? '' : 'clean'}`;

  lightboxBackdrop.classList.add('open');
  document.body.style.overflow = 'hidden';

  requestAnimationFrame(() => updateLightboxView());
}

function updateLightboxView() {
  if (!currentLightboxData) return;
  const r = currentLightboxData;

  if (currentLightboxMode === 'annotated' && r.annotated_image) {
    lightboxImg.src = r.annotated_image;
    lightboxToggle.innerHTML = '<span>⇄ View Raw Original</span>';
    lightboxMeta.textContent = `Dimension: ${r.image_size || '—'} · Showing detected spill bounding boxes`;
  } else {
    const src = r.raw_image_url || r.annotated_image || '';
    lightboxImg.src = src;
    lightboxToggle.innerHTML = '<span>⇄ View Detected Boxes</span>';
    lightboxMeta.textContent = `Dimension: ${r.image_size || '—'} · Showing unprocessed raw SAR image`;
  }
}

lightboxToggle.addEventListener('click', () => {
  if (!currentLightboxData || !currentLightboxData.annotated_image || !currentLightboxData.raw_image_url) return;
  currentLightboxMode = (currentLightboxMode === 'annotated') ? 'raw' : 'annotated';
  updateLightboxView();
});

lightboxDlBtn.addEventListener('click', () => {
  if (!currentLightboxData) return;
  const r = currentLightboxData;
  if (currentLightboxMode === 'annotated' && r.annotated_image) {
    downloadImage(r.annotated_image, r.filename, 'detected');
  } else if (r.raw_image_url) {
    downloadImage(r.raw_image_url, r.filename, 'raw');
  }
});

function closeLightboxFn() {
  lightboxBackdrop.classList.remove('open');
  document.body.style.overflow = '';
  currentLightboxData = null;
}

lightboxClose.addEventListener('click', closeLightboxFn);
lightboxBackdrop.addEventListener('click', e => { if (e.target === lightboxBackdrop) closeLightboxFn(); });


// ── GLOBAL ESCAPE KEY ────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (lightboxBackdrop.classList.contains('open')) closeLightboxFn();
  }
});


// ── RESET ────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  selectedFiles = [];
  checkedIndices.clear();
  lastResults = [];
  fileInput.value = '';
  previewRow.innerHTML = '';
  resultsGrid.innerHTML = '';
  selectToolbar.classList.remove('visible');
  actionBar.classList.remove('visible');
  summarySection.classList.remove('visible');
  loadingSection.classList.remove('visible');
  analyseBtn.disabled = true;
  analyseBtn.querySelector('span').textContent = 'Analyse Images';
  uploadZone.classList.remove('has-files');
  uploadTitle.textContent = 'Drag & drop images here';
  uploadHint.innerHTML = 'or <u>click to browse</u> · JPG, PNG, TIFF, BMP';
  closeLightboxFn();
});


// ── UTIL ─────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
