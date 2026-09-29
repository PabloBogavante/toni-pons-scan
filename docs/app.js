import {
  MODEL_ID, MODEL_DTYPE, EMB_DIM, IMG_SIZE,
  discounts, formatEUR, dequantizeAll, normalize, rankProducts, decide, toPercent, thumb, colorES,
} from './shared.js';
import * as AI from './ai.js';

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const $ = (id) => document.getElementById(id);

const state = {
  catalog: null,     // {generated, products, ...}
  matrix: null,      // Float32Array filas normalizadas
  model: null,       // {processor, model, RawImage}
  modelLoading: null,
  last: null,        // último análisis {ranked, photoUrl}
};

// ---------- Navegación ----------
function show(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id));
  const el = $(id);
  if (el.classList.contains('scroll')) el.scrollTop = 0;
}
let toastTimer;
function toast(msg, ms = 3200) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

// ---------- Logo opcional (docs/logo.png) ----------
document.querySelectorAll('.brand').forEach((b) => {
  const img = b.querySelector('.logo');
  const ok = () => { img.hidden = false; b.classList.add('has-logo'); };
  if (img.complete) { if (img.naturalWidth > 0) ok(); else img.remove(); return; }
  img.addEventListener('load', ok);
  img.addEventListener('error', () => img.remove());
});

// ---------- Catálogo ----------
async function loadCatalog({ fresh = false } = {}) {
  const opt = fresh ? { cache: 'reload' } : {};
  const [cat, bin] = await Promise.all([
    fetch('data/catalog.json', opt).then((r) => { if (!r.ok) throw new Error('catálogo no disponible'); return r.json(); }),
    fetch('data/embeddings.bin', opt).then((r) => { if (!r.ok) throw new Error('índice no disponible'); return r.arrayBuffer(); }),
  ]);
  const int8 = new Int8Array(bin);
  if (int8.length !== cat.rows * EMB_DIM) throw new Error('índice visual incompleto');
  state.catalog = cat;
  state.matrix = dequantizeAll(int8, EMB_DIM);
  renderCatalogInfo();
}

function renderCatalogInfo() {
  const c = state.catalog;
  if (!c) { $('catalogInfo').textContent = 'Catálogo aún no disponible'; return; }
  const d = new Date(c.generated);
  const f = d.toLocaleDateString('es-ES', { day: '2-digit', month: '2-digit' }) + ' ' +
            d.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' });
  $('catalogInfo').textContent = `${c.count} productos · actualizado ${f}`;
}

// ---------- Modelo de visión (se ejecuta en el iPhone) ----------
function loadModel() {
  if (state.model) return Promise.resolve(state.model);
  if (state.modelLoading) return state.modelLoading;
  state.modelLoading = (async () => {
    const { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    const files = {};
    const progress_callback = (p) => {
      if (p.status === 'progress' && p.total) {
        files[p.file] = [p.loaded, p.total];
        let l = 0, t = 0;
        for (const [a, b] of Object.values(files)) { l += a; t += b; }
        if (t > 5e6) setStatus(`Preparando reconocimiento (solo la primera vez)… ${Math.round((l / t) * 100)} %`);
      }
    };
    const processor = await AutoProcessor.from_pretrained(MODEL_ID);
    const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: MODEL_DTYPE, progress_callback });
    state.model = { processor, model, RawImage };
    setStatus('');
    return state.model;
  })();
  state.modelLoading.catch(() => { state.modelLoading = null; });
  return state.modelLoading;
}
function setStatus(t) {
  $('modelStatus').textContent = t;
  if (t && $('busy').classList.contains('active')) $('busyText').textContent = t;
}

// ---------- Foto -> recortes -> embeddings ----------
function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => reject(new Error('No se pudo leer la foto'));
    img.src = url;
  });
}

// Tres vistas de la foto: entera (con márgenes blancos, como las fotos del catálogo),
// el centro cuadrado y un zoom al centro (por si solo sale una parte del producto).
function makeCrops(img) {
  const W = img.naturalWidth, H = img.naturalHeight;
  const m = Math.min(W, H);
  const views = [
    { sx: 0, sy: 0, sw: W, sh: H, contain: true },
    { sx: (W - m) / 2, sy: (H - m) / 2, sw: m, sh: m },
    { sx: (W - m * 0.62) / 2, sy: (H - m * 0.62) / 2, sw: m * 0.62, sh: m * 0.62 },
  ];
  return views.map((v) => {
    const c = document.createElement('canvas');
    c.width = c.height = IMG_SIZE;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
    if (v.contain) {
      const s = IMG_SIZE / Math.max(v.sw, v.sh);
      const dw = v.sw * s, dh = v.sh * s;
      ctx.drawImage(img, v.sx, v.sy, v.sw, v.sh, (IMG_SIZE - dw) / 2, (IMG_SIZE - dh) / 2, dw, dh);
    } else {
      ctx.drawImage(img, v.sx, v.sy, v.sw, v.sh, 0, 0, IMG_SIZE, IMG_SIZE);
    }
    return ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  });
}

async function embedCrops(crops) {
  const { processor, model, RawImage } = await loadModel();
  const images = crops.map((d) => new RawImage(d.data, d.width, d.height, 4).rgb());
  const inputs = await processor(images);
  const { image_embeds } = await model(inputs);
  const out = [];
  for (let k = 0; k < images.length; k++) out.push(normalize(image_embeds.data.subarray(k * EMB_DIM, (k + 1) * EMB_DIM)));
  return out;
}

async function analyze(file) {
  if (!state.catalog) {
    try { await loadCatalog(); } catch { toast('El catálogo aún no está disponible. Prueba ACTUALIZAR CATÁLOGO.'); return; }
  }
  const { img, url } = await loadImage(file);
  $('busyPhoto').src = url;
  $('busyText').textContent = state.model ? 'Analizando…' : 'Preparando reconocimiento…';
  show('busy');
  try {
    const t0 = performance.now();
    if (!state.model) {
      await loadModel();
      $('busyText').textContent = 'Analizando…';
    }
    const vecs = await embedCrops(makeCrops(img));
    const ranked = rankProducts(vecs, state.matrix, state.catalog.products, EMB_DIM);
    console.log(`análisis ${Math.round(performance.now() - t0)} ms`, ranked.slice(0, 5).map((r) => [r.p.model, r.p.color, r.score.toFixed(3)]));

    if (AI.provider()) {
      try {
        const r = await AI.identify(img, ranked, state.catalog, (t) => ($('busyText').textContent = t));
        console.log('IA', r.match?.p.model, r.confidence, r.reason);
        state.last = { ranked: r.list, photoUrl: url, reason: r.reason };
        if (r.match && r.confidence >= 70) showResult(r.match, { ai: r.confidence });
        else showCandidates();
        return;
      } catch (e) {
        console.error(e);
        toast('IA no disponible (' + e.message + '). Resultado sin IA.', 4500);
      }
    }
    state.last = { ranked, photoUrl: url };
    const { confident } = decide(ranked);
    if (confident) showResult(ranked[0], { auto: true });
    else showCandidates();
  } catch (e) {
    console.error(e);
    show('home');
    toast('No se pudo analizar: ' + (e.message || e));
  }
}

// ---------- Pantallas de resultado ----------
function productLine(p) {
  return [p.name, p.color && `Color: ${colorES(p.color).toLowerCase()}`].filter(Boolean).join(' · ');
}

function showResult(item, { auto = false, ai = 0 } = {}) {
  const p = item.p;
  const dsc = discounts(p.price);
  $('rImg').src = thumb(p.images[0], 900);
  $('rMine').src = state.last?.photoUrl || '';
  $('rMine').hidden = !state.last?.photoUrl;
  $('rModel').textContent = p.model;
  $('rSub').textContent = [p.type, productLine(p)].filter(Boolean).join(' · ');
  $('rConf').textContent = ai ? `Identificado con IA · confianza ${ai} %`
    : auto ? `Coincidencia ${toPercent(item.score)} %` : 'Confirmado por ti';
  $('rPrice').textContent = formatEUR(p.price);
  const sale = $('rSale');
  if (p.onSale && p.compareAt) {
    sale.hidden = false;
    sale.innerHTML = `Rebajado · precio original <s>${formatEUR(p.compareAt)}</s>. Los descuentos se calculan sobre el precio actual.`;
  } else sale.hidden = true;
  $('rD10').textContent = formatEUR(dsc.d10);
  $('rD20').textContent = formatEUR(dsc.d20);
  $('rD30').textContent = formatEUR(dsc.d30);
  $('rLink').href = p.url;
  $('rUrl').textContent = p.url + (p.available ? '' : ' · (agotado en la web)');
  $('rOthers').hidden = !state.last;
  show('result');
}

function showCandidates() {
  const list = $('candList');
  list.innerHTML = '';
  $('candNote').textContent = state.last.reason || '';
  for (const item of state.last.ranked.slice(0, 8)) {
    const p = item.p;
    const li = document.createElement('li');
    li.innerHTML = `
      <img src="${thumb(p.images[0], 300)}" alt="" loading="lazy">
      <div class="t"><div class="m"></div><div class="d"></div></div>
      <div class="p"><div class="pr"></div></div>`;
    li.querySelector('.m').textContent = p.model;
    li.querySelector('.d').textContent = [p.color && colorES(p.color).toLowerCase(), p.type].filter(Boolean).join(' · ');
    li.querySelector('.pr').textContent = formatEUR(p.price);
    li.addEventListener('click', () => showResult(item, { auto: false }));
    list.appendChild(li);
  }
  show('cands');
}

// ---------- Actualizar catálogo ----------
// Descarga el último catálogo publicado. Si has guardado una clave de GitHub (una sola vez),
// además lanza el rastreo de tonipons.com en la nube y espera a que termine.
function repoFromLocation() {
  const host = location.hostname;
  if (!host.endsWith('.github.io')) return null;
  const owner = host.replace('.github.io', '');
  const repo = location.pathname.split('/').filter(Boolean)[0] || `${owner}.github.io`;
  return { owner, repo };
}

async function triggerCloudUpdate() {
  const r = repoFromLocation();
  if (!r) return false;
  let token = null;
  try { token = localStorage.getItem('gh_token'); } catch {}
  if (!token) {
    const t = prompt('Para forzar un rastreo nuevo de tonipons.com pega tu clave de GitHub (solo se pide una vez).\n\nDeja vacío para solo descargar el último catálogo.');
    if (!t) return false;
    token = t.trim();
    try { localStorage.setItem('gh_token', token); } catch {}
  }
  const res = await fetch(`https://api.github.com/repos/${r.owner}/${r.repo}/actions/workflows/update-catalog.yml/dispatches`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' },
    body: JSON.stringify({ ref: 'main' }),
  });
  if (res.status === 401 || res.status === 403 || res.status === 404) {
    try { localStorage.removeItem('gh_token'); } catch {}
    throw new Error('clave de GitHub no válida');
  }
  if (!res.ok) throw new Error(`GitHub respondió ${res.status}`);
  return true;
}

async function waitForNewVersion(since, minutes = 20) {
  const end = Date.now() + minutes * 60e3;
  while (Date.now() < end) {
    await new Promise((r) => setTimeout(r, 30e3));
    try {
      const v = await fetch('data/version.json', { cache: 'reload' }).then((r) => r.json());
      if (v.generated !== since) return true;
    } catch {}
  }
  return false;
}

let updating = false;
$('updateBtn').addEventListener('click', async () => {
  if (updating) return;
  updating = true;
  const btn = $('updateBtn');
  btn.textContent = 'ACTUALIZANDO…';
  try {
    const before = state.catalog?.generated;
    await loadCatalog({ fresh: true }).catch(() => {});
    let launched = false;
    try { launched = await triggerCloudUpdate(); } catch (e) { toast('No se pudo lanzar el rastreo: ' + e.message); }
    if (launched) {
      toast('Rastreando tonipons.com… tarda unos minutos. Puedes seguir usando la app.', 5000);
      const done = await waitForNewVersion(state.catalog?.generated || before);
      if (done) { await loadCatalog({ fresh: true }); toast('Catálogo actualizado'); }
      else toast('El rastreo sigue en marcha; vuelve a pulsar más tarde.');
    } else {
      toast(state.catalog ? 'Catálogo descargado' : 'Catálogo aún no disponible');
    }
  } finally {
    btn.textContent = 'ACTUALIZAR CATÁLOGO';
    updating = false;
  }
});

// ---------- IA ----------
function renderAiBtn() {
  const p = AI.provider();
  $('aiBtn').textContent = p === 'gemini' ? 'IA: GEMINI' : p === 'claude' ? 'IA: CLAUDE' : 'IA: DESACTIVADA';
}
$('aiBtn').addEventListener('click', () => {
  const has = AI.getKey();
  const t = prompt(has
    ? 'IA activada. Pega una clave nueva para cambiarla, o escribe BORRAR para desactivarla.'
    : 'Pega tu clave de IA. Gratis: clave de Google AI Studio (empieza por AQ. o AIza). De pago: clave de Anthropic (sk-ant-). Se guarda solo en este iPhone.');
  if (t === null) return;
  const v = t.trim();
  if (/^borrar$/i.test(v)) { AI.setKey(''); toast('IA desactivada'); }
  else if (/^(AIza|AQ\.|sk-ant-)/.test(v) && v.length > 20 && !/\s/.test(v)) { AI.setKey(v); toast('IA activada'); }
  else if (v) toast('Esa clave no parece válida (debe empezar por AQ., AIza o sk-ant-)');
  renderAiBtn();
});
renderAiBtn();

// ---------- Eventos ----------
$('camera').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  e.target.value = '';
  if (f) analyze(f);
});
document.querySelectorAll('.again').forEach((b) => b.addEventListener('click', () => show('home')));
$('rOthers').addEventListener('click', showCandidates);

// ---------- Arranque ----------
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

(async () => {
  const splashMin = new Promise((r) => setTimeout(r, 700));
  await Promise.race([loadCatalog().catch(() => renderCatalogInfo()), new Promise((r) => setTimeout(r, 2500))]);
  await splashMin;
  show('home');
  if (!state.catalog) renderCatalogInfo();
  // El modelo se prepara en segundo plano para que SCAN sea inmediato.
  loadModel().catch((e) => setStatus('Sin conexión para preparar el reconocimiento'));
})();

// Acceso para pruebas desde la consola.
window.__tp = { state, showResult, showCandidates, show };
