// Identificación con IA (Claude, visión) en dos pasos:
//   1) Mira tu foto y dice qué tipo de producto es, colores, material y texto visible.
//   2) Con esa pista se filtra el catálogo, se ordenan los candidatos por parecido visual
//      (índice local) y la IA compara tu foto con las fotos oficiales de esos candidatos.
// La clave de la API se guarda solo en este iPhone.

import { thumb, colorES } from './shared.js?v=7';

const API = 'https://api.anthropic.com/v1/messages';
const MODEL_FAST = 'claude-haiku-4-5-20251001';
const MODEL_MATCH = 'claude-sonnet-5-5';
const N_CANDIDATES = 30;
const KEY_NAME = 'anthropic_key';

export function getKey() {
  try { return localStorage.getItem(KEY_NAME) || ''; } catch { return ''; }
}
export function setKey(k) {
  try { k ? localStorage.setItem(KEY_NAME, k) : localStorage.removeItem(KEY_NAME); } catch {}
}

// Proveedor según la clave: "AIza…" = Google Gemini (plan gratuito), "sk-ant-…" = Anthropic Claude.
export function provider() {
  const k = getKey();
  if (!k) return null;
  if (k.startsWith('sk-ant-')) return 'claude';
  return 'gemini'; // claves de Google: "AIza…" o el formato nuevo "AQ.…"
}

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';
let geminiModels = null; // lista ordenada, el primero es el preferido

// Ordena los modelos "flash" disponibles para esta clave (Google va cambiando nombres).
export function rankGeminiModels(names) {
  const bad = /(image|tts|audio|live|embedding|native|aqa|learnlm|robotics|computer|thinking-exp|gemma)/;
  const ok = [...new Set(names.map((n) => n.replace(/^models\//, '')))]
    .filter((n) => /^gemini-/.test(n) && /flash/.test(n) && !bad.test(n));
  const ver = (n) => parseFloat((n.match(/^gemini-(\d+(?:\.\d+)?)/) || [])[1] || '0');
  const rank = (n) => ver(n) * 100 - (/preview|exp/.test(n) ? 1 : 0) - (/-\d{3,}$/.test(n) ? 0.5 : 0)
    + (/-latest$/.test(n) ? 0.2 : 0) - (/lite/.test(n) ? 150 : 0); // "lite" solo como último recurso
  return ok.sort((a, b) => rank(b) - rank(a));
}
export function pickGeminiModel(names) {
  return rankGeminiModels(names).filter((n) => !/lite/.test(n))[0] || null;
}

async function listGeminiModels() {
  const names = [];
  let token = '';
  for (let i = 0; i < 5; i++) {
    const r = await fetch(`${GEMINI_BASE}/models?pageSize=200${token ? '&pageToken=' + token : ''}`, {
      headers: { 'x-goog-api-key': getKey() },
    });
    if (!r.ok) { const e = new Error(`Gemini ${r.status}`); e.status = r.status; throw e; }
    const d = await r.json();
    for (const m of d.models || []) {
      if ((m.supportedGenerationMethods || []).includes('generateContent')) names.push(m.name);
    }
    if (!d.nextPageToken) break;
    token = d.nextPageToken;
  }
  return names;
}

async function getGeminiModels(force = false) {
  if (geminiModels && !force) return geminiModels;
  if (!force) {
    try { const c = JSON.parse(localStorage.getItem('gemini_models') || 'null'); if (c && c.k === getKey().slice(-6) && c.m?.length) return (geminiModels = c.m); } catch {}
  }
  const list = rankGeminiModels(await listGeminiModels());
  if (!list.length) throw new Error('tu clave de Gemini no tiene ningún modelo de visión disponible');
  geminiModels = list;
  try { localStorage.setItem('gemini_models', JSON.stringify({ k: getKey().slice(-6), m: list })); } catch {}
  return list;
}
async function getGeminiModel(force = false) { return (await getGeminiModels(force))[0]; }

// Comprueba que la clave funciona (consulta gratuita, sin gastar cuota de generación).
export async function ping() {
  const p = provider();
  if (!p) return { ok: false, msg: 'sin clave' };
  try {
    if (p === 'gemini') {
      try {
        const m = await getGeminiModel(true);
        return { ok: true, model: m };
      } catch (e) {
        if (e.status) return { ok: false, msg: e.status === 400 || e.status === 403 ? 'clave no válida' : `error ${e.status}` };
        if (/modelo/.test(e.message)) return { ok: false, msg: e.message };
        throw e;
      }
    }
    const r = await fetch('https://api.anthropic.com/v1/models?limit=1', {
      headers: { 'x-api-key': getKey(), 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
    });
    if (r.ok) return { ok: true };
    return { ok: false, msg: r.status === 401 ? 'clave no válida' : `error ${r.status}` };
  } catch {
    return { ok: null, msg: 'sin conexión' };
  }
}

async function toInline(part) {
  if (part.type !== 'image') return { text: part.text };
  if (part.source.type === 'base64') return { inline_data: { mime_type: part.source.media_type, data: part.source.data } };
  let r = await fetch(part.source.url).catch(() => null);
  if (!r || !r.ok) {
    // Plan B si el CDN no permite leer la imagen desde el móvil: servicio público de imágenes.
    r = await fetch(`https://wsrv.nl/?url=${encodeURIComponent(part.source.url)}&w=300&output=jpg`);
  }
  if (!r.ok) throw new Error('imagen ' + r.status);
  const blob = await r.blob();
  const data = await new Promise((ok, ko) => {
    const fr = new FileReader();
    fr.onload = () => ok(String(fr.result).split(',')[1]);
    fr.onerror = ko;
    fr.readAsDataURL(blob);
  });
  return { inline_data: { mime_type: blob.type || 'image/webp', data } };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function callGemini(content) {
  // Las imágenes del catálogo se descargan en el móvil y se envían dentro de la petición.
  const parts = [];
  const settled = await Promise.allSettled(content.map(toInline));
  settled.forEach((s) => parts.push(s.status === 'fulfilled' ? s.value : { text: '(imagen no disponible)' }));
  const body = JSON.stringify({
    contents: [{ role: 'user', parts }],
    generationConfig: { temperature: 0, responseMimeType: 'application/json' },
  });

  // Si un modelo está saturado (503/500) o sin cuota (429), reintenta y luego prueba el siguiente.
  const models = (await getGeminiModels()).slice(0, 4);
  let last = '';
  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-goog-api-key': getKey() },
        body,
      });
      if (res.ok) {
        if (model !== geminiModels[0]) geminiModels = [model, ...geminiModels.filter((x) => x !== model)]; // recordar el que va
        const data = await res.json();
        const text = (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || '').join('');
        const m = text.match(/\{[\s\S]*\}/);
        if (!m) throw new Error('respuesta de IA sin datos');
        return JSON.parse(m[0]);
      }
      let msg = ''; try { msg = (await res.json()).error.message; } catch {}
      if (res.status === 400 || res.status === 401 || res.status === 403) {
        throw new Error(/api key|key not valid/i.test(msg) ? 'clave de Gemini no válida' : `Gemini: ${msg || res.status}`);
      }
      last = `Gemini ${res.status}${msg ? ': ' + msg : ''}`;
      if (res.status === 404) break;               // ese modelo ya no existe: siguiente
      if (res.status === 429) break;               // sin cuota en ese modelo: siguiente
      if (attempt === 0) await sleep(1500);        // 500/503: un reintento y luego siguiente
    }
  }
  throw new Error(/429/.test(last) ? 'límite gratuito de Gemini alcanzado, espera un minuto' : `Gemini saturado ahora mismo, prueba en un rato (${last})`);
}

async function call(model, content, maxTokens = 500) {
  if (provider() === 'gemini') return callGemini(content);
  const res = await fetch(API, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': getKey(),
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({ model, max_tokens: maxTokens, messages: [{ role: 'user', content }] }),
  });
  if (res.status === 401) throw new Error('clave de IA no válida');
  if (res.status === 402 || res.status === 403) throw new Error('la cuenta de IA no tiene saldo o permiso');
  if (!res.ok) {
    let msg = `IA respondió ${res.status}`;
    try { msg += ': ' + (await res.json()).error.message; } catch {}
    throw new Error(msg);
  }
  const data = await res.json();
  const text = (data.content || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error('respuesta de IA sin datos');
  return JSON.parse(m[0]);
}

// Foto -> JPEG base64 de 1024 px máx.
export function photoBase64(img) {
  const s = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * s);
  c.height = Math.round(img.naturalHeight * s);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85).split(',')[1];
}

const norm = (s) => String(s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');

// ranked: salida del índice local [{p, score}] con TODOS los productos.
export async function identify(img, ranked, catalog, onStep = () => {}) {
  const photo = { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: photoBase64(img) } };

  // ---- Paso 1: describir la foto ----
  onStep('Mirando la foto…');
  const typeCount = new Map();
  for (const p of catalog.products) typeCount.set(p.type, (typeCount.get(p.type) || 0) + 1);
  const types = [...typeCount.keys()].filter(Boolean).sort();
  const desc = await call(MODEL_FAST, [
    photo,
    { type: 'text', text:
`Esta foto muestra un producto de la marca Toni Pons (calzado o complemento).
Elige de ESTA lista las categorías que encajan (1 a 3, copia el texto exacto, la más probable primero):
${types.join(' | ')}

Responde SOLO con JSON:
{"types": ["..."], "colors": ["colores principales en español"], "material": "piel, serraje, lona, yute, goma...", "features": "rasgos distintivos breves (hebillas, tiras, suela, puntera, cierre...)", "visible_text": "cualquier texto legible en el producto o etiqueta, o vacío"}` },
  ], 400);

  // ---- Filtrar y ordenar candidatos ----
  const wantTypes = new Set((desc.types || []).filter((t) => typeCount.has(t)));
  const colorWords = (desc.colors || []).map(norm).flatMap((c) => c.split(/[\s/-]+/)).filter((w) => w.length > 2);
  const textWords = norm(desc.visible_text).split(/[^a-z0-9]+/).filter((w) => w.length > 3 && w !== 'toni' && w !== 'pons');

  const scored = ranked.map(({ p, score }) => {
    let s = score;
    const hay = norm(`${colorES(p.color || '')} ${p.handle}`);
    if (colorWords.some((w) => hay.includes(w))) s += 0.03;
    const model = norm(p.model);
    if (textWords.some((w) => model.includes(w))) s += 0.5;
    const typeOk = wantTypes.size === 0 || wantTypes.has(p.type);
    return { p, score, s, typeOk };
  });
  let pool = scored.filter((x) => x.typeOk);
  if (pool.length < 5) pool = scored;
  pool.sort((a, b) => b.s - a.s);
  const cands = pool.slice(0, N_CANDIDATES);

  // ---- Paso 2: comparar con las fotos oficiales ----
  onStep(`Comparando con ${cands.length} productos…`);
  const content = [
    { type: 'text', text: 'FOTO DEL CLIENTE (producto a identificar):' },
    photo,
    { type: 'text', text: `Pista: ${desc.material || ''}. ${desc.features || ''}. Colores: ${(desc.colors || []).join(', ')}.\nCANDIDATOS del catálogo oficial Toni Pons (foto de estudio):` },
  ];
  cands.forEach((c, i) => {
    const p = c.p;
    content.push({ type: 'text', text: `#${i}: ${p.model} · ${colorES(p.color || '').toLowerCase()} · ${p.type}${p.name ? ' · ' + p.name : ''}` });
    content.push({ type: 'image', source: { type: 'url', url: thumb(p.images[0], 300) } });
  });
  content.push({ type: 'text', text:
`¿Cuál de los candidatos es EXACTAMENTE el mismo producto (mismo modelo y mismo color) que la foto del cliente?
Fíjate en forma, suela, tiras, hebillas, costuras, material y color. La foto del cliente puede tener otro ángulo, luz y fondo.
Si el mismo modelo aparece en varios colores, elige el color que coincida.
Si ninguno es claramente el mismo producto, pon "match": null.
Responde SOLO con JSON:
{"match": número o null, "confidence": 0-100, "alternatives": [hasta 4 números, los siguientes más probables], "reason": "frase corta en español"}` });

  const res = await call(MODEL_MATCH, content, 300);
  const pick = (i) => (Number.isInteger(i) && cands[i] ? cands[i] : null);
  const match = pick(res.match);
  const alts = (res.alternatives || []).map(pick).filter(Boolean);
  const list = [];
  for (const c of [match, ...alts, ...cands]) if (c && !list.includes(c)) list.push(c);
  return {
    match,
    confidence: Number(res.confidence) || 0,
    reason: res.reason || '',
    list: list.map((c) => ({ p: c.p, score: c.score })),
  };
}
