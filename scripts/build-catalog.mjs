// Recolector del catálogo Toni Pons + índice visual.
// Se ejecuta en GitHub Actions cada 3 días (o a mano). Uso: node scripts/build-catalog.mjs
//
// 1. Descarga https://tonipons.com/products.json (API pública de Shopify), todas las páginas.
// 2. Normaliza productos (precio, precio original, SKU, color, imágenes, URL...).
// 3. Detecta altas, bajas, cambios de precio y rebajas respecto al catálogo anterior.
// 4. Calcula embeddings visuales (CLIP) SOLO de las imágenes nuevas o cambiadas.
// 5. Escribe docs/data/catalog.json, docs/data/embeddings.bin y docs/data/changes.json.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import {
  MODEL_ID, MODEL_DTYPE, EMB_DIM, IMG_SIZE,
  normalizeProduct, diffCatalogs, thumb, quantize,
} from '../docs/shared.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DATA = path.join(ROOT, 'docs', 'data');
const BASE = 'https://tonipons.com';
const IMAGES_PER_PRODUCT = Number(process.env.IMAGES_PER_PRODUCT || 3);
const UA = 'ToniPonsScan/1.0 (uso personal; catalogo semanal)';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJSON(url, tries = 4) {
  for (let i = 1; i <= tries; i++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json' } });
      if (res.status === 429 || res.status >= 500) throw new Error(`HTTP ${res.status}`);
      if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
      return await res.json();
    } catch (e) {
      if (i === tries) throw e;
      await sleep(2000 * i);
    }
  }
}

async function fetchAllProducts() {
  const all = new Map();
  for (let page = 1; page <= 100; page++) {
    const { products } = await getJSON(`${BASE}/products.json?limit=250&page=${page}`);
    if (!products || products.length === 0) break;
    let fresh = 0;
    for (const p of products) if (!all.has(p.id)) { all.set(p.id, p); fresh++; }
    console.log(`  página ${page}: ${products.length} productos (${fresh} nuevos)`);
    if (fresh === 0) break; // protección ante paginación que se repite
    await sleep(800);
  }
  return [...all.values()];
}

async function readJSON(file, fallback) {
  try { return JSON.parse(await fs.readFile(file, 'utf8')); } catch { return fallback; }
}

async function loadPrevious() {
  const catalog = await readJSON(path.join(DATA, 'catalog.json'), null);
  const cache = new Map(); // url imagen -> Int8Array
  if (!catalog) return { catalog: null, cache };
  if (catalog.model !== MODEL_ID || catalog.dim !== EMB_DIM || catalog.imgSize !== IMG_SIZE) {
    console.log('  modelo cambiado: se recalculan todos los embeddings');
    return { catalog, cache };
  }
  try {
    const buf = await fs.readFile(path.join(DATA, 'embeddings.bin'));
    const all = new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    for (const p of catalog.products) {
      (p.emb || []).forEach((row, k) => {
        const url = p.embImgs?.[k];
        if (url) cache.set(url, all.slice(row * EMB_DIM, (row + 1) * EMB_DIM));
      });
    }
  } catch { /* sin embeddings previos */ }
  return { catalog, cache };
}

let extractor = null;
async function getExtractor() {
  if (extractor) return extractor;
  const { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } = await import('@huggingface/transformers');
  env.cacheDir = path.join(ROOT, '.model-cache');
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: MODEL_DTYPE });
  extractor = { processor, model, RawImage };
  return extractor;
}

// Cuadrado blanco con el producto entero dentro (igual que hace la app con la foto).
async function loadSquare(url, RawImage) {
  const res = await fetch(thumb(url, 400), { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new Error(`imagen HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const { data, info } = await sharp(buf)
    .flatten({ background: '#ffffff' })
    .resize(IMG_SIZE, IMG_SIZE, { fit: 'contain', background: '#ffffff' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);
}

async function embedUrls(urls) {
  const { processor, model, RawImage } = await getExtractor();
  const out = new Map();
  const BATCH = 16;
  for (let i = 0; i < urls.length; i += BATCH) {
    const chunk = urls.slice(i, i + BATCH);
    const imgs = [];
    const ok = [];
    await Promise.all(chunk.map(async (u, k) => {
      try { imgs[k] = await loadSquare(u, RawImage); } catch (e) { console.warn(`  ! ${u}: ${e.message}`); }
    }));
    chunk.forEach((u, k) => { if (imgs[k]) ok.push([u, imgs[k]]); });
    if (!ok.length) continue;
    const inputs = await processor(ok.map(([, im]) => im));
    const { image_embeds } = await model(inputs);
    const data = image_embeds.data;
    ok.forEach(([u], k) => out.set(u, quantize(data.subarray(k * EMB_DIM, (k + 1) * EMB_DIM))));
    console.log(`  embeddings ${Math.min(i + BATCH, urls.length)}/${urls.length}`);
  }
  return out;
}

async function main() {
  const t0 = Date.now();
  await fs.mkdir(DATA, { recursive: true });

  console.log('1) Descargando catálogo de tonipons.com…');
  const raw = await fetchAllProducts();
  if (raw.length < 50) throw new Error(`Solo ${raw.length} productos: algo va mal, no se sobrescribe el catálogo.`);
  const products = raw
    .filter((p) => (p.images || []).length > 0)
    .map((p) => normalizeProduct(p, BASE));
  console.log(`   ${products.length} productos con imagen`);

  console.log('2) Comparando con el catálogo anterior…');
  const { catalog: prev, cache } = await loadPrevious();
  const changes = diffCatalogs(prev?.products || [], products);
  console.log(`   +${changes.added.length} nuevos, -${changes.removed.length} eliminados, ` +
    `${changes.priceChanged.length} cambios de precio, ${changes.newlyOnSale.length} nuevas rebajas`);

  console.log('3) Índice visual…');
  const wanted = [];
  for (const p of products) p.embImgs = p.images.slice(0, IMAGES_PER_PRODUCT);
  for (const p of products) for (const u of p.embImgs) if (!cache.has(u)) wanted.push(u);
  const uniqueWanted = [...new Set(wanted)];
  console.log(`   ${uniqueWanted.length} imágenes nuevas/cambiadas (reutilizadas: ${cache.size})`);
  if (uniqueWanted.length) for (const [u, v] of await embedUrls(uniqueWanted)) cache.set(u, v);

  const rows = [];
  for (const p of products) {
    p.emb = [];
    const kept = [];
    for (const u of p.embImgs) {
      const v = cache.get(u);
      if (!v) continue;
      p.emb.push(rows.length);
      kept.push(u);
      rows.push(v);
    }
    p.embImgs = kept;
  }
  const bin = new Int8Array(rows.length * EMB_DIM);
  rows.forEach((v, r) => bin.set(v, r * EMB_DIM));

  const now = new Date().toISOString();
  const catalog = {
    generated: now,
    source: BASE,
    model: MODEL_ID,
    dim: EMB_DIM,
    imgSize: IMG_SIZE,
    count: products.length,
    rows: rows.length,
    products,
  };

  const history = await readJSON(path.join(DATA, 'changes.json'), []);
  history.unshift({ date: now, total: products.length, ...changes });

  await fs.writeFile(path.join(DATA, 'embeddings.bin'), Buffer.from(bin.buffer));
  await fs.writeFile(path.join(DATA, 'catalog.json'), JSON.stringify(catalog));
  await fs.writeFile(path.join(DATA, 'changes.json'), JSON.stringify(history.slice(0, 30), null, 1));
  await fs.writeFile(path.join(DATA, 'version.json'), JSON.stringify({ generated: now, count: products.length }));
  console.log(`Hecho en ${Math.round((Date.now() - t0) / 1000)} s: ${products.length} productos, ${rows.length} imágenes indexadas.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
