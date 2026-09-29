// Lógica compartida entre el recolector (Node) y la app (navegador).
// Sin dependencias: se puede importar en ambos entornos.

export const MODEL_ID = 'Xenova/clip-vit-base-patch32';
export const MODEL_DTYPE = 'q8';
export const EMB_DIM = 512;
export const IMG_SIZE = 256; // lado del cuadrado al que se normaliza cada imagen antes del modelo

// Umbrales de confianza (similitud coseno entre embeddings de imagen).
export const CONF = {
  sure: 0.90,        // por encima: resultado directo
  good: 0.85,        // resultado directo si además hay margen suficiente
  margin: 0.02,      // ventaja mínima sobre el segundo producto
};

const r2 = (x) => Math.round((x + Number.EPSILON) * 100) / 100;

// Los SKU de Toni Pons llevan el color en catalán (ENZA_BLAU_25).
const COLORS = {
  NEGRE: 'NEGRO', BLANC: 'BLANCO', BLAU: 'AZUL', MARINO: 'MARINO', VERMELL: 'ROJO', GROC: 'AMARILLO',
  VERD: 'VERDE', MARRO: 'MARRÓN', CUIRO: 'CUERO', CAMEL: 'CAMEL', GRIS: 'GRIS', ROSA: 'ROSA',
  TAUPE: 'TAUPE', BEIX: 'BEIGE', CRU: 'CRUDO', OR: 'ORO', PLATA: 'PLATA', LILA: 'LILA',
  TARONJA: 'NARANJA', MOSTASSA: 'MOSTAZA', BORDEUS: 'BURDEOS', XOCOLATA: 'CHOCOLATE',
  MARI: 'MARINO', PLATI: 'PLATINO', VI: 'VINO', TEXA: 'TEJANO', TORRAT: 'TOSTADO', PEDRA: 'PIEDRA',
  BRU: 'MARRÓN OSCURO', MORAT: 'MORADO', TEULA: 'TEJA', CEL: 'CELESTE', PLOM: 'PLOMO', OCEA: 'OCÉANO',
  TABAC: 'TABACO', GRANA: 'GRANATE', ROIG: 'ROJO', SALMO: 'SALMÓN', MARINER: 'MARINERO', SAND: 'ARENA',
};
export function colorES(c) {
  return String(c).split(' ').map((w) => COLORS[w] || w).join(' ');
}

export function discounts(price) {
  return {
    d10: r2(price * 0.90),
    d20: r2(price * 0.80),
    d30: r2(price * 0.70),
  };
}

export function formatEUR(n) {
  return n.toFixed(2).replace('.', ',').replace(/\B(?=(\d{3})+(?!\d),)/g, '.') + ' €';
}

// Convierte un producto del products.json de Shopify al formato compacto del catálogo.
export function normalizeProduct(p, baseUrl = 'https://tonipons.com') {
  const variants = p.variants || [];
  const avail = variants.filter((v) => v.available);
  const ref = avail[0] || variants[0] || {};
  const price = Number(ref.price) || 0;
  const cmp = Number(ref.compare_at_price) || 0;
  const onSale = cmp > price + 0.001;

  const [namePart, modelPart] = String(p.title || '').split('|').map((s) => s.trim());
  const model = modelPart || namePart || p.handle;

  // SKU base: ENZA_OCRE_25 -> ENZA_OCRE (quitamos la talla final)
  const sku = ref.sku ? String(ref.sku).replace(/_[^_]*$/, '') : null;
  const skuParts = sku ? sku.split('_') : [];
  const color = skuParts.length > 1 ? colorES(skuParts.slice(1).join(' ')) : null;

  const images = (p.images || [])
    .slice()
    .sort((a, b) => (a.position || 0) - (b.position || 0))
    .map((i) => i.src)
    .filter(Boolean);

  const text = String(p.body_html || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 300);

  return {
    id: p.id,
    handle: p.handle,
    model,
    name: modelPart ? namePart : '',
    type: p.product_type || '',
    tags: p.tags || [],
    url: `${baseUrl}/products/${p.handle}`,
    price,
    compareAt: onSale ? cmp : null,
    onSale,
    available: avail.length > 0,
    sku,
    color,
    sizes: avail.map((v) => v.option1).filter(Boolean),
    images: images.slice(0, 6),
    desc: text,
  };
}

// Compara el catálogo anterior con el nuevo.
export function diffCatalogs(oldList, newList) {
  const oldMap = new Map((oldList || []).map((p) => [p.id, p]));
  const newMap = new Map(newList.map((p) => [p.id, p]));
  const added = [];
  const removed = [];
  const priceChanged = [];
  const newlyOnSale = [];
  for (const p of newList) {
    const o = oldMap.get(p.id);
    if (!o) { added.push({ id: p.id, model: p.model, price: p.price }); continue; }
    if (o.price !== p.price) priceChanged.push({ id: p.id, model: p.model, from: o.price, to: p.price });
    if (p.onSale && !o.onSale) newlyOnSale.push({ id: p.id, model: p.model, price: p.price, compareAt: p.compareAt });
  }
  for (const o of oldList || []) if (!newMap.has(o.id)) removed.push({ id: o.id, model: o.model });
  return { added, removed, priceChanged, newlyOnSale };
}

// Imagen sin el parámetro ?v= para usarla como clave estable... pero conservando la versión
// para detectar cambios: la clave es la URL completa.
export function thumb(src, width = 400) {
  const u = new URL(src);
  u.searchParams.set('width', String(width));
  return u.toString();
}

// Cuantización int8 por vector (la escala se descarta: al comparar se renormaliza).
export function quantize(vec) {
  let m = 0;
  for (const x of vec) m = Math.max(m, Math.abs(x));
  const q = new Int8Array(vec.length);
  if (m === 0) return q;
  for (let i = 0; i < vec.length; i++) q[i] = Math.round((vec[i] / m) * 127);
  return q;
}

// Carga embeddings int8 en una matriz Float32 normalizada (fila por imagen).
export function dequantizeAll(int8, dim = EMB_DIM) {
  const n = int8.length / dim;
  const out = new Float32Array(int8.length);
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let i = 0; i < dim; i++) { const v = int8[r * dim + i]; s += v * v; }
    const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
    for (let i = 0; i < dim; i++) out[r * dim + i] = int8[r * dim + i] * inv;
  }
  return out;
}

export function normalize(vec) {
  let s = 0;
  for (const x of vec) s += x * x;
  const inv = s > 0 ? 1 / Math.sqrt(s) : 0;
  return Float32Array.from(vec, (x) => x * inv);
}

// Puntúa cada producto: máximo sobre (vectores de la foto) × (imágenes del producto).
// products[i].emb = [filas en la matriz]
export function rankProducts(queryVecs, matrix, products, dim = EMB_DIM) {
  const rowScore = new Float32Array(matrix.length / dim);
  rowScore.fill(-1);
  for (const q of queryVecs) {
    for (let r = 0; r < rowScore.length; r++) {
      let s = 0;
      const o = r * dim;
      for (let i = 0; i < dim; i++) s += q[i] * matrix[o + i];
      if (s > rowScore[r]) rowScore[r] = s;
    }
  }
  const scored = [];
  for (const p of products) {
    if (!p.emb || !p.emb.length) continue;
    let best = -1;
    for (const r of p.emb) if (rowScore[r] > best) best = rowScore[r];
    scored.push({ p, score: best });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

export function decide(ranked) {
  if (!ranked.length) return { confident: false };
  const [a, b] = ranked;
  const margin = b ? a.score - b.score : 1;
  const confident = a.score >= CONF.sure || (a.score >= CONF.good && margin >= CONF.margin);
  return { confident, margin };
}

// Similitud -> porcentaje orientativo para mostrar.
export function toPercent(score) {
  const t = (score - 0.60) / (0.95 - 0.60);
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}
