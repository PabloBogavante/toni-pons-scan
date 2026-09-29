import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as S from '../docs/shared.js';

const raw = JSON.parse(fs.readFileSync(new URL('./fixture.json', import.meta.url))).products;

test('normaliza un producto real de Shopify', () => {
  const p = S.normalizeProduct(raw[0]);
  assert.equal(p.model, 'ENZA');
  assert.equal(p.name, 'Zapatilla deportiva infantil en piel con velcro');
  assert.equal(p.type, 'Zapatos infantiles');
  assert.equal(p.price, 39.95);
  assert.equal(p.onSale, false);
  assert.equal(p.compareAt, null);
  assert.equal(p.sku, 'ENZA_OCRE');
  assert.equal(p.color, 'OCRE');
  assert.equal(S.colorES('NEGRE'), 'NEGRO');
  assert.deepEqual(p.sizes, ['26']);
  assert.equal(p.url, 'https://tonipons.com/products/zapatos-nino-enza-ocre');
  assert.match(p.images[0], /ENZA_OCRE__1/); // ordenadas por posición
  assert.match(p.desc, /^ENZA\. Estas sneakers/);
});

test('detecta rebaja y precio original', () => {
  const p = S.normalizeProduct(raw[1]);
  assert.equal(p.onSale, true);
  assert.equal(p.price, 55.96);
  assert.equal(p.compareAt, 79.95);
});

test('descuentos redondeados a 2 decimales', () => {
  assert.deepEqual(S.discounts(39.95), { d10: 35.96, d20: 31.96, d30: 27.97 });
  assert.deepEqual(S.discounts(55.96), { d10: 50.36, d20: 44.77, d30: 39.17 });
  assert.deepEqual(S.discounts(100), { d10: 90, d20: 80, d30: 70 });
  assert.deepEqual(S.discounts(1.05), { d10: 0.95, d20: 0.84, d30: 0.74 });
});

test('formato en euros', () => {
  assert.equal(S.formatEUR(39.95), '39,95 €');
  assert.equal(S.formatEUR(27.97), '27,97 €');
  assert.equal(S.formatEUR(1234.5), '1.234,50 €');
  assert.equal(S.formatEUR(90), '90,00 €');
});

test('diff: altas, bajas, precio y rebajas', () => {
  const a = raw.map((r) => S.normalizeProduct(r));
  const oldList = [
    { ...a[1], price: 79.95, onSale: false, compareAt: null },
    { id: 999, model: 'VIEJO', price: 10 },
  ];
  const d = S.diffCatalogs(oldList, a);
  assert.deepEqual(d.added.map((x) => x.model), ['ENZA']);
  assert.deepEqual(d.removed.map((x) => x.model), ['VIEJO']);
  assert.deepEqual(d.priceChanged, [{ id: 222, model: 'VIC', from: 79.95, to: 55.96 }]);
  assert.equal(d.newlyOnSale.length, 1);
});

test('cuantización y búsqueda eligen el producto correcto', () => {
  const dim = 8;
  const rnd = (seed) => { let x = seed; return () => ((x = (x * 16807) % 2147483647) / 2147483647) - 0.5; };
  const r = rnd(7);
  const base = Array.from({ length: 4 }, () => Float32Array.from({ length: dim }, r));
  const int8 = new Int8Array(4 * dim);
  base.forEach((v, i) => int8.set(S.quantize(v), i * dim));
  const M = S.dequantizeAll(int8, dim);
  const products = [{ id: 'a', emb: [0, 1] }, { id: 'b', emb: [2] }, { id: 'c', emb: [3] }];
  const noisy = S.normalize(base[2].map((x, i) => x + (i % 2 ? 0.02 : -0.02)));
  const ranked = S.rankProducts([noisy], M, products, dim);
  assert.equal(ranked[0].p.id, 'b');
  assert.ok(ranked[0].score > 0.95);
  assert.equal(S.decide(ranked).confident, ranked[0].score - ranked[1].score >= S.CONF.margin);
});

test('miniaturas de Shopify', () => {
  assert.equal(S.thumb('https://cdn.shopify.com/a.webp?v=1', 400), 'https://cdn.shopify.com/a.webp?v=1&width=400');
});
