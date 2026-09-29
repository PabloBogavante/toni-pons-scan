// Guarda la app para que abra al instante y funcione sin cobertura.
const SHELL = 'tp-shell-v1';
const ASSETS = ['./', 'index.html', 'style.css', 'app.js', 'shared.js', 'manifest.webmanifest', 'icons/icon-180.png', 'icons/icon-192.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k.startsWith('tp-shell-') && k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Librería de visión desde el CDN: primero caché.
  if (url.hostname === 'cdn.jsdelivr.net') {
    e.respondWith(caches.open('tp-lib').then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    }));
    return;
  }
  if (url.origin !== location.origin) return; // imágenes de Shopify, modelo, GitHub: sin tocar

  // Datos del catálogo y la propia app: primero red, si no hay, caché.
  e.respondWith((async () => {
    const c = await caches.open(url.pathname.includes('/data/') ? 'tp-data' : SHELL);
    try {
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    } catch {
      const hit = await c.match(req, { ignoreSearch: true });
      if (hit) return hit;
      throw new Error('sin conexión');
    }
  })());
});
