# TP Scan

App personal para el iPhone: haces una foto a un producto Toni Pons y te dice qué modelo es, su precio actual y el precio con 10 %, 20 % y 30 % de descuento.

## Cómo funciona

- **App**: web instalable (carpeta `docs/`). Se añade a la pantalla de inicio del iPhone desde Safari y se abre a pantalla completa.
- **Reconocimiento**: modelo de visión CLIP (`Xenova/clip-vit-base-patch32`) ejecutado dentro del iPhone. La foto nunca sale del teléfono. Coste por foto: 0 €.
- **Catálogo**: `scripts/build-catalog.mjs` lee `https://tonipons.com/products.json` (API pública de Shopify), detecta altas, bajas, cambios de precio y rebajas, y solo calcula el índice visual de las imágenes nuevas.
- **Automatización**: `.github/workflows/update-catalog.yml` lo ejecuta cada 3 días en GitHub (gratis) y publica la app en GitHub Pages. El PC puede estar apagado.

## Archivos generados (`docs/data/`)

- `catalog.json` — productos (modelo, nombre, tipo, color, SKU, precio, precio original, tallas, imágenes, URL)
- `embeddings.bin` — índice visual (int8, 512 valores por imagen)
- `changes.json` — historial de altas, bajas, precios y rebajas
- `version.json` — fecha de la última actualización

## Ajustes

- Umbrales de confianza: `CONF` en `docs/shared.js`.
- Logo: coloca tu propio archivo `docs/logo.png` y la app lo mostrará en lugar del texto.
- Forzar actualización desde el móvil: botón ACTUALIZAR CATÁLOGO. Para que además lance el rastreo en la nube, la primera vez pide una clave de GitHub (fine-grained, solo este repositorio, permiso *Actions: read and write*).

## Pruebas

`npm test`
