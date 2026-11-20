/* Verificación geométrica: solapamientos entre textos y tablas, y límites de página */
const { slides, W, H } = require('F:/zillow_actualizado/.work/scripts/deck_openclaw.js');

function charEm(font, bold) {
  if (font === 'Consolas') return 0.55;
  return bold ? 0.545 : 0.505;
}
function occupied(sh) {
  const size = sh.size || 11;
  const perLine = Math.max(1, Math.floor((sh.w * 72) / (charEm(sh.font, sh.bold) * size)));
  let lines = 0;
  String(sh.s).split('\n').forEach((seg) => { lines += Math.max(1, Math.ceil(seg.length / perLine)); });
  const hNeed = Math.min(sh.h, (lines * size * (sh.lineSpacing || 1) * 1.2) / 72);
  return { x: sh.x, y: sh.y, w: sh.w, h: hNeed };
}
function inter(a, b) {
  const x = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const y = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return x * y;
}

const problems = [];
slides.forEach((s, i) => {
  const boxes = [];
  s.shapes.forEach((sh, k) => {
    if (sh.t === 'text') boxes.push({ id: `t${k}`, label: String(sh.s).slice(0, 40), ...occupied(sh) });
    if (sh.t === 'table') boxes.push({ id: `tb${k}`, label: 'tabla', x: sh.x, y: sh.y, w: sh.w, h: sh.rowH * sh.rows.length });
  });
  boxes.forEach((b) => {
    if (b.x < -0.01 || b.y < -0.01 || b.x + b.w > W + 0.01 || b.y + b.h > H + 0.01) {
      problems.push(`D${i + 1}: fuera de página -> ${b.label}`);
    }
  });
  for (let a = 0; a < boxes.length; a++) {
    for (let b = a + 1; b < boxes.length; b++) {
      const area = inter(boxes[a], boxes[b]);
      const minArea = Math.min(boxes[a].w * boxes[a].h, boxes[b].w * boxes[b].h);
      if (area > minArea * 0.12 && area > 0.01) {
        problems.push(`D${i + 1}: solape ${(area / minArea * 100).toFixed(0)}% entre "${boxes[a].label}" y "${boxes[b].label}"`);
      }
    }
  }
});
console.log(`Diapositivas: ${slides.length} · incidencias: ${problems.length}`);
problems.forEach((p) => console.log(' - ' + p));
