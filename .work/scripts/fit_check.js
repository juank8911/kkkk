/* Validador de ajuste de texto para la especificación de la presentación */
const path = require('path');

// Modelo de ancho medio por carácter (em) para Segoe UI / Consolas
function charEm(font, bold) {
  if (font === 'Consolas') return 0.55;
  return bold ? 0.545 : 0.505;
}

function wrapLines(str, boxWpt, sizePt, font, bold) {
  const perLine = Math.max(1, Math.floor(boxWpt / (charEm(font, bold) * sizePt)));
  let lines = 0;
  String(str).split('\n').forEach((seg) => {
    lines += Math.max(1, Math.ceil(seg.length / perLine));
  });
  return { lines, perLine };
}

function check(slides) {
  const issues = [];
  slides.forEach((s, i) => {
    s.shapes.forEach((sh) => {
      if (sh.t === 'text') {
        const size = sh.size || 11;
        const { lines, perLine } = wrapLines(sh.s, sh.w * 72, size, sh.font, sh.bold);
        const needed = (lines * size * (sh.lineSpacing || 1.0) * 1.2) / 72;
        const ratio = needed / sh.h;
        if (ratio > 0.98) {
          issues.push({
            slide: i + 1, kind: 'text', ratio: +ratio.toFixed(2), need: +needed.toFixed(2), box: sh.h,
            chars: String(sh.s).length, perLine,
            sample: String(sh.s).slice(0, 60)
          });
        }
      }
      if (sh.t === 'table') {
        sh.rows.forEach((row, ri) => {
          row.forEach((cell, ci) => {
            const o = cell.options || {};
            const size = o.size || 10;
            const { lines, perLine } = wrapLines(cell.text, sh.colW[ci] * 72 - 12, size, o.font, o.bold);
            if (lines > 1) {
              issues.push({
                slide: i + 1, kind: 'cell', ratio: lines, need: lines, box: sh.rowH,
                chars: String(cell.text).length, perLine,
                sample: `fila ${ri} col ${ci}: ${cell.text}`
              });
            }
          });
        });
      }
    });
  });
  return issues;
}

module.exports = { check };

if (require.main === module) {
  const { slides } = require('F:/zillow_actualizado/.work/scripts/deck_openclaw.js');
  const issues = check(slides);
  console.log(`Avisos: ${issues.length}`);
  issues.forEach((it) => console.log(JSON.stringify(it)));
}
