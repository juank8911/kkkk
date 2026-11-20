/* OpenClaw — generador de presentación (PPTX editable + HTML de control) */
const fs = require('fs');
const path = require('path');
const PptxGenJS = require('F:/zillow_actualizado/.work/tools/slides/node_modules/pptxgenjs');

const OUT_DIR = 'F:/zillow_actualizado/OpenCalw_Zillow/openClaw';
const PREVIEW_DIR = 'F:/zillow_actualizado/.work/previews/slides';

const W = 13.333, H = 7.5;
const C = {
  navy: '0F1E33', ink: '16233A', muted: '5D6E85', faint: '93A3B8',
  line: 'DEE6EF', white: 'FFFFFF', soft: 'F4F7FB',
  teal: '12796F', tealSoft: 'E4F2EF',
  amber: 'A9701A', amberSoft: 'FAEFD9',
  red: 'AE3A2C', redSoft: 'FAE9E6',
  blue: '2B5FD9', blueSoft: 'E8EEFB',
  paper: 'FBFCFE'
};
const FONT = 'Segoe UI';

/* ---------- helpers de geometría ---------- */
const rect = (o) => ({ t: 'rect', ...o });
const text = (o) => ({ t: 'text', ...o });
const arrow = (o) => ({ t: 'arrow', ...o });
const dot = (o) => ({ t: 'dot', ...o });
const table = (o) => ({ t: 'table', ...o });

function footer(n) {
  return [
    text({ x: 0.75, y: 7.02, w: 7, h: 0.24, s: `OpenClaw · Revisión técnica · Noviembre 2026`, size: 8, color: C.faint }),
    text({ x: 11.4, y: 7.02, w: 1.18, h: 0.24, s: `${n}`, size: 8.5, color: C.faint, align: 'right' })
  ];
}

function header(kicker, title, n, subtitle) {
  const out = [
    rect({ x: 0, y: 0, w: W, h: H, fill: C.white }),
    rect({ x: 0, y: 0, w: 0.16, h: H, fill: C.navy }),
    text({ x: 0.75, y: 0.52, w: 11.8, h: 0.26, s: kicker.toUpperCase(), size: 9, bold: true, color: C.blue, charSpacing: 1.6 }),
    text({ x: 0.75, y: 0.78, w: 11.8, h: 0.62, s: title, size: 26, bold: true, color: C.navy }),
  ];
  if (subtitle) out.push(text({ x: 0.75, y: 1.34, w: 11.8, h: 0.3, s: subtitle, size: 12, color: C.muted }));
  out.push(rect({ x: 0.75, y: subtitle ? 1.7 : 1.48, w: 11.83, h: 0.014, fill: C.line }));
  out.push(...footer(n));
  return out;
}

/* ---------- diapositivas ---------- */
const slides = [];

/* 1 · Portada */
slides.push({
  notes: 'Presentación de revisión técnica de OpenClaw: qué existe hoy, qué falta frente a los documentos de actualización y en qué orden construirlo.',
  shapes: [
    rect({ x: 0, y: 0, w: W, h: H, fill: C.navy }),
    rect({ x: 0, y: 0, w: 0.18, h: H, fill: C.teal }),
    rect({ x: 9.6, y: 0, w: 0.012, h: H, fill: '24405F' }),
    text({ x: 9.95, y: 1.55, w: 3, h: 0.26, s: 'ESTADO DEL SISTEMA', size: 8.5, bold: true, color: '6E8CAB', charSpacing: 1.6 }),
    text({ x: 9.95, y: 1.95, w: 3, h: 0.9, s: '29', size: 40, bold: true, color: C.white }),
    text({ x: 9.95, y: 2.62, w: 3, h: 0.3, s: 'componentes definidos', size: 9.5, color: '8FA7C0' }),
    text({ x: 9.95, y: 3.3, w: 3, h: 0.9, s: '9', size: 40, bold: true, color: C.teal }),
    text({ x: 9.95, y: 3.97, w: 3, h: 0.3, s: 'con código', size: 9.5, color: '8FA7C0' }),
    text({ x: 9.95, y: 4.65, w: 3, h: 0.9, s: '20', size: 40, bold: true, color: 'C9A25E' }),
    text({ x: 9.95, y: 5.32, w: 3, h: 0.3, s: 'pendientes', size: 9.5, color: '8FA7C0' }),

    text({ x: 1.05, y: 1.85, w: 8, h: 0.3, s: 'REVISIÓN TÉCNICA', size: 10.5, bold: true, color: C.teal, charSpacing: 3 }),
    text({ x: 1.05, y: 2.2, w: 8.4, h: 1.25, s: 'OpenClaw', size: 62, bold: true, color: C.white }),
    text({ x: 1.05, y: 3.5, w: 8.2, h: 0.9, s: 'Arquitectura actual y plan de actualización', size: 23, color: 'A9C1DC' }),
    rect({ x: 1.05, y: 4.45, w: 2.6, h: 0.035, fill: C.teal }),
    text({ x: 1.05, y: 4.72, w: 8.2, h: 0.32, s: 'Juan Carlos Guzmán Escandón · JeiKei AI Solutions', size: 12.5, color: 'D3E0EE' }),
    text({ x: 1.05, y: 5.12, w: 8.2, h: 0.32, s: '20 de noviembre de 2026 · OpenClaw_Clean_Architecture', size: 10.5, color: '8AA0B8' }),
    text({ x: 1.05, y: 6.35, w: 8.2, h: 0.3, s: 'Base: 3 documentos de fases + 21 archivos Python', size: 9.5, color: '6E8CAB' })
  ]
});

/* 2 · Resumen ejecutivo */
const kpi = (x, num, color, label) => [
  rect({ x, y: 1.95, w: 3.7, h: 1.85, fill: C.soft, line: C.line, radius: 0.06 }),
  text({ x: x + 0.32, y: 2.15, w: 3.1, h: 0.95, s: num, size: 46, bold: true, color }),
  text({ x: x + 0.32, y: 3.05, w: 3.05, h: 0.6, s: label, size: 11, color: C.muted, lineSpacing: 1.15 })
];
slides.push({
  notes: 'Los tres números que resumen la brecha: 29 componentes definidos en los documentos, 9 con código real y 20 sin implementar. El veredicto es prototipo parcial.',
  shapes: [
    ...header('Resumen ejecutivo', 'Un prototipo a mitad de camino', 2),
    ...kpi(0.75, '29', C.navy, 'componentes definidos entre los tres documentos'),
    ...kpi(4.72, '9', C.teal, 'con código en el repositorio (31 %)'),
    ...kpi(8.69, '20', C.red, 'pendientes de implementar'),

    rect({ x: 0.75, y: 4.05, w: 11.83, h: 0.95, fill: C.redSoft, radius: 0.05 }),
    rect({ x: 0.75, y: 4.05, w: 0.07, h: 0.95, fill: C.red }),
    text({ x: 1.02, y: 4.18, w: 11.3, h: 0.7, s: 'Veredicto: prototipo funcional parcial. No está listo para producción ni para ejecutar operaciones financieras reales.', size: 13.5, bold: true, color: '7E2A1F' }),

    text({ x: 0.75, y: 5.25, w: 11.8, h: 0.26, s: 'TRES CONCLUSIONES', size: 8.5, bold: true, color: C.faint, charSpacing: 1.4 }),
    text({ x: 0.75, y: 5.55, w: 11.8, h: 0.3, s: '1.  El pipeline se corta tras la extracción: no hay normalización, persistencia ni decisión.', size: 12, color: C.ink }),
    text({ x: 0.75, y: 5.95, w: 11.8, h: 0.3, s: '2.  El grafo LangGraph no está compilado; los nodos existen aislados y sin estado compartido.', size: 12, color: C.ink }),
    text({ x: 0.75, y: 6.35, w: 11.8, h: 0.3, s: '3.  El webhook /buy puede ordenar un retiro fijo de 125.000 USDT sin autenticación ni confirmación.', size: 12, color: C.ink })
  ]
});

/* 3 · Qué hace el sistema */
const step = (x, n, name, status, color, soft) => [
  rect({ x, y: 2.0, w: 2.1, h: 1.2, fill: C.white, line: C.line, radius: 0.05 }),
  text({ x: x + 0.2, y: 2.12, w: 1.7, h: 0.24, s: n, size: 9, bold: true, color: C.faint }),
  text({ x: x + 0.2, y: 2.42, w: 1.75, h: 0.34, s: name, size: 12.5, bold: true, color: C.navy }),
  rect({ x: x + 0.2, y: 2.84, w: 1.02, h: 0.26, fill: soft, radius: 0.12 }),
  text({ x: x + 0.2, y: 2.87, w: 1.02, h: 0.22, s: status, size: 8, bold: true, color, align: 'center' })
];
slides.push({
  notes: 'El objetivo de negocio es filtrar oportunidades con cap rate ≥ 8 % y precio ≤ 150.000 USD. Hoy el sistema cubre descubrimiento y extracción, pero no la evaluación conectada.',
  shapes: [
    ...header('Propósito', 'Qué hace el sistema, en cinco pasos', 3),
    ...step(0.75, '01', 'Descubrir', 'activo', C.teal, C.tealSoft),
    ...step(3.1, '02', 'Extraer', 'activo', C.teal, C.tealSoft),
    ...step(5.45, '03', 'Normalizar', 'falta', C.red, C.redSoft),
    ...step(7.8, '04', 'Evaluar', 'parcial', C.amber, C.amberSoft),
    ...step(10.15, '05', 'Alertar', 'activo', C.teal, C.tealSoft),

    text({ x: 0.75, y: 3.62, w: 11.8, h: 0.3, s: 'Objetivo de negocio', size: 11.5, bold: true, color: C.navy }),
    text({ x: 0.75, y: 3.95, w: 7.3, h: 1.1, s: 'Detectar propiedades con cap rate ≥ 8 % y precio ≤ 150.000 USD, y descartar el resto antes de consumir tiempo comercial. El sistema recorre los dos primeros pasos de forma sólida; los tres restantes están incompletos o desconectados.', size: 11.5, color: C.ink, lineSpacing: 1.35 }),

    rect({ x: 8.45, y: 3.9, w: 4.13, h: 1.25, fill: C.soft, line: C.line, radius: 0.05 }),
    text({ x: 8.65, y: 4.02, w: 3.8, h: 0.24, s: 'FÓRMULA IMPLEMENTADA', size: 8, bold: true, color: C.faint, charSpacing: 1.2 }),
    text({ x: 8.65, y: 4.3, w: 3.8, h: 0.75, s: 'NOI = ingreso − (8 % + 10 % + 2 % × precio + 1.200)\ncap rate = NOI / precio × 100', size: 9, color: C.ink, font: 'Consolas', lineSpacing: 1.3 }),

    text({ x: 0.75, y: 5.4, w: 11.8, h: 0.26, s: 'LÍMITE DEL ALCANCE ACTUAL', size: 8.5, bold: true, color: C.faint, charSpacing: 1.4 }),
    text({ x: 0.75, y: 5.7, w: 11.8, h: 0.9, s: 'La inspección visual no analiza imágenes: aplica un CAPEX fijo de 15.000 USD. El resultado de la extracción nunca se valida contra el esquema ni se guarda, así que no existe historial de oportunidades ni forma de auditar una decisión pasada.', size: 11.5, color: C.muted, lineSpacing: 1.35 })
  ]
});

/* 4 · Arquitectura actual */
const band = (y, title, sub, h) => [
  rect({ x: 0.75, y, w: 8.35, h: h || 0.78, fill: C.white, line: C.line, radius: 0.05 }),
  text({ x: 0.98, y: y + 0.1, w: 7.9, h: 0.28, s: title, size: 12, bold: true, color: C.navy }),
  text({ x: 0.98, y: y + 0.38, w: 7.9, h: 0.28, s: sub, size: 9, color: C.muted, font: 'Consolas' })
];
const panelRow = (y, name, desc, color) => [
  dot({ x: 9.72, y: y + 0.09, r: 0.05, fill: color }),
  text({ x: 9.92, y, w: 2.5, h: 0.24, s: name, size: 9.5, bold: true, color: C.navy }),
  text({ x: 9.92, y: y + 0.22, w: 2.5, h: 0.24, s: desc, size: 8, color: C.muted })
];
slides.push({
  notes: 'Flujo real del código: API FastAPI, Redis Streams, worker con consumer group y Firecrawl con rate limiter. El orquestador existe como archivos sueltos, sin grafo.',
  shapes: [
    ...header('Arquitectura actual', 'Cuatro piezas conectadas y un vacío', 4),
    ...band(1.9, 'API Gateway · FastAPI', 'main.py · /iniciar-barrido · sin auth ni validación'),
    arrow({ x: 4.83, y: 2.72, w: 0.2, h: 0.24, fill: C.faint }),
    ...band(3.0, 'Redis Streams', 'openclaw_scraping_queue · sin dead-letter queue'),
    arrow({ x: 4.83, y: 3.82, w: 0.2, h: 0.24, fill: C.faint }),
    ...band(4.1, 'Worker asíncrono', "consumer group 'scrapers' · XREADGROUP + XACK"),
    arrow({ x: 4.83, y: 4.92, w: 0.2, h: 0.24, fill: C.faint }),
    ...band(5.2, 'Firecrawl + rate limiter', '3 reintentos · 15–30 rpm por dominio · fail-open'),

    arrow({ x: 4.83, y: 6.02, w: 0.2, h: 0.22, fill: C.red }),
    rect({ x: 0.75, y: 6.28, w: 8.35, h: 0.52, fill: C.redSoft, line: C.red, lineWidth: 1, dash: 'dash', radius: 0.05 }),
    text({ x: 0.98, y: 6.4, w: 7.9, h: 0.3, s: 'El resultado se descarta: no hay normalización, persistencia ni respuesta al cliente', size: 10, bold: true, color: '7E2A1F' }),

    rect({ x: 9.35, y: 1.9, w: 3.23, h: 4.9, fill: C.soft, line: C.line, radius: 0.05 }),
    text({ x: 9.55, y: 2.02, w: 2.9, h: 0.26, s: 'ORQUESTADOR', size: 8.5, bold: true, color: C.faint, charSpacing: 1.3 }),
    text({ x: 9.55, y: 2.26, w: 2.9, h: 0.26, s: 'piezas aisladas', size: 9.5, color: C.muted }),
    ...panelRow(2.62, 'financial.py', 'NOI y cap rate · funcional', C.teal),
    ...panelRow(3.24, 'vision.py', 'CAPEX fijo · placeholder', C.amber),
    ...panelRow(3.86, 'telegram_alert.py', 'alerta · sin control de errores', C.amber),
    ...panelRow(4.48, 'workflow.py', 'solo enrutador de precio', C.amber),
    rect({ x: 9.55, y: 5.18, w: 2.83, h: 1.32, fill: C.redSoft, radius: 0.04 }),
    text({ x: 9.68, y: 5.28, w: 2.6, h: 0.26, s: 'GRAFO LANGGRAPH', size: 8, bold: true, color: '7E2A1F', charSpacing: 1.1 }),
    text({ x: 9.68, y: 5.54, w: 2.6, h: 0.85, s: 'No compilado. Ningún flujo recorre los nodos de principio a fin.', size: 8.5, color: '7E2A1F', lineSpacing: 1.25 })
  ]
});

/* 5 · Flujo end-to-end */
const flowRows = [
  ['Etapa', 'Implementación', 'Estado'],
  ['Recibir la orden', 'POST /v1/openclaw/iniciar-barrido', 'Listo'],
  ['Encolar', 'XADD al stream de Redis', 'Listo'],
  ['Extraer', 'Firecrawl search + reintentos', 'Listo'],
  ['Validar y normalizar', 'DatosPropiedadNormalizados', 'Falta'],
  ['Evaluar finanzas', 'NOI y cap rate', 'Aislado'],
  ['Inspección visual', 'CAPEX fijo, sin imágenes', 'Simulado'],
  ['Persistir y decidir', 'Sin almacenamiento', 'Falta'],
  ['Alertar', 'Telegram sendMessage', 'Listo']
];
const statusColor = { Listo: C.teal, Falta: C.red, Aislado: C.amber, Simulado: C.amber };
const statusSoft = { Listo: C.tealSoft, Falta: C.redSoft, Aislado: C.amberSoft, Simulado: C.amberSoft };
const flowTable = flowRows.map((r, i) => {
  if (i === 0) return r.map((c) => ({ text: c, options: { bold: true, color: C.navy, size: 10 } }));
  return [
    { text: r[0], options: { bold: true, color: C.ink, size: 10.5 } },
    { text: r[1], options: { color: C.muted, size: 10, font: 'Consolas' } },
    { text: r[2], options: { color: statusColor[r[2]], bold: true, size: 9.5, align: 'left' } }
  ];
});
slides.push({
  notes: 'Ocho etapas: cuatro funcionan, dos son simuladas y dos no existen. La tabla es el mapa de trabajo inmediato.',
  shapes: [
    ...header('Flujo end-to-end', 'Dónde se rompe el recorrido', 5),
    table({ x: 0.75, y: 1.9, w: 11.83, colW: [3.5, 5.3, 3.03], rowH: 0.44, rows: flowTable, headFill: C.soft }),
    text({ x: 0.75, y: 6.2, w: 11.8, h: 0.6, s: 'Cuatro etapas funcionan, dos son simulaciones y dos no existen. Sin la validación del esquema y sin persistencia, ninguna decisión puede reconstruirse después.', size: 11, color: C.muted, lineSpacing: 1.3 })
  ]
});

/* 6 · Inventario de código */
const invRows = [
  ['Archivo', 'Responsabilidad', 'Estado'],
  ['api/main.py + routes/', 'Gateway, ingesta y webhook', 'Parcial'],
  ['services/scraper.py', 'Extracción con Firecrawl', 'Completo'],
  ['services/rate_limiter.py', 'Límite por dominio en Redis', 'Completo'],
  ['workers/ingestion_worker.py', 'Consumo del stream', 'Completo'],
  ['models/property.py', 'Esquema de propiedad', 'Sin uso'],
  ['orchestrator/nodes/financial.py', 'NOI y cap rate', 'Completo'],
  ['orchestrator/nodes/vision.py', 'CAPEX fijo de 15.000 USD', 'Placeholder'],
  ['orchestrator/workflow.py', 'Enrutador de due diligence', 'Incompleto']
];
const invColor = { Parcial: C.amber, Completo: C.teal, 'Sin uso': C.amber, Placeholder: C.amber, Incompleto: C.red };
const invTable = invRows.map((r, i) => {
  if (i === 0) return r.map((c) => ({ text: c, options: { bold: true, color: C.navy, size: 10 } }));
  return [
    { text: r[0], options: { color: C.ink, size: 10, font: 'Consolas' } },
    { text: r[1], options: { color: C.muted, size: 10 } },
    { text: r[2], options: { color: invColor[r[2]], bold: true, size: 9.5 } }
  ];
});
slides.push({
  notes: 'Inventario de los 21 archivos Python. Ocho tienen lógica real; el resto son init vacíos. Faltan dependencias declaradas para lo que los documentos proponen.',
  shapes: [
    ...header('Inventario', 'Qué hay realmente en el repositorio', 6),
    table({ x: 0.75, y: 1.9, w: 11.83, colW: [4.0, 5.1, 2.73], rowH: 0.43, rows: invTable, headFill: C.soft }),
    rect({ x: 0.75, y: 6.15, w: 11.83, h: 0.66, fill: C.soft, radius: 0.04 }),
    text({ x: 0.95, y: 6.24, w: 11.4, h: 0.5, s: 'requirements.txt declara fastapi, uvicorn, firecrawl-py, pydantic, langchain-openai, langgraph, redis y httpx. Faltan prometheus_client, fpdf y langchain_core que exigen las fases 95–104.', size: 9.5, color: C.muted, lineSpacing: 1.25 })
  ]
});

/* 7 · Mapa de fases */
const bar = (y, label, done, total) => {
  const trackW = 7.2, pct = done / total;
  const out = [
    text({ x: 0.75, y: y + 0.12, w: 1.7, h: 0.3, s: label, size: 13, bold: true, color: C.navy }),
    rect({ x: 2.55, y, w: trackW, h: 0.54, fill: 'EDF2F8', radius: 0.04 })
  ];
  if (pct > 0) out.push(rect({ x: 2.55, y, w: Math.max(trackW * pct, 0.06), h: 0.54, fill: C.teal, radius: 0.04 }));
  out.push(text({ x: 2.72, y: y + 0.14, w: 6.9, h: 0.3, s: `${done} de ${total} con código`, size: 10, bold: true, color: pct > 0.25 ? C.white : C.muted }));
  out.push(text({ x: 10.0, y: y + 0.1, w: 1.4, h: 0.34, s: `${Math.round(pct * 100)} %`, size: 15, bold: true, color: pct === 0 ? C.red : (pct < 1 ? C.amber : C.teal) }));
  return out;
};
slides.push({
  notes: 'Solo el bloque 0–94 tiene código, y de forma parcial. Los bloques 95–104 y 110–117 son especificación. Además, dos de los cuatro documentos son el mismo archivo.',
  shapes: [
    ...header('Cobertura', 'Qué bloque de fases está implementado', 7),
    text({ x: 0.75, y: 1.85, w: 11.8, h: 0.26, s: 'COMPONENTES CON CÓDIGO SOBRE EL TOTAL DEFINIDO', size: 8.5, bold: true, color: C.faint, charSpacing: 1.3 }),
    ...bar(2.3, 'Fases 0–94', 9, 10),
    ...bar(3.15, 'Fases 95–104', 0, 10),
    ...bar(4.0, 'Fases 110–117', 0, 9),

    rect({ x: 0.75, y: 5.05, w: 5.8, h: 1.5, fill: C.soft, line: C.line, radius: 0.05 }),
    text({ x: 0.95, y: 5.16, w: 5.4, h: 0.26, s: 'BRECHA TOTAL', size: 8.5, bold: true, color: C.faint, charSpacing: 1.3 }),
    text({ x: 0.95, y: 5.44, w: 5.4, h: 1.0, s: 'El 69 % de los componentes solicitados no tiene una línea de código. Los documentos 95–104 y 110–117 describen una hoja de ruta, no un sistema instalado.', size: 10.5, color: C.ink, lineSpacing: 1.3 }),

    rect({ x: 6.78, y: 5.05, w: 5.8, h: 1.5, fill: C.amberSoft, radius: 0.05 }),
    text({ x: 6.98, y: 5.16, w: 5.4, h: 0.26, s: 'HALLAZGO DOCUMENTAL', size: 8.5, bold: true, color: C.amber, charSpacing: 1.3 }),
    text({ x: 6.98, y: 5.44, w: 5.4, h: 1.0, s: '“Actualización en Ruta” y “Fases 110–117” son el mismo documento: solo cambia la primera línea del título. Son tres documentos distintos, no cuatro.', size: 10.5, color: '6B4713', lineSpacing: 1.3 })
  ]
});

/* 8 · Bloque 95–104 */
const cards95 = [
  ['95', 'Rastreador post-oferta', 'Vigila si la propiedad pasa a pending o sold'],
  ['96', 'Estratega macro', 'Ajusta el cap rate objetivo con un LLM'],
  ['97', 'Auditoría legal', 'Liens, back taxes y quitclaim deeds'],
  ['98', 'WebSocket', 'Telemetría en vivo sobre Redis Pub/Sub'],
  ['99', 'Seguridad Zero-Trust', 'API key X-OpenClaw-Tactical-Key'],
  ['100', 'Despliegue Docker', 'Redis, pgvector, workers, Prometheus, Grafana'],
  ['101', 'Métricas', 'Contador Prometheus por estado final'],
  ['102', 'Contratos LOI', 'Carta de intención en PDF con FPDF'],
  ['103', 'Puente escrow', 'Transferencia de earnest money'],
  ['104', 'Negociador NLP', 'Evalúa contraofertas por correo']
];
const card95 = (i, x, y) => {
  const [n, title, desc] = cards95[i];
  return [
    rect({ x, y, w: 2.15, h: 2.05, fill: C.white, line: C.line, radius: 0.05 }),
    text({ x: x + 0.18, y: y + 0.14, w: 1.8, h: 0.24, s: `FASE ${n}`, size: 8, bold: true, color: C.red, charSpacing: 1 }),
    text({ x: x + 0.18, y: y + 0.42, w: 1.8, h: 0.62, s: title, size: 11, bold: true, color: C.navy, lineSpacing: 1.1 }),
    text({ x: x + 0.18, y: y + 1.12, w: 1.8, h: 0.72, s: desc, size: 8.5, color: C.muted, lineSpacing: 1.2 }),
    text({ x: x + 0.18, y: y + 1.76, w: 1.8, h: 0.2, s: 'sin código', size: 8, bold: true, color: C.red })
  ];
};
slides.push({
  notes: 'Diez módulos pedidos por el documento 95–104. Ninguno existe. Incluyen seguridad, observabilidad, contratos y escrow: son la base operativa antes de cualquier integración financiera.',
  shapes: [
    ...header('Bloque 95–104', 'Diez módulos solicitados, ninguno implementado', 8),
    ...[0, 1, 2, 3, 4].flatMap((i) => card95(i, 0.75 + i * 2.42, 1.85)),
    ...[5, 6, 7, 8, 9].flatMap((i) => card95(i, 0.75 + (i - 5) * 2.42, 4.1)),
    text({ x: 0.75, y: 6.35, w: 11.8, h: 0.5, s: 'Este bloque aporta lo que falta para operar: autenticación, métricas, despliegue reproducible, documentos y control de pagos.', size: 10.5, color: C.muted })
  ]
});

/* 9 · Bloque 110–117 */
const cards110 = [
  ['110–111', 'Ingesta endurecida', 'Validación de payload y errores controlados'],
  ['—', 'Worker 9 plataformas', 'Zillow, Realtor, Redfin, Trulia, LoopNet y 4 más'],
  ['117', 'Circuit breaker WAF', '2 fallos → modo sigiloso; 4 → pausa de 900 s'],
  ['116', 'Telemetría WebSocket', 'Feed en vivo del scraping para JeiKeiStation'],
  ['112', 'Tesorería DeFi', 'Capital ocioso hacia UAO_Grid en USDC'],
  ['113', 'Off-market + SMS', 'Skip tracing y mensajería con Twilio'],
  ['114', 'AGNET RPC', 'EIP-712 y liquidación USDC en Base'],
  ['115', 'Oráculo geoespacial', 'Gentrification Score por código postal'],
  ['—', 'Seguridad Zero-Trust', 'Protección de la ruta de ingesta']
];
const card110 = (i, x, y) => {
  const [n, title, desc] = cards110[i];
  return [
    rect({ x, y, w: 3.75, h: 1.32, fill: C.white, line: C.line, radius: 0.05 }),
    text({ x: x + 0.2, y: y + 0.14, w: 1.6, h: 0.22, s: n === '—' ? 'TRANSVERSAL' : `FASE ${n}`, size: 7.5, bold: true, color: C.red, charSpacing: 0.9 }),
    text({ x: x + 0.2, y: y + 0.38, w: 3.3, h: 0.28, s: title, size: 11.5, bold: true, color: C.navy }),
    text({ x: x + 0.2, y: y + 0.7, w: 3.3, h: 0.5, s: desc, size: 8.5, color: C.muted, lineSpacing: 1.2 })
  ];
};
slides.push({
  notes: 'Nueve componentes del bloque 110–117. Varios implican decisiones legales y de cumplimiento, no solo técnicas: evasión de WAF, skip tracing, SMS y liquidación on-chain.',
  shapes: [
    ...header('Bloque 110–117', 'La capa avanzada, también sin código', 9),
    ...[0, 1, 2].flatMap((i) => card110(i, 0.75 + i * 3.99, 1.85)),
    ...[3, 4, 5].flatMap((i) => card110(i, 0.75 + (i - 3) * 3.99, 3.32)),
    ...[6, 7, 8].flatMap((i) => card110(i, 0.75 + (i - 6) * 3.99, 4.79)),
    rect({ x: 0.75, y: 6.28, w: 11.83, h: 0.62, fill: C.amberSoft, radius: 0.04 }),
    text({ x: 0.95, y: 6.38, w: 11.4, h: 0.45, s: 'Antes de programar: evasión de WAF, skip tracing, SMS y liquidación on-chain requieren revisión legal y de cumplimiento. Empezar en sandbox con límites e interruptor de apagado.', size: 9.5, color: '6B4713', lineSpacing: 1.2 })
  ]
});

/* 10 · Riesgos */
const risk = (x, y, tag, tagColor, tagSoft, title, body) => [
  rect({ x, y, w: 5.8, h: 1.9, fill: C.white, line: C.line, radius: 0.05 }),
  rect({ x, y, w: 0.06, h: 1.9, fill: tagColor }),
  rect({ x: x + 0.25, y: y + 0.18, w: 1.05, h: 0.26, fill: tagSoft, radius: 0.12 }),
  text({ x: x + 0.25, y: y + 0.21, w: 1.05, h: 0.2, s: tag, size: 7.5, bold: true, color: tagColor, align: 'center' }),
  text({ x: x + 1.45, y: y + 0.16, w: 4.1, h: 0.3, s: title, size: 13, bold: true, color: C.navy }),
  text({ x: x + 0.25, y: y + 0.62, w: 5.3, h: 1.15, s: body, size: 10, color: C.muted, lineSpacing: 1.3 })
];
slides.push({
  notes: 'Cuatro riesgos ordenados por gravedad. El primero es el único que puede mover dinero: /buy ejecuta un retiro fijo sin validar origen, monto ni idempotencia.',
  shapes: [
    ...header('Riesgos', 'Lo que hay que cerrar antes de seguir', 10),
    ...risk(0.75, 1.85, 'CRÍTICO', C.red, C.redSoft, '/buy mueve dinero sin control', 'El webhook ejecuta un retiro fijo de 125.000 USDT hacia fiat_bridge_bank. No valida el chat de origen, el monto, el saldo ni la idempotencia, y no tiene timeout ni auditoría.'),
    ...risk(6.78, 1.85, 'ALTO', C.amber, C.amberSoft, 'Ingesta abierta', 'La ruta de barrido no exige credencial. Un payload incompleto produce un error no controlado en lugar de un 400, y no hay lista de plataformas permitidas.'),
    ...risk(0.75, 3.95, 'ALTO', C.amber, C.amberSoft, 'Sin trazabilidad', 'No se guarda la propiedad, la decisión ni el error. Es imposible reconstruir por qué se descartó una oportunidad o qué se envió a Telegram.'),
    ...risk(6.78, 3.95, 'MEDIO', C.blue, C.blueSoft, 'Configuración frágil', 'La API key de Firecrawl vale “demo” por defecto y el rate limiter deja pasar todo si Redis falla. Ambos degradan en silencio, sin alerta.'),

    rect({ x: 0.75, y: 6.1, w: 11.83, h: 0.62, fill: C.navy, radius: 0.04 }),
    text({ x: 0.98, y: 6.2, w: 11.4, h: 0.42, s: 'Antes de cualquier prueba con dinero real: congelar /buy y operar solo en sandbox con confirmación explícita.', size: 10.5, bold: true, color: C.white })
  ]
});

/* 11 · Plan por etapas */
const stage = (i, x, letter, title, items, outcome) => {
  const y = 1.85, h = 4.55;
  const out = [
    rect({ x, y, w: 2.15, h, fill: i === 0 ? C.navy : C.white, line: i === 0 ? C.navy : C.line, radius: 0.05 }),
    text({ x: x + 0.2, y: y + 0.18, w: 0.6, h: 0.3, s: letter, size: 16, bold: true, color: i === 0 ? C.teal : C.blue }),
    text({ x: x + 0.2, y: y + 0.55, w: 1.78, h: 0.5, s: title, size: 11, bold: true, color: i === 0 ? C.white : C.navy, lineSpacing: 1.1 })
  ];
  items.forEach((it, k) => {
    out.push(text({ x: x + 0.2, y: y + 1.2 + k * 0.62, w: 1.78, h: 0.58, s: it, size: 8.5, color: i === 0 ? 'C3D3E5' : C.muted, lineSpacing: 1.2 }));
  });
  out.push(rect({ x: x + 0.2, y: y + 3.5, w: 1.75, h: 0.012, fill: i === 0 ? '3B5675' : C.line }));
  out.push(text({ x: x + 0.2, y: y + 3.62, w: 1.75, h: 0.75, s: outcome, size: 8, bold: true, color: i === 0 ? C.teal : C.blue, lineSpacing: 1.2 }));
  return out;
};
slides.push({
  notes: 'Cinco etapas secuenciales. A y B son condiciones para tocar dinero; C cierra el producto; D y E amplían. Cada etapa tiene un resultado verificable.',
  shapes: [
    ...header('Plan de actualización', 'Cinco etapas, en este orden', 11),
    ...stage(0, 0.75, 'A', 'Estabilizar el núcleo', ['Contratos Pydantic', 'Validación y errores 400', 'API y worker separados', 'Health checks'], 'Resultado: barridos que no se pierden'),
    ...stage(1, 3.17, 'B', 'Cerrar seguridad', ['Auth en la ingesta', 'Telegram restringido', '/buy parametrizado', 'Secretos fuera del código'], 'Resultado: ninguna orden anónima'),
    ...stage(2, 5.59, 'C', 'Completar el análisis', ['Compilar LangGraph', 'Conectar los nodos', 'Persistir cada decisión', 'Visión real o dato nulo'], 'Resultado: decisión auditable'),
    ...stage(3, 8.01, 'D', 'Observabilidad', ['Métricas Prometheus', 'Logs con job id', 'Telemetría WebSocket', 'Docker reproducible'], 'Resultado: sistema operable'),
    ...stage(4, 10.43, 'E', 'Integraciones', ['Contratos y escrow', 'Off-market y SMS', 'DeFi y AGNET', 'Oráculo geoespacial'], 'Resultado: solo tras revisión legal')
  ]
});

/* 12 · Decisiones abiertas */
const decision = (y, q, a, b, rec) => [
  rect({ x: 0.75, y, w: 11.83, h: 1.02, fill: C.white, line: C.line, radius: 0.04 }),
  text({ x: 0.95, y: y + 0.14, w: 4.2, h: 0.72, s: q, size: 11.5, bold: true, color: C.navy, lineSpacing: 1.15 }),
  text({ x: 5.35, y: y + 0.16, w: 3.3, h: 0.3, s: a, size: 9, color: C.muted }),
  text({ x: 5.35, y: y + 0.5, w: 3.3, h: 0.3, s: b, size: 9, color: C.muted }),
  rect({ x: 8.95, y: y + 0.16, w: 3.45, h: 0.7, fill: C.tealSoft, radius: 0.04 }),
  text({ x: 9.12, y: y + 0.22, w: 3.1, h: 0.22, s: 'RECOMENDADO', size: 7, bold: true, color: C.teal, charSpacing: 1 }),
  text({ x: 9.12, y: y + 0.44, w: 3.1, h: 0.34, s: rec, size: 8.5, bold: true, color: '0C5B53' })
];
slides.push({
  notes: 'Cuatro decisiones bloquean el diseño. Mientras no se resuelvan, cada módulo nuevo se construye sobre un contrato distinto.',
  shapes: [
    ...header('Decisiones', 'Cuatro definiciones que desbloquean el resto', 12),
    ...decision(1.85, '¿Plataformas por URL o por catálogo?', 'Hoy: URL libre en el payload', 'Documentos: claves tipo “zillow”', 'Catálogo validado en servidor'),
    ...decision(2.98, '¿Qué patrón de Redis?', 'Hoy: consumer group + XACK', 'Documentos: XREAD 0-0 + XDEL', 'Consumer group con DLQ'),
    ...decision(4.11, '¿Dónde se persiste?', 'Hoy: nada', 'Documentos: pgvector sin detalle', 'Postgres; Redis solo como cola'),
    ...decision(5.24, '¿Qué alcance tiene /buy?', 'Hoy: activo con monto fijo', 'Riesgo: retiro no autorizado', 'Desactivado; sandbox con confirmación'),
    text({ x: 0.75, y: 6.5, w: 11.8, h: 0.4, s: 'Mientras no se resuelvan, cada módulo nuevo se construye sobre un contrato distinto y habrá que reescribirlo.', size: 10, color: C.faint })
  ]
});

/* 13 · Criterios de aceptación */
const check = (x, y, s) => [
  rect({ x, y: y + 0.02, w: 0.22, h: 0.22, fill: C.white, line: C.teal, lineWidth: 1.2, radius: 0.03 }),
  text({ x: x + 0.36, y, w: 5.1, h: 0.62, s, size: 10, color: C.ink, lineSpacing: 1.25 })
];
slides.push({
  notes: 'Definición de terminado para la primera prueba integrada. Si estos ocho puntos no se cumplen, el sistema sigue siendo una demo.',
  shapes: [
    ...header('Criterios', 'Definición de “listo para probar”', 13),
    ...check(0.75, 1.9, 'La ingesta exige credencial y devuelve 400 ante payload incompleto.'),
    ...check(0.75, 2.62, 'Cada barrido tiene id y estado: queued, running, completed o failed.'),
    ...check(0.75, 3.34, 'La propiedad se valida contra el esquema y se guarda con su origen.'),
    ...check(0.75, 4.06, 'El worker confirma mensajes y tiene reintentos y dead-letter queue.'),
    ...check(6.85, 1.9, 'El grafo recorre todas las etapas y conserva cada decisión.'),
    ...check(6.85, 2.62, 'Telegram solo escribe al chat autorizado y registra errores.'),
    ...check(6.85, 3.34, '/buy está desactivado o solo opera en sandbox con confirmación.'),
    ...check(6.85, 4.06, 'Hay pruebas de auth, validación, rate limit, worker y finanzas.'),

    rect({ x: 0.75, y: 5.05, w: 11.83, h: 1.45, fill: C.soft, line: C.line, radius: 0.05 }),
    text({ x: 0.98, y: 5.18, w: 11.3, h: 0.26, s: 'ORDEN SUGERIDO DE TRABAJO', size: 8.5, bold: true, color: C.faint, charSpacing: 1.3 }),
    text({ x: 0.98, y: 5.46, w: 11.3, h: 0.95, s: 'Primero congelar /buy y cerrar la ingesta (etapas A y B). Después compilar el grafo y añadir persistencia (etapa C), porque sin historial ninguna prueba posterior es verificable. La telemetría y los módulos avanzados van al final.', size: 10.5, color: C.ink, lineSpacing: 1.3 })
  ]
});

/* 14 · Cierre */
slides.push({
  notes: 'Cierre: tres acciones concretas para la próxima semana. El sistema tiene buena base técnica; el problema es de alcance y de controles, no de arquitectura.',
  shapes: [
    rect({ x: 0, y: 0, w: W, h: H, fill: C.navy }),
    rect({ x: 0, y: 0, w: 0.18, h: H, fill: C.teal }),
    text({ x: 1.05, y: 1.35, w: 9, h: 0.3, s: 'SIGUIENTE PASO', size: 10, bold: true, color: C.teal, charSpacing: 2.6 }),
    text({ x: 1.05, y: 1.75, w: 10.5, h: 1.4, s: 'Tres acciones para esta semana', size: 40, bold: true, color: C.white }),

    text({ x: 1.05, y: 3.35, w: 0.5, h: 0.4, s: '1', size: 17, bold: true, color: C.teal }),
    text({ x: 1.6, y: 3.3, w: 9.6, h: 0.42, s: 'Congelar /buy y dejar el webhook en modo simulación con registro.', size: 13.5, color: 'D3E0EE' }),
    text({ x: 1.05, y: 3.95, w: 0.5, h: 0.4, s: '2', size: 17, bold: true, color: C.teal }),
    text({ x: 1.6, y: 3.9, w: 9.6, h: 0.42, s: 'Definir contratos: catálogo de plataformas, patrón Redis y almacenamiento.', size: 13.5, color: 'D3E0EE' }),
    text({ x: 1.05, y: 4.55, w: 0.5, h: 0.4, s: '3', size: 17, bold: true, color: C.teal }),
    text({ x: 1.6, y: 4.5, w: 9.6, h: 0.42, s: 'Compilar el grafo y persistir una propiedad de punta a punta.', size: 13.5, color: 'D3E0EE' }),

    rect({ x: 1.05, y: 5.35, w: 3.2, h: 0.035, fill: '3B5675' }),
    text({ x: 1.05, y: 5.55, w: 10.5, h: 0.35, s: 'La base técnica es razonable: el problema es de alcance y de controles, no de arquitectura.', size: 11, color: '8AA0B8' }),
    text({ x: 1.05, y: 6.45, w: 10.5, h: 0.3, s: 'OpenClaw · JeiKei AI Solutions · Noviembre 2026', size: 9, color: '5C7A96' })
  ]
});

/* ---------- render PPTX ---------- */
function renderPptx() {
  const pptx = new PptxGenJS();
  pptx.defineLayout({ name: 'OC', width: W, height: H });
  pptx.layout = 'OC';
  pptx.author = 'Juan Carlos Guzmán Escandón';
  pptx.company = 'JeiKei AI Solutions';
  pptx.title = 'OpenClaw — Arquitectura y plan de actualización';

  slides.forEach((spec) => {
    const slide = pptx.addSlide();
    spec.shapes.forEach((sh) => {
      if (sh.t === 'rect') {
        slide.addShape(sh.radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
          x: sh.x, y: sh.y, w: sh.w, h: sh.h,
          fill: sh.fill ? { color: sh.fill } : { color: C.white },
          line: sh.line ? { color: sh.line, width: sh.lineWidth || 0.75, dashType: sh.dash || 'solid' } : { color: sh.fill || C.white, width: 0 },
          rectRadius: sh.radius || 0
        });
      } else if (sh.t === 'text') {
        slide.addText(sh.s, {
          x: sh.x, y: sh.y, w: sh.w, h: sh.h,
          fontFace: sh.font || FONT,
          fontSize: sh.size || 11,
          color: sh.color || C.ink,
          bold: !!sh.bold,
          italic: !!sh.italic,
          align: sh.align || 'left',
          valign: sh.valign || 'top',
          charSpacing: sh.charSpacing || 0,
          lineSpacingMultiple: sh.lineSpacing || 1.0,
          margin: 0,
          fit: 'shrink'
        });
      } else if (sh.t === 'arrow') {
        slide.addShape(pptx.ShapeType.downArrow, {
          x: sh.x, y: sh.y, w: sh.w, h: sh.h,
          fill: { color: sh.fill }, line: { color: sh.fill, width: 0 }
        });
      } else if (sh.t === 'dot') {
        slide.addShape(pptx.ShapeType.ellipse, {
          x: sh.x, y: sh.y, w: sh.r * 2, h: sh.r * 2,
          fill: { color: sh.fill }, line: { color: sh.fill, width: 0 }
        });
      } else if (sh.t === 'table') {
        slide.addTable(sh.rows, {
          x: sh.x, y: sh.y, w: sh.w, colW: sh.colW, rowH: sh.rowH,
          fontFace: FONT, fontSize: 10, color: C.ink,
          border: { type: 'solid', color: C.line, pt: 0.5 },
          fill: { color: C.white },
          valign: 'middle', margin: 0.05, autoPage: false
        });
      }
    });
    if (spec.notes) slide.addNotes(spec.notes);
  });

  return pptx.writeFile({ fileName: path.join(OUT_DIR, 'OpenClaw_Arquitectura_y_Plan.pptx') });
}

/* ---------- render HTML de control ---------- */
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function renderHtml() {
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });
  const px = (v) => (v * 100).toFixed(2) + 'px';
  const pt = (v) => (v * 100 / 72).toFixed(2) + 'px';

  slides.forEach((spec, idx) => {
    const body = spec.shapes.map((sh) => {
      if (sh.t === 'rect') {
        const bg = sh.fill ? '#' + sh.fill : '#fff';
        const border = sh.line ? `1px ${sh.dash === 'dash' ? 'dashed' : 'solid'} #${sh.line}` : 'none';
        return `<div style="position:absolute;left:${px(sh.x)};top:${px(sh.y)};width:${px(sh.w)};height:${px(sh.h)};background:${bg};border:${border};border-radius:${(sh.radius || 0) * 100}px;box-sizing:border-box"></div>`;
      }
      if (sh.t === 'text') {
        const lh = sh.lineSpacing || 1.0;
        return `<div style="position:absolute;left:${px(sh.x)};top:${px(sh.y)};width:${px(sh.w)};height:${px(sh.h)};font-family:${sh.font || FONT};font-size:${pt(sh.size || 11)};line-height:${lh};color:#${sh.color || C.ink};font-weight:${sh.bold ? 600 : 400};text-align:${sh.align || 'left'};letter-spacing:${(sh.charSpacing || 0) / 72 * 100}px;white-space:pre-wrap;overflow:hidden">${esc(sh.s)}</div>`;
      }
      if (sh.t === 'arrow') {
        return `<div style="position:absolute;left:${px(sh.x)};top:${px(sh.y)};width:${px(sh.w)};height:${px(sh.h)};background:#${sh.fill};clip-path:polygon(30% 0,70% 0,70% 55%,100% 55%,50% 100%,0 55%,30% 55%)"></div>`;
      }
      if (sh.t === 'dot') {
        return `<div style="position:absolute;left:${px(sh.x)};top:${px(sh.y)};width:${px(sh.r * 2)};height:${px(sh.r * 2)};background:#${sh.fill};border-radius:50%"></div>`;
      }
      if (sh.t === 'table') {
        const cols = sh.colW.map((w) => `<col style="width:${px(w)}">`).join('');
        const rows = sh.rows.map((row, ri) => {
          const cells = row.map((cell) => {
            const o = cell.options || {};
            const bg = ri === 0 ? `#${C.soft}` : (o.fill ? '#' + o.fill : 'transparent');
            return `<td style="padding:0 6px;height:${px(sh.rowH)};border-bottom:1px solid #${C.line};background:${bg};font-family:${o.font || FONT};font-size:${pt(o.size || 10)};color:#${o.color || C.ink};font-weight:${o.bold ? 600 : 400};text-align:${o.align || 'left'};vertical-align:middle">${esc(cell.text)}</td>`;
          }).join('');
          return `<tr>${cells}</tr>`;
        }).join('');
        return `<table style="position:absolute;left:${px(sh.x)};top:${px(sh.y)};width:${px(sh.w)};border-collapse:collapse;table-layout:fixed">${cols}${rows}</table>`;
      }
      return '';
    }).join('\n');

    const html = `<!doctype html><html><head><meta charset="utf-8"><style>
      html,body{margin:0;padding:0;background:#fff}
      #stage{position:relative;width:1333.33px;height:750px;overflow:hidden;background:#fff}
    </style></head><body><div id="stage">${body}</div></body></html>`;
    fs.writeFileSync(path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, '0')}.html`), html);
  });
  return slides.length;
}

module.exports = { slides, C, FONT, W, H };

if (require.main === module) {
  (async () => {
    const n = renderHtml();
    await renderPptx();
    console.log(`OK: ${n} diapositivas generadas`);
  })();
}
