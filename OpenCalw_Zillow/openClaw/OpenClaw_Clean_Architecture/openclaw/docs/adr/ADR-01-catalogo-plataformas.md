# ADR-01: Contrato de plataformas con catálogo cerrado en servidor

- **Estado:** Aprobado
- **Fecha:** 2026-11-20
- **Autores / Firmantes:** Dev B (Dominio e IA) · Dev A (Plataforma)
- **Tarea backlog:** `S0-03` (habilita `S1-02`, `S1-07`, `S1-11`)

## Contexto

El prototipo inicial en `app/api/routes/ingestion.py` y `app/services/scraper.py` acepta URLs arbitrarias en el campo `plataformas`, mientras que la especificación de fases 110–117 introduce claves simbólicas (`zillow`, `realtor`, `redfin`, etc.). Permitir URLs libres enviadas por el cliente expone al servicio a vulnerabilidades de tipo SSRF (Server-Side Request Forgery), dificulta el control de tasa por dominio real y vuelve impredecible la normalización posterior.

## Decisión

Se adopta un **catálogo cerrado de 9 claves de plataforma mantenido en el servidor**, cada una asociada a una plantilla de URL validada y a su dominio canónico:

| Clave | Dominio canónico | Plantilla de URL |
|---|---|---|
| `zillow` | `zillow.com` | `https://www.zillow.com/homes/{query}_rb/` |
| `realtor` | `realtor.com` | `https://www.realtor.com/realestateandhomes-search/{query}` |
| `redfin` | `redfin.com` | `https://www.redfin.com/city/{query}` |
| `trulia` | `trulia.com` | `https://www.trulia.com/{query}` |
| `loopnet` | `loopnet.com` | `https://www.loopnet.com/search/commercial-real-estate/{query}/for-sale/` |
| `craigslist` | `craigslist.org` | `https://{query}.craigslist.org/search/rea` |
| `century21` | `century21.com` | `https://www.century21.com/real-estate/{query}/` |
| `coldwell` | `coldwellbankerhomes.com` | `https://www.coldwellbankerhomes.com/{query}/` |
| `remax` | `remax.com` | `https://www.remax.com/homes-for-sale/{query}` |

Reglas obligatorias:
1. El cliente solo puede enviar claves del catálogo (normalizadas en minúsculas y deduplicadas).
2. Cualquier clave desconocida —incluyendo URLs crudas— es rechazada con HTTP `422 Unprocessable Entity` indicando explícitamente la lista de claves válidas.
3. El parámetro `query` se sanea antes de interpolarlo en la plantilla y se verifica en servidor que el host resultante pertenezca al dominio canónico de la plataforma (defensa en profundidad contra SSRF).

## Alternativa descartada

- **URL libre enviada por el cliente:** descartada por riesgo de SSRF hacia redes internas o hosts no autorizados, inconsistencia en el rate limiting por dominio y fragilidad en los parsers.

## Consecuencias

- **Positivas:** elimina SSRF por construcción, habilita límites de tasa y concurrencia por dominio real y alinea contratos de API, worker y normalizador.
- **Costes:** agregar una décima plataforma exige actualización controlada en el catálogo del servidor y migración de datos semilla (`S2-04`).
