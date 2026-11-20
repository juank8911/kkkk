# Fase 115 — Revisión de Sesgos, Fair Housing y Licencias Geoespaciales

**Estado actual:** `disabled` por defecto. No se permite activar scoring productivo sin dictamen legal firmado y aprobación de la Puerta G2.

## 1. Fuentes autorizadas y licencias documentadas

Todo dato ingerido por `GeospatialDataPort` debe registrar obligatoriamente `source`, `license`, `data_version`, `spatial_resolution` y `observed_at`:

| Fuente | Tipo de señal | Licencia / Uso permitido | Restricciones |
|---|---|---|---|
| Portales municipales de datos abiertos (Permisos de construcción / comercio) | `commercial_permits_yoy_pct`, `infrastructure_projects_count` | Public Domain / CC-BY-4.0 / Open Government License | Conservar atribución y fecha de corte; solo agregados por zona |
| Autoridades de transporte público (GTFS / planes de inversión en tránsito) | `transit_investment_index` | CC0 / Public Domain / Licencia abierta municipal | Prohibido inferir perfiles individuales de residentes |
| OpenStreetMap (POI comerciales agregados) | `commercial_density_per_km2` | ODbL 1.0 | Cumplir atribución `© OpenStreetMap contributors` y requisitos ODbL |

## 2. Variables estrictamente prohibidas (Fair Housing Act / ECOA)

`app/services/geospatial_oracle.py` aplica validación estricta por allowlist + lista negra explícita (`PROHIBITED_PROXY_VARIABLES`). Se rechaza con `ProhibitedProxyVariableError` cualquier atributo o proxy de clases protegidas:

- Raza, etnia, color, ascendencia u origen nacional (`race`, `ethnicity`, `color`, `ancestry`, `national_origin`, `minority_pct`).
- Religión, idioma hablado, sexo, género, estado civil, edad, discapacidad o estado familiar (`religion`, `language_spoken`, `sex`, `gender`, `marital_status`, `age_group`, `disability`, `familial_status`).
- Proxies históricos o discriminatorios de redlining (`redlining_grade`, `holc_grade`, `income_by_race`, `demographic_composition`, `zip_demographic_proxy`, `crime_by_demographic`).

## 3. Uso permitido y prohibición de decisiones automáticas

- El resultado (`GentrificationScoreResult`) se marca siempre con `informativo_unicamente=True` y `uso_decision_automatica_prohibido=True`.
- **Nunca** puede bloquear, aprobar, rechazar ni priorizar automáticamente a personas, solicitudes de vivienda, ofertas o crédito.
- Cada salida expone la descomposición lineal exacta (`contribuciones`) de las 4 variables objetivas normalizadas para garantizar reproducibilidad y auditabilidad.

## 4. Criterios de pruebas de sesgo y puerta G2

Antes de cambiar el módulo desde `GeospatialMode.DISABLED`:

1. Ejecutar `GeospatialOracleService.evaluar_estabilidad_y_disparidad(...)` sobre un conjunto representativo de zonas comparables verificando estabilidad ante perturbaciones (±5 %) y ausencia de disparidad injustificada.
2. Completar revisión legal independiente de Fair Housing Act (42 U.S.C. § 3601 et seq.) y ECOA / Regulation B.
3. Verificar licencias de redistribución y atribución de cada proveedor activo en `GeospatialDataPort`.
4. Confirmar firma explícita de G2 (`g2_legal_and_bias_approved=True` solo tras aprobación documentada).
