# Requisitos Legales y de Cumplimiento — Escrow Bancario (Fase 99/103 · Puerta G2)

- **Estado operativo actual:** **BLOQUEADO** (`ESCROW_MODE=disabled` por defecto; solo existe `FakeEscrowProvider` en sandbox).
- **Alcance:** Ningún flujo del repositorio puede conectar cuentas bancarias reales ni mover fondos sin cierre escrito de **Puerta G2**.

## 1. Requisitos KYC / KYB y AML / BSA obligatorios antes de G2

1. **Verificación de identidad (KYC/KYB):**
   - Verificación documental de compradores (entidad LLC/corporativa, beneficiarios finales / UBO) y vendedores antes de emitir instrucciones de depósito.
2. **Prevención de lavado de activos (AML/BSA):**
   - Screening OFAC/SDN de todas las contrapartes y cuentas beneficiarias.
   - Límites máximos por instrucción y diarios con registro inmutable de auditoría.
3. **Entidad regulada y segregación de fondos:**
   - OpenClaw **no custodia fondos**; toda instrucción real debe ejecutarse mediante una compañía de título / escrow licenciada por estado de EE. UU. con cuentas fiduciarias segregadas (*IOLTA / Trust Escrow Account*).
4. **Control dual (4-eyes) y conciliación:**
   - Dos identidades humanas distintas obligatorias para aprobar cualquier instrucción.
   - Idempotencia estricta por `idempotency_key` y paso automático a `requiere_revision` ante cualquier discrepancia de monto o referencia.
