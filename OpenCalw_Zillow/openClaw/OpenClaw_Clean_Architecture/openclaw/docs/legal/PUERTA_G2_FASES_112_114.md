# Puerta Legal Bloqueante G2 — Fases 112, 113 y 114 (Dev 2)

- **Estado:** **BLOQUEADAS** (`kill_switch_activo=True`, `dictamen_legal_firmado=False`).
- **Política:** No existe integración operativa con Twilio, Aave V3, wallets ni contratos EIP-712 en producción. Solo se entregan modelos contractuales, documentación y pruebas de cumplimiento (`app/services/compliance_gates.py`).

## Matriz de requisitos previos a cualquier implementación real

| Fase | Capacidad bloqueada | Controles contractuales modelados | Casos de fraude / reversión / sanciones | Criterio bloqueante G2 |
|---|---|---|---|---|
| **Fase 112** | Tesorería DeFi (USDC / Aave V3) | KYC/KYB, screening OFAC, doble control, kill switch | Congelamiento automático si `score_fraude >= 0.70`, revisión manual de reversión, prohibición de smart contracts sin auditoría externa | Dictamen contable/fiscal/regulatorio sobre custodia de activos digitales + auditoría externa de smart contracts. |
| **Fase 113** | Adquisición off-market (skip tracing + SMS) | `SmsConsentRecord`: consentimiento previo expreso escrito, verificación DNC, opt-out inmediato | Bloqueo duro en DNC y revocación opt-out; prohibición de mensajería sin base de licitud | Dictamen TCPA, DNC Registry, CCPA/CPRA y políticas de retención/borrado de datos personales. |
| **Fase 114** | AGNET RPC con EIP-712 y liquidación USDC | Screening OFAC de direcciones, KYC/KYB de contrapartes, control dual y mTLS/HSM | Bloqueo ante sanciones o jurisdicción prohibida; registro inmutable para reporte regulatorio | Dictamen FinCEN/MSB (transmisión de dinero), auditoría criptográfica externa y sandbox segregado. |
