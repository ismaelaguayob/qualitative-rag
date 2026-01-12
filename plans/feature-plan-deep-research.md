# Plan de Feature: Flujo Agéntico para Libro de Códigos (Deep Research)

Fecha: 2026-01-10

## Objetivo
Diseñar e implementar un flujo agéntico que produzca un **libro de códigos** para entrevistas, combinando RAG, topic modeling y citas textuales con trazabilidad. El agente debe **planificar**, **ejecutar pasos**, **auto‑replanificar** y **registrar eventos** de manera auditable.

---

## Alcance funcional
- Generar **códigos cualitativos** con definición y criterios de inclusión/exclusión.
- Sustentar cada código con **citas empíricas** trazables (chunk_id + fuente).
- Incorporar **antecedentes** (literatura/teoría) como soporte o tensión teórica.
- Permitir **revisión y edición manual** por código.
- Exportar a **JSON y Markdown**.

---

## Inputs esperados
1) **Metadatos** del estudio (disciplina, perspectiva, tema, etc.).
2) **Datos empíricos** (entrevistas, transcripciones) indexados en Chroma.
3) **Antecedentes** (literatura/teoría) indexados en Chroma.
4) **Tópicos** (topic_id, keywords, labels por chunk).

---

## Principios de diseño
- **Trazabilidad por defecto:** todo quote debe mapear a chunk_id existente.
- **Diversidad de evidencia:** limitar citas por fuente para evitar sesgo.
- **Planificación explícita:** el agente debe listar pasos y replanificar.
- **Observabilidad:** logs por step y evento.
- **Compatibilidad futura:** diseño listo para migrar a LlamaIndex Workflows.

---

## Alternativas de implementación

### Opción A: Orquestador manual (recomendado para iterar rápido)
**Pros:** control total, menos dependencias, integración directa con Streamlit.
**Contras:** hay que implementar reintentos, eventos y validación a mano.

**Recomendación:** usar esta opción pero con interfaz “workflow‑ready”.

### Opción B: LlamaIndex Workflows
**Pros:** event‑driven, pasos tipados, validación del grafo, streaming de eventos.
**Contras:** potencial fricción de versiones y refactor async.

**Decisión propuesta:** iniciar manual con eventos compatibles y evaluar migración.

---

## Diseño del flujo agéntico (alto nivel)

### 1) Preparación
- Verificar índices (datos + antecedentes).
- Verificar tópicos asignados y labels.
- Cargar configuración (k, max_iter, límites de citas por fuente).

### 2) Planificación inicial
- Generar plan JSON con pasos + herramientas sugeridas.
- Restringir a 6‑10 pasos.

### 3) Exploración / Discovery
- Resumen global de tópicos dominantes.
- Recuperación de chunks representativos por tópico.
- Generación de **candidatos de códigos** (borrador).

### 4) Evidencia empírica
- Para cada candidato:
  - retrieve_top_k_empirical(query=code_label + keywords)
  - filtrar duplicados y limitar por fuente
  - seleccionar 2‑4 citas con chunk_id

### 5) Triangulación teórica
- retrieve_top_k_theory(query=code_label + términos clave)
- resumir 1‑2 aportes teóricos con fuente

### 6) Síntesis de código
- Definición, inclusión/exclusión
- Citas empíricas + soporte teórico
- Nivel de confianza + notas

### 7) Control de calidad
- Cobertura por tópicos (sin tópicos dominantes huérfanos)
- Detección de redundancias por similitud
- Re‑planificación si hay huecos

---

## Herramientas disponibles para el agente
(El número debe mantenerse acotado para evitar saturación)

1) `get_topics_summary()`
2) `retrieve_empirical(query, k, filters)`
3) `retrieve_theory(query, k, filters)`
4) `get_topic_chunks(topic_id, k)`
5) `dedupe_quotes(quotes, per_source_limit)`
6) `score_code_coverage(code, topics)`
7) `merge_codes(c1, c2)`
8) `log_event(step, payload)`

---

## Estado interno (state) sugerido
- `session_id`, `run_id`
- `inputs`: metadatos, query inicial, config
- `memory`: pasos hechos, decisiones, hipótesis
- `artifacts`: mapa de tópicos, candidatos, citas
- `errors`: fallas parciales + fallback
- `metrics`: tiempos, tokens, cobertura

---

## Contrato de salida estructurado (JSON)
```json
{
  "codebook": [
    {
      "code_id": "c1",
      "title": "…",
      "definition": "…",
      "inclusion": ["…"],
      "exclusion": ["…"],
      "empirical_quotes": [
        {"quote": "…", "source": "archivo.pdf", "chunk_id": "c34"}
      ],
      "theory_support": [
        {"summary": "…", "source": "paper.pdf"}
      ],
      "confidence": "low|medium|high",
      "notes": "…"
    }
  ]
}
```

---

## Registro de eventos (logging)
- `StartEvent`, `PlanEvent`, `ToolCallEvent`, `ObservationEvent`, `ErrorEvent`, `StopEvent`.
- Persistir en `data_temp/debug/` para trazabilidad y troubleshooting.

---

## UI / UX
- Botón “Generar libro de códigos”.
- Vista de revisión por código con edición inline.
- Botón “Re‑generar solo este código”.
- Export JSON + Markdown.

---

## Riesgos y mitigaciones
- **Sobre‑fragmentación:** limitar códigos por tópico.
- **Sesgo teórico:** forzar al menos una fuente de antecedentes cuando existan.
- **Baja diversidad empírica:** limitar citas por fuente.
- **Prompt drift:** plantillas consistentes por step.

---

## Entregables
1) Orquestador agéntico con pasos explícitos y re‑planificación.
2) Schema JSON + validación de trazabilidad.
3) UI de revisión/edición y export.
4) Logs por evento para auditoría.

---

## Tests manuales
- Generar libro de códigos con 3 entrevistas:
  - cada código tiene >= 2 citas empíricas
  - diversidad de fuentes
  - criterios claros
- Forzar un tópico y comprobar aparición en codebook.
- Editar un código y guardar sin perder trazabilidad.

---

## Plan de implementación sugerido

### Fase 1 — Orquestador manual “workflow‑ready”
- Definir `state` + eventos y logging.
- Implementar pasos base (plan → discovery → evidencia → teoría → síntesis → QA).
- Validar schema JSON y trazabilidad.

### Fase 2 — UI + edición
- UI por código + re‑generación parcial.
- Export JSON/Markdown.

### Fase 3 — Migración opcional a LlamaIndex Workflows
- Adaptar pasos a `@step`.
- Integrar streaming de eventos si aplica.

---

## Notas finales
- Mantener entradas nuevas de UI en `I18N` (es/en).
- No guardar datos sensibles en repo.
