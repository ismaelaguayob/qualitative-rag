# Plan de Feature: Contextual Citations (citas inline)

Fecha: 2026-01-07

## Objetivo
Forzar salida estructurada del LLM con mapeo directo `answer_block → citations` y renderizar citas inline con anchors a fragmentos concretos.

---

## Entregables
1) **Metadatos de chunks estables**
   - Cada chunk con: `chunk_id`, `doc_id`, `title`, `page`, `start_char`, `end_char`.

2) **Contrato de salida estructurado**
   - JSON con `answer_blocks[]` y `citations[]`.
   - Listado de `sources[]` desnormalizado.

3) **Validador de citas**
   - Verificar que `chunk_id` exista en el top‑k usado.
   - Fallback a formato clásico si falla el parseo.

4) **Render UI**
   - Inline markers por bloque (`[1]`, `[2]`).
   - Popover/expander con texto del chunk + metadatos.

---

## Detalles técnicos propuestos
- Prompt de LLM: “JSON‑only, sin texto adicional”.
- Cada bloque (párrafo/lista) debe citar uno o más chunks.
- Las fuentes no usadas no se muestran.

---

## Tests manuales (dev)
- Hacer una consulta factual y comprobar que cada párrafo tiene al menos una cita.
- Click en un marcador inline y verificar que abre el fragmento correcto.
- Confirmar que no aparecen citas de chunks fuera del top‑k.
- Forzar un error de formato (cambiando el prompt) y verificar fallback.

---

## Notas
- Evitar persistir data sensible en `data_temp/`.
- Variables de entorno existentes: `GROQ_KEY`, `VOYAGE_KEY`.
