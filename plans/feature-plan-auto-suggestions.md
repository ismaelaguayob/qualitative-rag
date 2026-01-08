# Plan de Feature: Auto-suggestions (topic modelling + modelo ligero)

Fecha: 2026-01-07

## Objetivo
Generar sugerencias de preguntas relevantes al **query del usuario** y al **contexto recuperado**, combinando topic modelling sobre embeddings existentes y un modelo ligero para redactar/estructurar las sugerencias. Además, persistir tópicos (ID + label + keywords) en columnas/metadata para análisis futuros.

---

## Decisión técnica principal
- **Topic modelling con K fijo** usando K‑means como clustering principal.
- **Etiquetado de tópicos** con TF‑IDF (o c‑TF‑IDF) por cluster para obtener keywords representativas.
- **Modelo ligero** sólo para:
  1) crear una **label general** de tópico (una vez por tópico), y
  2) redactar/normalizar preguntas sugeridas.

---

## Enfoque propuesto (híbrido)

### A) Construcción/actualización de tópicos
- **Inicial (batch)**
  - Usar embeddings ya calculados.
  - Reducir dimensionalidad (opcional) para clustering estable.
  - Aplicar **K‑means** para asignar `topic_id` a cada chunk/documento.
  - Calcular keywords por tópico con TF‑IDF/c‑TF‑IDF.

### B) Persistencia de tópicos (columnas)
- Guardar en metadata/columnas:
  - `topic_id` (int), `topic_label` (str), `topic_keywords` (list/str), `topic_version` (int/date).
- Esos campos deben vivir junto a cada documento/chunk en el vector store.

### C) Flujo de sugerencias (por query)
1) Recuperar top‑k chunks relevantes (ya existe en RAG).
2) Agregar los `topic_id` más frecuentes y sus keywords.
3) Filtrar tópicos por relevancia al query (similaridad embeddings o matching TF‑IDF).
4) Generar **candidatas base** con plantillas (por tipo de tópico).
5) Pasar (query + tópicos + keywords + candidatas) al **modelo ligero** para:
   - Redacción clara en español.
   - Evitar duplicados.
   - Entregar 3‑6 sugerencias.

---

## Contrato de salida estructurado
```json
{
  "suggestions": [
    {
      "id": "s1",
      "text": "…",
      "topic_id": 12,
      "topic_label": "…",
      "source_chunk_ids": ["c34", "c51"]
    }
  ]
}
```

---

## Flujo de labels (requerido por la solicitud)

### Primera vez (tópico sin label)
- Entradas: keywords + snippets representativos + ejemplos de chunks del tópico.
- Salida del modelo ligero:
  - `topic_label` general (comprehensivo).
  - `suggestions[]` iniciales.
- Persistir `topic_label` en columna/metadata para todos los documentos del tópico.

### Próximas veces (tópico ya etiquetado)
- El modelo **no** vuelve a generar `topic_label`.
- Usa `query + topic_label + topic_keywords` para producir nuevas sugerencias alineadas al query.

---

## Entregables
1) **Módulo de topic modelling** ✅
   - K‑means con K configurable.
   - TF‑IDF/c‑TF‑IDF por tópico.
   - Export de `topic_id`, `topic_keywords`.

2) **Persistencia en columnas/metadata** ✅
   - Guardar `topic_id`, `topic_label`, `topic_keywords` en el vector store.

3) **Servicio de sugerencias** ✅
   - Selección de tópicos por relevancia al query.
   - Modelo ligero para redacción final.
   - Salida JSON estructurada.

4) **UI** ✅
   - Botones de quick‑reply debajo de la respuesta.
   - Logs de clicks (para evaluar utilidad).

---

## Tests manuales (dev)
- **Relevancia al query:** hacer 3 consultas distintas y verificar cambio de tópicos/sugerencias.
- **Persistencia:** confirmar que `topic_id` y `topic_label` quedan guardados en metadata.
- **Primera vez vs posteriores:**
  - Forzar un tópico nuevo y comprobar que se crea label.
  - Repetir query similar y confirmar que no se recalcula el label.
- **Calidad de sugerencias:** verificar 3‑6 preguntas claras, no duplicadas.
- **Actualización incremental (si aplica):** agregar documentos nuevos y validar que sólo se actualicen tópicos necesarios.

Estado: **Completado**.

---

## Notas de configuración
- Variables de entorno existentes: `GROQ_KEY`, `VOYAGE_KEY`.
- Evitar persistir data sensible en `data_temp/`.
