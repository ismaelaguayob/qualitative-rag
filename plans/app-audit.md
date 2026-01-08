# Auditoría Técnica de `app.py` y Flujo Actual

Fecha: 2026-01-08

## Resumen ejecutivo
Se detectan riesgos de **i18n inconsistente**, **errores de reset/estado** y **flujo de sugerencias** que pueden generar comportamientos inesperados. No hay fallos críticos visibles, pero sí varios puntos de fragilidad que conviene resolver antes de ampliar features (agentes, deploy).

---

## Hallazgos (por severidad)

### Alto
1) **Strings fuera de i18n**
   - Ejemplos: warnings/errors directos (e.g. “VOYAGE_KEY no encontrado…”, errores de embedding, pipeline).
   - Impacto: UI incoherente y difícil de mantener si se amplía la internacionalización.
   - Acción: mover a `I18N` y usar `get_text`.

2) **Reset y consistencia del estado**
   - El reset por API está correcto, pero el flujo de bootstrap y build de tópicos depende de `index_datos` en session state y de colecciones que podrían no existir si el usuario no inicializa índices.
   - Impacto: warnings y fallos silenciosos (bootstrap suggestions fallidas).
   - Acción: añadir guards explícitos antes de bootstrap y label generation, y mensajes de estado claros.

### Medio
3) **Sugerencias se regeneran sin control de “vistas”**
   - Bootstrap se recalcula en cada carga si no hay mensajes. Puede consumir tokens innecesarios.
   - Acción: memoizar por sesión o poner botón “Generar sugerencias”.

4) **Metadatos de tópicos**
   - Dependencia fuerte en metadata local y refresh de índice.
   - Acción: asegurar refresh post‑update (ya existe), pero agregar verificación de metadatos faltantes.

### Bajo
5) **Stopwords / keywords**
   - Aún pueden aparecer nombres propios frecuentes. No crítico, pero mejora calidad.
   - Acción: ajustar stopwords o filtrar nombres propios frecuentes por frecuencia en corpus.

---

## Observaciones sobre flujo de sugerencias
- Labels se generan globalmente por tópico (top‑k chunks), lo cual es correcto.
- Sugerencias se generan con tópicos filtrados por frecuencia en top‑k nodes.
- Bootstrap puede fallar si no hay colecciones; conviene manejar ese caso con feedback.

---

## Recomendaciones inmediatas
1) Normalizar **todas** las strings de UI/errores a `I18N` (ES/EN).
2) Añadir guard en bootstrap para no llamar si colecciones no existen.
3) Añadir memoización simple de bootstrap (una vez por sesión).

---

## Posibles siguientes pasos
- Añadir status visible: “Índices no inicializados”.
- Registrar en logs cuando se omite bootstrap por no existir colecciones.
- Considerar un “Modo debug” que muestre metadata en UI temporalmente.
