# Plan de Feature: Búsqueda Agéntica (Deep Research) para Libro de Códigos

Fecha: 2026-01-07

## Objetivo
Implementar un flujo de investigación agéntica que produzca un **libro de códigos** para entrevistas, combinando RAG, topic modelling y citas textuales de los datos empíricos. Debe usar metadatos ingresados, literatura y evidencia empírica con trazabilidad.

---

## Alcance funcional
- Generar **códigos cualitativos** (definición, criterios de inclusión/exclusión, ejemplos).
- Sustentar cada código con **citas textuales** (fragmentos empíricos).
- Incorporar **literatura** y **metadatos** para teoría y contexto.
- Permitir revisión y edición manual del libro de códigos.

---

## Inputs esperados
1) **Metadatos** del estudio (disciplina, perspectiva, tema, etc.).
2) **Datos empíricos** (entrevistas, transcripciones).
3) **Antecedentes** (literatura / teoría).
4) **Resultados de topic modelling** (topics + keywords + labels).

---

## Flujo agéntico propuesto (alto nivel)
1) **Preparación**
   - Verificar índices (datos + antecedentes).
   - Verificar tópicos asignados y labels por chunk.

2) **Exploración / Discovery**
   - Agente recupera temas dominantes por consulta global.
   - Genera un mapa inicial de posibles códigos (borrador).

3) **Evidencia empírica**
   - Para cada código, recuperar top‑k fragmentos.
   - Filtrar por relevancia y diversidad (no repetir la misma fuente).

4) **Triangulación teórica**
   - Buscar conceptos en literatura que respalden o tensionen el código.
   - Agregar referencias teóricas breves.

5) **Síntesis**
   - Emitir un libro de códigos estructurado con:
     - Nombre del código
     - Definición
     - Criterios (inclusión/exclusión)
     - Evidencia empírica (citas)
     - Soporte teórico (resumen + fuente)

---

## Contrato de salida estructurado (propuesta)
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
      ]
    }
  ]
}
```

---

## Entregables
1) **Orquestador agéntico** para ejecutar los pasos (discovery → evidencia → triangulación → síntesis).
2) **Schema JSON** para codebook con trazabilidad.
3) **UI de revisión** para editar códigos y re‑generar secciones.
4) **Export** (JSON + Markdown).

---

## Tests manuales (dev)
- Generar un libro de códigos con 3 entrevistas y verificar que:
  - Cada código tiene al menos 2 citas empíricas.
  - Hay variedad de fuentes.
  - Los criterios de inclusión/exclusión son claros.
- Forzar un tópico y comprobar que aparece en el codebook.
- Editar un código en UI y guardar sin perder trazabilidad.

---

## Riesgos / Consideraciones
- Balance entre granularidad de códigos vs. sobre‑fragmentación.
- Evitar sesgo del modelo en definiciones (verificación humana).
- Mantener trazabilidad en todo el flujo.
