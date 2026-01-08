# Conversando con los Datos (MVP)

Herramienta de análisis cualitativo asistido por IA que permite "conversar" con datos empíricos y triangular hallazgos con literatura teórica mediante un flujo secuencial (RAG en Cascada).

## Características
* **Pipeline Secuencial**: Grounding (Datos) -> Bridging (Consulta Teórica) -> Theorizing (Literatura) -> Síntesis.
* **Embeddings Serverless**: Usa la API de VoyageAI (`voyage-3-large`) para evitar costos y cuotas.
* **Sugerencias automáticas**: Topic modelling + LLM ligero para propuestas de preguntas.
* **Trazabilidad**: Visualización transparente de la evidencia utilizada en cada paso.

## Documentación
* `documentation/` contiene notas de integración de proveedores (embeddings/LLMs).
* `plans/` contiene los planes de implementación discutidos.

## Requisitos
* Python 3.11+
* Groq API Key
* VoyageAI API Key (Gratuita)

## Instalación

1. Crear un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
Renombrar o crear un archivo `.env` y añadir claves:
```
GROQ_KEY=tu_api_key_de_groq
VOYAGE_KEY=tu_api_key_de_voyage
```

## Ejecución
```bash
streamlit run app.py
```
