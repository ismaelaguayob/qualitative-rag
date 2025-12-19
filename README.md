# Conversando con los Datos (MVP)

Herramienta de análisis cualitativo asistido por IA que permite "conversar" con datos empíricos y triangular hallazgos con literatura teórica mediante un flujo secuencial (RAG en Cascada).

## Características
* **Pipeline Secuencial**: Grounding (Datos) -> Bridging (Consulta Teórica) -> Theorizing (Literatura) -> Síntesis.
* **Embeddings Serverless**: Usa la API de Hugging Face (`intfloat/multilingual-e5-large`) para evitar costos y cuotas.
* **Trazabilidad**: Visualización transparente de la evidencia utilizada en cada paso.

## Requisitos
* Python 3.11+
* `google-generativeai` API Key (Gemini)
* Hugging Face User Access Token (Gratuito)

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
GOOGLE_API_KEY=tu_api_key_de_google
HF_TOKEN=tu_token_de_hugging_face
```

## Ejecución
```bash
streamlit run app.py
```
