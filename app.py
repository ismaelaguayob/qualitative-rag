import streamlit as st
import os
import shutil
import re
import json
from collections import Counter, defaultdict
from dotenv import load_dotenv
import chromadb
import time
import traceback
import numpy as np
import hashlib
import sqlite3
import logging
import sys
import warnings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

from llama_index.llms.groq import Groq
from llama_index.embeddings.voyageai import VoyageEmbedding
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words as get_lang_stop_words

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CHROMA_PATH = "./chroma_db"
DATA_TEMP_PATH = "./data_temp"
DATOS_DIR = os.path.join(DATA_TEMP_PATH, "datos")
ANTECEDENTES_DIR = os.path.join(DATA_TEMP_PATH, "antecedentes")
EXTRACTED_TEXT_DIR = os.path.join(DATA_TEMP_PATH, "extracted_text")
DEBUG_DIR = os.path.join(DATA_TEMP_PATH, "debug")
DEV_LOGS_DIR = os.path.join(DATA_TEMP_PATH, "dev_sessions")
RERANK_LOG_PATH = os.path.join(DEBUG_DIR, "rerank_log.txt")

# --- i18n: Internationalization ---
I18N = {
    "es": {
        "page_title": "Conversando con Datos Cualitativos",
        "sidebar_header": "Configuración del Investigador",
        "model_selector": "Selector de Modelo",
        "api_key_error": "Por favor configura GROQ_KEY en el archivo .env",
        "voyage_key_missing": "VOYAGE_KEY no encontrado en .env. Embeddings pueden fallar.",
        "embedding_init_error": "Error al inicializar el modelo de embeddings: {error}",
        "indices_init_error": "Error al inicializar índices: {error}",
        "reset_error": "Error al reiniciar índices: {error}",
        "topic_build_error": "No se pudo crear tópicos: {error}",
        "pipeline_error": "Ocurrió un error al procesar la consulta. Revisa el log de errores.",
        "suggestions_error": "No se pudieron generar sugerencias: {error}",
        "indices_not_ready": "⚠️ Índices no inicializados. Sube archivos y crea índices para usar el chat.",
        "bootstrap_skipped": "Bootstrap omitido: índices no disponibles.",
        "discipline": "Disciplina",
        "perspective": "Perspectiva Teórica",
        "topic": "Tema de Investigación",
        "file_upload": "Carga de Archivos",
        "empirical_data": "Datos Empíricos / Fuentes Primarias",
        "literature": "Antecedentes / Literatura",
        "init_indices": "Inicializar Indices",
        "reset_all": "Reiniciar Todo",
        "clear_chat": "Limpiar Conversación",
        "explore_sources": "📂 Explorar Fuentes",
        "no_sources": "No hay fuentes indexadas aún.",
        "content": "Contenido",
        "main_title": "Conversando con los Datos",
        "chat_placeholder": "Escribe tu pregunta de investigación...",
        "init_indices_first": "Por favor inicializa los índices primero desde el panel lateral.",
        "analyzing": "Analizando datos y triangulando con teoría...",
        "evidence_title": "🔍 Revisar Evidencia y Trazabilidad",
        "tab_data": "Datos Empíricos",
        "tab_theory": "Antecedentes Teóricos",
        "source": "Fuente",
        "similarity": "Similitud",
        "view_full_source": "📄 Ver Fuente Completa",
        "extracted_content": "Contenido Extraído",
        "text_not_available": "Texto extraído no disponible para {source}. Reinicia los índices.",
        "bridge_query": "Query Puente",
        "indices_created": "Indices creados. Datos: {d} docs, Antecedentes: {a} docs.",
        "indices_failed": "Fallo al crear índices.",
        "indices_reset": "Índices reiniciados. Puedes subir nuevos archivos.",
        "processing": "Procesando archivos e índices...",
        "language": "Idioma",
        "advanced_settings": "⚙️ Ajustes Avanzados",
        "topic_k": "Número de tópicos (K)",
        "topic_k_help": "Cantidad de clusters temáticos para sugerencias.",
        "chunk_size": "Tamaño de Chunk (tokens)",
        "chunk_overlap": "Overlap de Chunk (tokens)",
        "top_k_data": "Top-K Datos Empíricos",
        "top_k_theory": "Top-K Antecedentes",
        "chunk_size_help": "Tamaño de cada fragmento en tokens. Menor = más granular.",
        "chunk_overlap_help": "Solapamiento en tokens entre chunks. Mayor = menos pérdida de contexto.",
        "top_k_help": "Cantidad de fragmentos a recuperar por consulta.",
        "rerank_model_label": "Modelo de Re-rank",
        "rerank_model_help": "Modelo de VoyageAI para re-rank de resultados.",
        "rerank_top_n_data": "Top-N Re-rank (Datos)",
        "rerank_top_n_theory": "Top-N Re-rank (Antecedentes)",
        "rerank_top_n_help": "Cantidad de resultados finales tras re-rank (debe ser <= Top-K).",
        "rerank_ratio_hint": "Sugerencia: usa un Top-K 2–3× mayor que el Top-N para mejorar el re-rank.",
        "settings_note": "⚠️ Reinicializa los índices después de cambiar estos valores.",
        "suggestions_title": "Sugerencias de preguntas",
        "suggestion_button": "➕ Usar sugerencia",
        "suggestions_system_prompt": (
            "Eres un asistente que genera sugerencias de investigación. "
            "Responde SOLO en JSON válido. No incluyas texto adicional."
        ),
        "suggestions_bootstrap_instructions": (
            "Genera 3-6 preguntas sugeridas relevantes al contexto y a los tópicos. "
            "No repitas preguntas. Responde en JSON con 'suggestions'."
        ),
        "suggestions_query_instructions": (
            "Genera 3-6 preguntas sugeridas relevantes al user_query y a los tópicos. "
            "No repitas preguntas. Responde en JSON con 'suggestions'."
        ),
        "suggestions_constraints": [
            "No hables como si fueras una persona entrevistada.",
            "No uses 'tu experiencia' ni primera/segunda persona dirigida al entrevistado.",
            "Formula preguntas como investigador/a (tercera persona)."
        ],
        "citations_title": "📌 Citas y Fuentes",
        "citation_label": "Cita",
        "citation_missing": "Fragmento no disponible para la cita {cite_id}.",
        "citations_parse_error": "La respuesta con citas no se pudo parsear. Mostrando formato clásico.",
        "citations_system_prompt": "Responde SOLO en JSON válido. No incluyas texto adicional fuera del JSON.",
        "citations_user_instructions": (
            "Usa SOLO los chunk_id disponibles en CATALOGO_CHUNKS_JSON. "
            "Cada bloque debe citar 1+ citas. "
            "Los labels deben usar títulos o nombres de archivo (no IDs crudos). "
            "Responde en JSON con el siguiente esquema:\n"
            "{\n"
            "  \"answer_blocks\": [{\"id\": \"string\", \"type\": \"paragraph|list\", \"text\": \"string\", \"citations\": [\"c1\"]}],\n"
            "  \"citations\": [{\"id\": \"c1\", \"label\": \"string\", \"chunk_ids\": [\"chunk_id\"]}],\n"
            "  \"sources\": [{\"chunk_id\": \"chunk_id\", \"source_type\": \"data|theory\"}]\n"
            "}"
        ),
        "page_label": "p.",
        "source_type_data": "Datos",
        "source_type_theory": "Antecedentes",
        "system_prompt": """Eres un investigador experto con un Doctorado en {discipline}.
Estás analizando fuentes primarias desde una perspectiva {perspective}.
Tu tema de investigación actual es: {topic}.

TU OBJETIVO:
Responder a la consulta del usuario analizando los DATOS EMPÍRICOS proporcionados.

TUS FUENTES DE INFORMACIÓN:
1. [DATOS EMPÍRICOS]: Esta es la evidencia textual que debes analizar (noticias, discursos, entrevistas, etc.). Es la verdad absoluta del caso.
2. [ANTECEDENTES]: Es literatura y teoría para apoyar tu interpretación.

INSTRUCCIONES DE RESPUESTA:
- Basa tus afirmaciones principalmente en los [DATOS EMPÍRICOS].
- Usa los [ANTECEDENTES] para teorizar o dar contexto a lo que encuentras en los datos.
- Si hay antecedentes relevantes, incluye al menos una cita de [ANTECEDENTES].
- Si los datos muestran algo distinto a la teoría, destaca esa tensión.
- No incluyas citas inline en el texto. Las citas se entregan solo por JSON.""",
        "bridge_prompt": """TAREA: Genera palabras clave para buscar literatura académica relevante.

CONTEXTO:
- Pregunta del usuario: "{query}"
- Datos empíricos encontrados: {context}

INSTRUCCIONES:
1. Extrae 5-10 conceptos clave de los datos que sean relevantes para la pregunta.
2. Los conceptos deben ser términos académicos/teóricos buscables en literatura científica.
3. Responde ÚNICAMENTE con los conceptos separados por comas.
4. NO incluyas explicaciones, solo la lista de conceptos.

FORMATO DE RESPUESTA:
concepto1, concepto2, concepto3, concepto4, concepto5

RESPUESTA:"""
    },
    "en": {
        "page_title": "Conversing with Qualitative Data",
        "sidebar_header": "Researcher Configuration",
        "model_selector": "Model Selector",
        "api_key_error": "Please set GROQ_KEY in the .env file",
        "voyage_key_missing": "VOYAGE_KEY not found in .env. Embeddings may fail.",
        "embedding_init_error": "Error initializing embedding model: {error}",
        "indices_init_error": "Error initializing indices: {error}",
        "reset_error": "Error resetting indices: {error}",
        "topic_build_error": "Failed to build topics: {error}",
        "pipeline_error": "An error occurred while processing the query. Check the error log.",
        "suggestions_error": "Failed to generate suggestions: {error}",
        "indices_not_ready": "⚠️ Indices not initialized. Upload files and create indices to use chat.",
        "bootstrap_skipped": "Bootstrap skipped: indices not available.",
        "discipline": "Discipline",
        "perspective": "Theoretical Perspective",
        "topic": "Research Topic",
        "file_upload": "File Upload",
        "empirical_data": "Empirical Data / Primary Sources",
        "literature": "Background / Literature",
        "init_indices": "Initialize Indices",
        "reset_all": "Reset All",
        "clear_chat": "Clear Chat",
        "explore_sources": "📂 Explore Sources",
        "no_sources": "No sources indexed yet.",
        "content": "Content",
        "main_title": "Conversing with Data",
        "chat_placeholder": "Write your research question...",
        "init_indices_first": "Please initialize the indices first from the sidebar.",
        "analyzing": "Analyzing data and triangulating with theory...",
        "evidence_title": "🔍 Review Evidence and Traceability",
        "tab_data": "Empirical Data",
        "tab_theory": "Theoretical Background",
        "source": "Source",
        "similarity": "Similarity",
        "view_full_source": "📄 View Full Source",
        "extracted_content": "Extracted Content",
        "text_not_available": "Extracted text not available for {source}. Reinitialize indices.",
        "bridge_query": "Bridge Query",
        "indices_created": "Indices created. Data: {d} docs, Background: {a} docs.",
        "indices_failed": "Failed to create indices.",
        "indices_reset": "Indices reset. You can upload new files.",
        "processing": "Processing files and indices...",
        "language": "Language",
        "advanced_settings": "⚙️ Advanced Settings",
        "topic_k": "Number of topics (K)",
        "topic_k_help": "Number of thematic clusters for suggestions.",
        "chunk_size": "Chunk Size (tokens)",
        "chunk_overlap": "Chunk Overlap (tokens)",
        "top_k_data": "Top-K Empirical Data",
        "top_k_theory": "Top-K Background",
        "chunk_size_help": "Size of each chunk in tokens. Smaller = more granular.",
        "chunk_overlap_help": "Token overlap between chunks. Higher = less context loss.",
        "top_k_help": "Number of fragments to retrieve per query.",
        "rerank_model_label": "Re-rank Model",
        "rerank_model_help": "VoyageAI model for reranking results.",
        "rerank_top_n_data": "Re-rank Top-N (Data)",
        "rerank_top_n_theory": "Re-rank Top-N (Background)",
        "rerank_top_n_help": "Number of final results after rerank (must be <= Top-K).",
        "rerank_ratio_hint": "Suggestion: keep Top-K about 2–3× larger than Top-N to improve reranking.",
        "settings_note": "⚠️ Reinitialize indices after changing these values.",
        "suggestions_title": "Suggested questions",
        "suggestion_button": "➕ Use suggestion",
        "suggestions_system_prompt": (
            "You are an assistant that generates research question suggestions. "
            "Respond ONLY with valid JSON. Do not include extra text."
        ),
        "suggestions_bootstrap_instructions": (
            "Generate 3-6 suggested questions relevant to the context and topics. "
            "Do not repeat questions. Respond in JSON with 'suggestions'."
        ),
        "suggestions_query_instructions": (
            "Generate 3-6 suggested questions relevant to the user_query and topics. "
            "Do not repeat questions. Respond in JSON with 'suggestions'."
        ),
        "suggestions_constraints": [
            "Do not speak as if you were an interviewee.",
            "Do not use 'your experience' or direct first/second person aimed at the interviewee.",
            "Phrase questions as a researcher (third person)."
        ],
        "citations_title": "📌 Citations & Sources",
        "citation_label": "Citation",
        "citation_missing": "Chunk not available for citation {cite_id}.",
        "citations_parse_error": "Could not parse structured citations. Showing classic format.",
        "citations_system_prompt": "Respond ONLY with valid JSON. Do not include any extra text outside the JSON.",
        "citations_user_instructions": (
            "Use ONLY chunk_id values available in CATALOGO_CHUNKS_JSON. "
            "Each block must cite 1+ citations. "
            "Labels must use titles or file names (no raw IDs). "
            "Respond in JSON with this schema:\n"
            "{\n"
            "  \"answer_blocks\": [{\"id\": \"string\", \"type\": \"paragraph|list\", \"text\": \"string\", \"citations\": [\"c1\"]}],\n"
            "  \"citations\": [{\"id\": \"c1\", \"label\": \"string\", \"chunk_ids\": [\"chunk_id\"]}],\n"
            "  \"sources\": [{\"chunk_id\": \"chunk_id\", \"source_type\": \"data|theory\"}]\n"
            "}"
        ),
        "page_label": "p.",
        "source_type_data": "Data",
        "source_type_theory": "Background",
        "system_prompt": """You are an expert researcher with a PhD in {discipline}.
You are analyzing primary sources from a {perspective} perspective.
Your current research topic is: {topic}.

YOUR OBJECTIVE:
Answer the user's query by analyzing the provided EMPIRICAL DATA.

YOUR SOURCES OF INFORMATION:
1. [EMPIRICAL DATA]: This is the textual evidence you must analyze (news, speeches, interviews, etc.). It is the absolute truth of the case.
2. [BACKGROUND]: Literature and theory to support your interpretation.

RESPONSE INSTRUCTIONS:
- Base your claims primarily on the [EMPIRICAL DATA].
- Use [BACKGROUND] to theorize or provide context to your findings.
- If relevant background exists, include at least one [BACKGROUND] citation.
- If the data shows something different from theory, highlight that tension.
- Do not include inline citations in the text. Citations are provided via JSON only.""",
        "bridge_prompt": """TASK: Generate keywords to search for relevant academic literature.

CONTEXT:
- User question: "{query}"
- Empirical data found: {context}

INSTRUCTIONS:
1. Extract 5-10 key concepts from the data that are relevant to the question.
2. The concepts should be academic/theoretical terms searchable in scientific literature.
3. Respond ONLY with concepts separated by commas.
4. Do NOT include explanations, just the list of concepts.

RESPONSE FORMAT:
concept1, concept2, concept3, concept4, concept5

RESPONSE:"""
    }
}

def get_text(key, lang="es", **kwargs):
    """Get translated text for the given key."""
    text = I18N.get(lang, I18N["es"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text

# Initialize Directories
os.makedirs(DATOS_DIR, exist_ok=True)
os.makedirs(ANTECEDENTES_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(DEV_LOGS_DIR, exist_ok=True)

def _append_log(log_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_topic_event(message):
    _append_log(os.path.join(DEBUG_DIR, "topic_log.txt"), message)


def log_rerank_event(message):
    _append_log(RERANK_LOG_PATH, message)

def log_bug_event(message):
    _append_log(os.path.join(DEBUG_DIR, "bug_log.txt"), message)

def setup_logging():
    log_path = os.path.join(DEBUG_DIR, "bug_log.txt")
    logger = logging.getLogger()
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == os.path.abspath(log_path)
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        log_bug_event(f"{category.__name__}: {message} ({filename}:{lineno})")

    warnings.showwarning = _showwarning

    def _excepthook(exc_type, exc, tb):
        log_bug_event(f"Uncaught {exc_type.__name__}: {exc}")
        log_bug_event("".join(traceback.format_tb(tb)))

    sys.excepthook = _excepthook

def init_dev_session_log():
    today = time.strftime("%Y-%m-%d")
    log_path = os.path.join(DEV_LOGS_DIR, f"{today}.md")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Avances de Desarrollo - {today}\n\n")
            f.write("## Features implementadas\n- \n\n")
            f.write("## Por hacer\n- \n")
    return log_path

def append_dev_session_entry(feature, done_items=None, todo_items=None):
    log_path = init_dev_session_log()
    done_items = done_items or []
    todo_items = todo_items or []
    with open(log_path, "a", encoding="utf-8") as f:
        if feature:
            f.write(f"\n### {feature}\n")
        if done_items:
            f.write("**Features implementadas**\n")
            for item in done_items:
                f.write(f"- {item}\n")
        if todo_items:
            f.write("**Por hacer**\n")
            for item in todo_items:
                f.write(f"- {item}\n")

init_dev_session_log()
setup_logging()

SPANISH_STOP_WORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
    "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu",
    "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía",
    "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos",
    "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
    "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos",
    "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "estaré",
    "estarás", "estará", "estaremos", "estaréis", "estarán"
}

DEFAULT_STOP_WORD_LANG = "spanish"

def get_stop_words(lang):
    if lang == "en":
        base = get_lang_stop_words("english")
    else:
        base = get_lang_stop_words(DEFAULT_STOP_WORD_LANG)
    return list(set(base + list(SPANISH_STOP_WORDS)))

try:
    voyage_key = os.getenv("VOYAGE_KEY")
    if not voyage_key:
        st.warning(get_text("voyage_key_missing", "es"))
        log_bug_event("VOYAGE_KEY missing; embeddings may fail.")

    primary_model = "voyage-3-large"
    fallback_model = "voyage-3.5"
    fallback_model_lite = "voyage-3.5-lite"

    try:
        Settings.embed_model = VoyageEmbedding(
            model_name=primary_model,
            voyage_api_key=voyage_key,
            truncation=True,
            embed_batch_size=256
        )
    except Exception as primary_error:
        log_bug_event(f"Primary embedding init failed: {primary_error}")
        log_bug_event(traceback.format_exc())
        try:
            Settings.embed_model = VoyageEmbedding(
                model_name=fallback_model,
                voyage_api_key=voyage_key,
                truncation=True,
                embed_batch_size=256
            )
        except Exception as fallback_error:
            log_bug_event(f"Fallback embedding init failed: {fallback_error}")
            log_bug_event(traceback.format_exc())
            Settings.embed_model = VoyageEmbedding(
                model_name=fallback_model_lite,
                voyage_api_key=voyage_key,
                truncation=True,
                embed_batch_size=256
            )
except Exception as e:
    log_bug_event(f"Embedding model init failed: {e}")
    log_bug_event(traceback.format_exc())
    st.error(get_text("embedding_init_error", "es", error=e))

# LlamaIndex Settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

# --- Helper Functions ---

def normalize_extracted_text(text):
    """
    [FIX 2] Normalize PDF-extracted text that has broken lines.
    Join lines that don't end with sentence-ending punctuation.
    """
    if not text:
        return ""
    
    # Replace single newlines with spaces (but keep double newlines as paragraph breaks)
    # First, normalize multiple newlines to a marker
    text = re.sub(r'\n{2,}', '<<<PARA>>>', text)
    # Replace single newlines with spaces
    text = re.sub(r'\n', ' ', text)
    # Restore paragraph breaks
    text = text.replace('<<<PARA>>>', '\n\n')
    # Clean up extra spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def reset_directories():
    """Clear temp directories for fresh start."""
    for folder in [DATOS_DIR, ANTECEDENTES_DIR, EXTRACTED_TEXT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

def save_uploaded_files(uploaded_files, target_dir):
    """Save Streamlit uploaded files to disk."""
    saved_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
    return saved_files

def format_text_for_display(text):
    """Convert paragraph breaks to HTML for proper rendering."""
    if not text:
        return ""
    # Normalize text first
    text = normalize_extracted_text(text)
    # Convert paragraph breaks to <br><br>
    return text.replace("\n\n", "<br><br>").replace("\n", "<br>")

def extract_topic_keywords(documents, topic_ids, lang, top_n=8):
    """Compute top keywords per topic using TF-IDF over topic-concatenated texts."""
    topic_texts = defaultdict(list)
    for doc_text, topic_id in zip(documents, topic_ids):
        if doc_text:
            topic_texts[topic_id].append(doc_text)

    topic_keywords = {}
    for topic_id, texts in topic_texts.items():
        corpus = ["\n".join(texts)]
        vectorizer = TfidfVectorizer(
            stop_words=get_stop_words(lang),
            ngram_range=(1, 3),
            max_features=4000
        )
        tfidf = vectorizer.fit_transform(corpus)
        if tfidf.shape[1] == 0:
            topic_keywords[topic_id] = []
            continue
        scores = tfidf.toarray()[0]
        terms = np.array(vectorizer.get_feature_names_out())
        top_indices = scores.argsort()[::-1][:top_n]
        topic_keywords[topic_id] = [terms[i] for i in top_indices if scores[i] > 0]
    return topic_keywords

def build_topics_for_collection(chroma_collection, topic_k, lang):
    """Cluster embeddings with K-means and persist topic metadata in Chroma."""
    payload = chroma_collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = payload.get("embeddings", None)
    ids = payload.get("ids", None)
    documents = payload.get("documents", None)
    metadatas = payload.get("metadatas", None)

    embeddings = embeddings if embeddings is not None else []
    ids = ids if ids is not None else []
    documents = documents if documents is not None else []
    metadatas = metadatas if metadatas is not None else []

    if len(embeddings) < 2:
        return 0

    k = min(topic_k, len(embeddings))
    if k < 2:
        return 0

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    topic_ids = kmeans.fit_predict(np.array(embeddings))

    topic_keywords = extract_topic_keywords(documents, topic_ids, lang)
    updated_metadatas = []
    for metadata, topic_id in zip(metadatas, topic_ids):
        new_metadata = dict(metadata or {})
        new_metadata["topic_id"] = int(topic_id)
        keywords = topic_keywords.get(int(topic_id), [])
        new_metadata["topic_keywords"] = ", ".join(keywords)
        if "topic_label" not in new_metadata:
            new_metadata["topic_label"] = ""
        updated_metadatas.append(new_metadata)

    chroma_collection.update(ids=ids, metadatas=updated_metadatas)

    topic_counts = Counter(topic_ids)
    top_topics = topic_counts.most_common(5)
    keyword_preview = {
        int(topic_id): topic_keywords.get(int(topic_id), [])[:3]
        for topic_id, _ in top_topics
    }
    log_topic_event(
        "Topics built for collection "
        f"{chroma_collection.name} with k={k} on {len(ids)} docs. "
        f"Top topics: {top_topics}. Keywords: {keyword_preview}"
    )
    return k

def parse_json_response(text):
    """Parse JSON from model response with basic recovery."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def chunk_meta_from_node(node):
    """Normalize node metadata for citations."""
    metadata = node.metadata or {}
    node_id = getattr(node, "node_id", None)
    if node_id is None and hasattr(node, "node"):
        node_id = getattr(node.node, "node_id", None)
    return {
        "chunk_id": node_id,
        "doc_id": metadata.get("doc_id") or metadata.get("file_name"),
        "title": metadata.get("title") or metadata.get("file_name"),
        "page": metadata.get("page"),
        "start_char": metadata.get("start_char"),
        "end_char": metadata.get("end_char"),
    }


def extract_chunk_evidence(nodes):
    """Build chunk evidence list from nodes."""
    evidence = []
    for node in nodes:
        text = normalize_extracted_text(node.text)
        snippet = text[:600] if text else ""
        meta = chunk_meta_from_node(node)
        meta["text"] = text
        meta["text_snippet"] = snippet
        evidence.append(meta)
    return evidence


def clip_text(text, limit=800):
    if not text:
        return ""
    return text[:limit]


def format_nodes_for_log(nodes, max_snippet=240):
    lines = []
    for idx, node in enumerate(nodes, start=1):
        meta = node.metadata or {}
        node_id = getattr(node, "node_id", None)
        if node_id is None and hasattr(node, "node"):
            node_id = getattr(node.node, "node_id", None)
        file_name = meta.get("file_name", "Unknown")
        score = node.score if node.score is not None else 0.0
        snippet = normalize_extracted_text(node.text or "")[:max_snippet]
        lines.append(
            f"{idx}. score={score:.4f} source={file_name} chunk_id={node_id} | {snippet}"
        )
    return "\n".join(lines)


def validate_citation_payload(payload, allowed_chunk_ids):
    """Validate structured citation payload against retrieved chunks."""
    if not isinstance(payload, dict):
        return None
    answer_blocks = payload.get("answer_blocks")
    citations = payload.get("citations")
    sources = payload.get("sources")
    if not isinstance(answer_blocks, list) or not isinstance(citations, list):
        return None

    citation_map = {}
    for item in citations:
        cite_id = item.get("id")
        chunk_ids = item.get("chunk_ids") or []
        if not cite_id or not isinstance(chunk_ids, list):
            continue
        valid_chunks = [cid for cid in chunk_ids if cid in allowed_chunk_ids]
        if not valid_chunks:
            continue
        citation_map[cite_id] = {
            "chunk_ids": valid_chunks,
            "label": item.get("label") or str(cite_id)
        }

    cleaned_blocks = []
    for block in answer_blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        block_cites = [cid for cid in (block.get("citations") or []) if cid in citation_map]
        if not block_cites:
            continue
        cleaned_blocks.append({
            "id": block.get("id") or f"b{len(cleaned_blocks)+1}",
            "text": text,
            "citations": block_cites,
            "type": block.get("type") or "paragraph"
        })

    if not cleaned_blocks:
        return None

    if not isinstance(sources, list):
        sources = []

    return {
        "answer_blocks": cleaned_blocks,
        "citations": citation_map,
        "sources": sources
    }

def update_topic_labels(chroma_collection, topic_id_to_label):
    """Persist generated topic labels to all docs in the topic."""
    for topic_id, topic_label in topic_id_to_label.items():
        if not topic_label:
            continue
        results = chroma_collection.get(
            where={"topic_id": int(topic_id)},
            include=["metadatas"]
        )
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        if not ids:
            continue
        updated = []
        for metadata in metadatas:
            new_metadata = dict(metadata or {})
            new_metadata["topic_label"] = topic_label
            updated.append(new_metadata)
        chroma_collection.update(ids=ids, metadatas=updated)

def generate_bootstrap_suggestions(context_text, llm_suggest, chroma_collection, lang):
    """Generate suggestions before any user query using top topics."""
    payload = chroma_collection.get(include=["metadatas", "documents"])
    metadatas = payload.get("metadatas", []) or []
    documents = payload.get("documents", []) or []

    topic_docs = defaultdict(list)
    topic_meta = {}
    for meta, doc in zip(metadatas, documents):
        meta = meta or {}
        topic_id = meta.get("topic_id")
        if topic_id is None:
            continue
        topic_id = int(topic_id)
        topic_docs[topic_id].append(doc)
        if topic_id not in topic_meta:
            topic_meta[topic_id] = {
                "topic_id": topic_id,
                "topic_label": meta.get("topic_label", "").strip(),
                "topic_keywords": meta.get("topic_keywords", "").strip()
            }

    if not topic_docs:
        log_topic_event("Bootstrap suggestions skipped: no topics in collection.")
        return []

    top_topics = sorted(topic_docs.keys(), key=lambda k: len(topic_docs[k]), reverse=True)[:3]
    topic_payload = []
    for topic_id in top_topics:
        snippets = []
        for doc in topic_docs[topic_id][:2]:
            if doc:
                snippets.append(normalize_extracted_text(doc)[:400])
        meta = topic_meta.get(topic_id, {})
        topic_payload.append({
            "topic_id": topic_id,
            "topic_label": meta.get("topic_label", ""),
            "topic_keywords": meta.get("topic_keywords", ""),
            "snippets": snippets
        })

    system_prompt = get_text("suggestions_system_prompt", lang)
    user_prompt = {
        "task": "bootstrap_suggest",
        "user_query": context_text,
        "topics": topic_payload,
        "instructions": get_text("suggestions_bootstrap_instructions", lang),
        "constraints": get_text("suggestions_constraints", lang),
        "schema": {
            "suggestions": [{
                "id": "string",
                "text": "string",
                "topic_id": "int",
                "topic_label": "string",
                "source_chunk_ids": ["string"]
            }]
        }
    }

    response = llm_suggest.complete(json.dumps(user_prompt, ensure_ascii=False))
    response_text = response.text.strip()
    payload = parse_json_response(response_text)
    if not payload:
        log_bug_event("Bootstrap suggestion JSON parse failed.")
        log_bug_event(f"Bootstrap raw response: {response_text[:2000]}")
        return []

    suggestions = payload.get("suggestions", [])
    log_topic_event(
        f"Bootstrap suggestion raw count={len(suggestions)}; keys={list(payload.keys())}"
    )
    log_topic_event(f"Bootstrap suggestions generated: {len(suggestions)}")
    normalized = []
    for item in suggestions:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        normalized.append({
            "id": item.get("id") or f"s{len(normalized)+1}",
            "text": text,
            "topic_id": item.get("topic_id"),
            "topic_label": item.get("topic_label"),
            "source_chunk_ids": item.get("source_chunk_ids", [])
        })
    return normalized

def generate_topic_labels(chroma_collection, llm_label, lang, top_k=3):
    """Generate labels for all topics using top-k chunks per topic."""
    payload = chroma_collection.get(include=["metadatas", "documents"])
    metadatas = payload.get("metadatas", []) or []
    documents = payload.get("documents", []) or []
    ids = payload.get("ids", []) or []

    topic_to_docs = defaultdict(list)
    topic_to_meta = {}
    for meta, doc, _id in zip(metadatas, documents, ids):
        meta = meta or {}
        topic_id = meta.get("topic_id")
        if topic_id is None:
            continue
        topic_id = int(topic_id)
        topic_to_docs[topic_id].append((doc, _id, meta))
        if topic_id not in topic_to_meta:
            topic_to_meta[topic_id] = {
                "topic_id": topic_id,
                "topic_keywords": meta.get("topic_keywords", "").strip()
            }

    if not topic_to_docs:
        log_topic_event("Label generation skipped: no topics found in collection.")
        return

    topic_payload = []
    for topic_id, items in topic_to_docs.items():
        snippets = []
        for doc, _id, _meta in items[:top_k]:
            if doc:
                snippets.append(normalize_extracted_text(doc)[:400])
        topic_payload.append({
            "topic_id": topic_id,
            "topic_keywords": topic_to_meta.get(topic_id, {}).get("topic_keywords", ""),
            "snippets": snippets
        })

    user_prompt = {
        "task": "label_topics",
        "topics": topic_payload,
        "instructions": (
            "Para cada tópico, genera un nombre general y comprensivo. "
            "Usa keywords y snippets. Responde SOLO en JSON con 'topic_labels'."
        ),
        "schema": {
            "topic_labels": [{"topic_id": "int", "topic_label": "string"}]
        }
    }

    response = llm_label.complete(json.dumps(user_prompt, ensure_ascii=False))
    response_text = response.text.strip()
    payload = parse_json_response(response_text)
    if not payload:
        log_bug_event("Topic label generation returned invalid JSON.")
        log_bug_event(f"Topic label raw response: {response_text[:2000]}")
        return

    topic_labels = payload.get("topic_labels", [])
    labels_map = {
        int(item.get("topic_id")): item.get("topic_label", "").strip()
        for item in topic_labels if item.get("topic_id") is not None
    }
    update_topic_labels(chroma_collection, labels_map)
    if labels_map:
        log_topic_event(f"Updated topic labels: {labels_map}")

def generate_suggestions(user_query, nodes_datos, llm_suggest, chroma_collection, lang):
    """Generate topic-aware query suggestions."""
    topic_counter = Counter()
    topic_data = {}
    topic_to_chunks = defaultdict(list)
    missing_topic_nodes = 0

    for node in nodes_datos:
        metadata = node.metadata or {}
        topic_id = metadata.get("topic_id")
        if topic_id is None:
            node_id = getattr(node, "node_id", None)
            if node_id is None and hasattr(node, "node"):
                node_id = getattr(node.node, "node_id", None)
            if node_id:
                try:
                    fetched = chroma_collection.get(ids=[node_id], include=["metadatas"])
                    fetched_meta = (fetched.get("metadatas") or [])
                    if fetched_meta:
                        metadata = fetched_meta[0] or metadata
                        topic_id = metadata.get("topic_id")
                except Exception as e:
                    log_bug_event(f"Chroma metadata fetch failed for node {node_id}: {e}")
            if topic_id is None:
                missing_topic_nodes += 1
                continue
        topic_id = int(topic_id)
        topic_counter[topic_id] += 1
        topic_to_chunks[topic_id].append(node)
        if topic_id not in topic_data:
            topic_data[topic_id] = {
                "topic_id": topic_id,
                "topic_label": metadata.get("topic_label", "").strip(),
                "topic_keywords": metadata.get("topic_keywords", "").strip()
            }

    if not topic_counter:
        sample_metadata = {}
        if nodes_datos:
            sample_metadata = nodes_datos[0].metadata or {}
        log_topic_event(
            "No topics found in retrieved nodes for suggestions. "
            f"nodes={len(nodes_datos)} missing_topic_nodes={missing_topic_nodes} "
            f"sample_metadata_keys={list(sample_metadata.keys())}"
        )
        return []

    top_topics = [topic_id for topic_id, _ in topic_counter.most_common(3)]
    selected_topics = [topic_data[t] for t in top_topics if t in topic_data]

    topic_payload = []
    for topic in selected_topics:
        samples = topic_to_chunks.get(topic["topic_id"], [])[:2]
        snippets = [normalize_extracted_text(s.text)[:400] for s in samples if s.text]
        topic_payload.append({
            "topic_id": topic["topic_id"],
            "topic_label": topic.get("topic_label", ""),
            "topic_keywords": topic.get("topic_keywords", ""),
            "snippets": snippets
        })

    user_prompt = {
        "task": "suggest_only",
        "user_query": user_query,
        "topics": topic_payload,
        "instructions": get_text("suggestions_query_instructions", lang),
        "constraints": get_text("suggestions_constraints", lang),
        "schema": {
            "suggestions": [{
                "id": "string",
                "text": "string",
                "topic_id": "int",
                "topic_label": "string",
                "source_chunk_ids": ["string"]
            }]
        }
    }

    response = llm_suggest.complete(json.dumps(user_prompt, ensure_ascii=False))
    response_text = response.text.strip()
    payload = parse_json_response(response_text)
    if not payload:
        log_bug_event("Suggestion model returned invalid JSON.")
        log_topic_event("Suggestion generation failed to parse JSON response.")
        log_bug_event(f"Suggestion raw response: {response_text[:2000]}")
        return []

    suggestions = payload.get("suggestions", [])
    log_topic_event(
        f"Suggestion raw count={len(suggestions)}; keys={list(payload.keys())}"
    )
    if not suggestions:
        log_topic_event("Suggestion payload empty; no suggestions returned by model.")
        log_bug_event(f"Suggestion payload empty. Raw response: {response_text[:2000]}")
    log_topic_event(
        f"Suggestions generated: {len(suggestions)} for query='{user_query[:80]}' "
        f"topics={top_topics}"
    )
    normalized = []
    for item in suggestions:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        normalized.append({
            "id": item.get("id") or f"s{len(normalized)+1}",
            "text": text,
            "topic_id": item.get("topic_id"),
            "topic_label": item.get("topic_label"),
            "source_chunk_ids": item.get("source_chunk_ids", [])
        })
    return normalized

def get_chroma_client():
    """Get or create the ChromaDB client stored in session state."""
    if 'chroma_client' not in st.session_state:
        st.session_state['chroma_client'] = chromadb.PersistentClient(path=CHROMA_PATH)
    return st.session_state['chroma_client']

def try_restore_indices():
    """
    Auto-restore indices from existing ChromaDB collections on page load.
    Returns True if indices were restored, False otherwise.
    """
    if 'index_datos' in st.session_state and 'index_antecedentes' in st.session_state:
        return True  # Already loaded
    
    try:
        db = get_chroma_client()
        collections = [c.name for c in db.list_collections()]
        
        # Check if both collections exist
        if "index_datos" not in collections or "index_antecedentes" not in collections:
            return False
        
        # Check if collections have data
        col_datos = db.get_collection("index_datos")
        col_ante = db.get_collection("index_antecedentes")
        
        if col_datos.count() == 0 and col_ante.count() == 0:
            return False  # Empty collections, nothing to restore
        
        # Restore indices from existing collections
        vector_store_datos = ChromaVectorStore(chroma_collection=col_datos)
        index_datos = VectorStoreIndex.from_vector_store(vector_store_datos)
        
        vector_store_antecedentes = ChromaVectorStore(chroma_collection=col_ante)
        index_antecedentes = VectorStoreIndex.from_vector_store(vector_store_antecedentes)
        
        st.session_state['index_datos'] = index_datos
        st.session_state['index_antecedentes'] = index_antecedentes
        
        return True
    except Exception as e:
        log_bug_event(f"Index restore failed: {e}")
        log_bug_event(traceback.format_exc())
        # Silent fail - user can manually initialize
        return False

def initialize_indices():
    """Ingest documents and create/persist vector indices."""
    try:
        # Use chunk settings from session state (or defaults)
        Settings.chunk_size = st.session_state.get("chunk_size", 512)
        Settings.chunk_overlap = st.session_state.get("chunk_overlap", 128)
        
        # Use shared ChromaDB client from session state
        db = get_chroma_client()

        # Delete existing collections if they exist (reset within same session)
        try:
            db.delete_collection("index_datos")
        except:
            pass
        try:
            db.delete_collection("index_antecedentes")
        except:
            pass

        # 1. Index A [DATOS]
        chroma_collection_datos = db.get_or_create_collection("index_datos")
        vector_store_datos = ChromaVectorStore(chroma_collection=chroma_collection_datos)
        storage_context_datos = StorageContext.from_defaults(vector_store=vector_store_datos)
        
        datos_docs = SimpleDirectoryReader(DATOS_DIR).load_data()
        
        # [FIX 2] Save extracted and normalized text for later viewing
        extracted_texts = {}
        for doc in datos_docs:
            file_name = doc.metadata.get('file_name', 'unknown')
            normalized_text = normalize_extracted_text(doc.text)
            if file_name not in extracted_texts:
                extracted_texts[file_name] = ""
            extracted_texts[file_name] += normalized_text + "\n\n---\n\n"
        
        for fname, content in extracted_texts.items():
            safe_name = fname.replace("/", "_").replace("\\", "_")
            txt_path = os.path.join(EXTRACTED_TEXT_DIR, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        if not datos_docs:
            index_datos = VectorStoreIndex.from_documents([], storage_context=storage_context_datos)
        else:
            try:
                index_datos = VectorStoreIndex.from_documents(
                    datos_docs, storage_context=storage_context_datos
                )
            except Exception as e:
                log_bug_event(f"Index build failed for datos: {e}")
                log_bug_event(traceback.format_exc())
                raise

        # 2. Index B [ANTECEDENTES]
        chroma_collection_antecedentes = db.get_or_create_collection("index_antecedentes")
        vector_store_antecedentes = ChromaVectorStore(chroma_collection=chroma_collection_antecedentes)
        storage_context_antecedentes = StorageContext.from_defaults(vector_store=vector_store_antecedentes)

        antecedentes_docs = SimpleDirectoryReader(ANTECEDENTES_DIR).load_data()
        
        for doc in antecedentes_docs:
            file_name = doc.metadata.get('file_name', 'unknown')
            normalized_text = normalize_extracted_text(doc.text)
            if file_name not in extracted_texts:
                extracted_texts[file_name] = ""
            extracted_texts[file_name] += normalized_text + "\n\n---\n\n"
        
        for fname, content in extracted_texts.items():
            safe_name = fname.replace("/", "_").replace("\\", "_")
            txt_path = os.path.join(EXTRACTED_TEXT_DIR, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        if not antecedentes_docs:
            index_antecedentes = VectorStoreIndex.from_documents([], storage_context=storage_context_antecedentes)
        else:
            try:
                index_antecedentes = VectorStoreIndex.from_documents(
                    antecedentes_docs, storage_context=storage_context_antecedentes
                )
            except Exception as e:
                log_bug_event(f"Index build failed for antecedentes: {e}")
                log_bug_event(traceback.format_exc())
                raise
            
        return index_datos, index_antecedentes, len(datos_docs), len(antecedentes_docs)

    except Exception as e:
        log_bug_event(f"Initialize indices failed: {e}")
        log_bug_event(traceback.format_exc())
        st.error(get_text("indices_init_error", lang, error=e))
        return None, None, 0, 0

def reset_indices_in_session():
    """Reset indices via Chroma API to avoid filesystem locks."""
    try:
        db = get_chroma_client()

        # Prefer API reset if available
        if hasattr(db, "reset"):
            try:
                db.reset()
            except Exception:
                pass

        # Delete collections via API
        try:
            db.delete_collection("index_datos")
        except Exception:
            pass
        try:
            db.delete_collection("index_antecedentes")
        except Exception:
            pass

        # Clear session state indices
        if 'index_datos' in st.session_state:
            del st.session_state['index_datos']
        if 'index_antecedentes' in st.session_state:
            del st.session_state['index_antecedentes']

        # Clear temp data directories
        reset_directories()

        return True
    except Exception as e:
        log_bug_event(f"Reset indices failed: {e}")
        log_bug_event(traceback.format_exc())
        st.error(get_text("reset_error", lang, error=e))
        return False

# --- Pipeline Logic ---

def run_sociological_pipeline(user_query, chat_history, index_datos, index_antecedentes, llm_main, llm_bridge, disciplina, perspectiva, tema, lang="es"):

    # Use top_k from session state (or defaults)
    top_k_data = st.session_state.get("top_k_data", 20)
    top_k_theory = st.session_state.get("top_k_theory", 15)
    rerank_top_n_data = st.session_state.get("rerank_top_n_data", min(7, top_k_data))
    rerank_top_n_theory = st.session_state.get("rerank_top_n_theory", min(5, top_k_theory))
    rerank_model = st.session_state.get("rerank_model", "rerank-2.5")

    if rerank_top_n_data > top_k_data:
        rerank_top_n_data = top_k_data
    if rerank_top_n_theory > top_k_theory:
        rerank_top_n_theory = top_k_theory

    def _apply_rerank(nodes, query, reranker, top_n, label):
        if not nodes:
            log_topic_event(f"Rerank skipped ({label}): no nodes.")
            return []
        try:
            reranked = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
            if reranked is None:
                log_bug_event(f"Rerank returned None ({label}).")
                return nodes[:top_n]
            log_topic_event(
                f"Rerank ok ({label}): input={len(nodes)} output={len(reranked)} top_n={top_n}"
            )
            return reranked
        except Exception as e:
            log_bug_event(f"Rerank failed ({label}): {e}")
            log_bug_event(traceback.format_exc())
            return nodes[:top_n]

    retriever_datos = index_datos.as_retriever(similarity_top_k=top_k_data)
    retriever_antecedentes = index_antecedentes.as_retriever(similarity_top_k=top_k_theory)

    # --- Step A: Grounding ---
    nodes_datos = retriever_datos.retrieve(user_query)
    log_rerank_event(
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | Query: {user_query[:120]} ==="
    )
    log_rerank_event(
        "TOP-K datos (pre-rerank):\n" + format_nodes_for_log(nodes_datos)
    )
    reranker_data = VoyageAIRerank(
        model=rerank_model,
        api_key=os.getenv("VOYAGE_KEY"),
        top_n=rerank_top_n_data
    )
    nodes_datos = _apply_rerank(nodes_datos, user_query, reranker_data, rerank_top_n_data, "datos")
    nodes_datos = nodes_datos[:rerank_top_n_data]
    log_rerank_event(
        "TOP-N datos (post-rerank):\n" + format_nodes_for_log(nodes_datos)
    )
    contexto_datos_str = ""
    evidence_datos = []
    
    for node in nodes_datos:
        text = normalize_extracted_text(node.text)
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_datos_str += f"- {text} (Fuente: {file_name})\n"
        evidence_datos.append({"text": text, "source": file_name, "score": score})

    chunk_catalog_datos = extract_chunk_evidence(nodes_datos)

    # --- Step B: Bridging ---
    if not contexto_datos_str:
        contexto_datos_str = "No specific empirical data found."
    
    # Use i18n bridge prompt
    bridge_prompt = get_text("bridge_prompt", lang, query=user_query, context=contexto_datos_str)
    try:
        query_teorica_resp = llm_bridge.complete(bridge_prompt)
        query_teorica = query_teorica_resp.text.strip()
    except Exception as e:
        log_bug_event(f"Bridge LLM call failed: {e}")
        log_bug_event(traceback.format_exc())
        raise
    
    # Debug Logging
    debug_dir = os.path.join(DATA_TEMP_PATH, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "query_log.txt"), "a") as f:
        f.write(f"Query: {user_query} | Bridge: {query_teorica}\n")

    # --- Step C: Theorizing ---
    nodes_antecedentes = retriever_antecedentes.retrieve(query_teorica)
    log_rerank_event(
        f"--- Bridge query (antecedentes): {query_teorica[:160]} ---"
    )
    log_rerank_event(
        "TOP-K antecedentes (pre-rerank):\n" + format_nodes_for_log(nodes_antecedentes)
    )
    reranker_theory = VoyageAIRerank(
        model=rerank_model,
        api_key=os.getenv("VOYAGE_KEY"),
        top_n=rerank_top_n_theory
    )
    nodes_antecedentes = _apply_rerank(
        nodes_antecedentes, query_teorica, reranker_theory, rerank_top_n_theory, "antecedentes"
    )
    nodes_antecedentes = nodes_antecedentes[:rerank_top_n_theory]
    log_rerank_event(
        "TOP-N antecedentes (post-rerank):\n" + format_nodes_for_log(nodes_antecedentes)
    )
    contexto_antecedentes_str = ""
    evidence_antecedentes = []

    for node in nodes_antecedentes:
        text = normalize_extracted_text(node.text)
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_antecedentes_str += f"- {text} (Fuente: {file_name})\n"
        evidence_antecedentes.append({"text": text, "source": file_name, "score": score})

    chunk_catalog_antecedentes = extract_chunk_evidence(nodes_antecedentes)

    # --- Step D: Synthesis ---
    # Use i18n system prompt
    system_content = get_text("system_prompt", lang, discipline=disciplina, perspective=perspectiva, topic=tema)

    user_content_final = (
        f"HISTORIAL CHAT:\n{chat_history}\n\n"
        f"[DATOS EMPÍRICOS]:\n{contexto_datos_str}\n\n"
        f"[ANTECEDENTES] (Búsqueda basada en datos: '{query_teorica}'):\n"
        f"{contexto_antecedentes_str}\n\n"
        f"PREGUNTA USUARIO: {user_query}\n"
    )
    
    # Debug Logging - System Prompt and Final Prompt
    with open(os.path.join(debug_dir, "system_prompt_log.txt"), "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Query: {user_query}\n")
        f.write(f"Language: {lang}\n")
        f.write(f"---\n{system_content}\n\n")
    
    with open(os.path.join(debug_dir, "final_prompt_log.txt"), "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Query: {user_query}\n")
        f.write(f"---\n{user_content_final}\n\n")

    chunk_catalog = []
    for item in chunk_catalog_datos:
        chunk_entry = dict(item)
        chunk_entry["source_type"] = "data"
        chunk_catalog.append(chunk_entry)
    for item in chunk_catalog_antecedentes:
        chunk_entry = dict(item)
        chunk_entry["source_type"] = "theory"
        chunk_catalog.append(chunk_entry)

    chunk_lookup = {item["chunk_id"]: item for item in chunk_catalog if item.get("chunk_id")}

    citations_system = get_text("citations_system_prompt", lang)
    citations_instructions = get_text("citations_user_instructions", lang)

    citations_payload = {
        "task": "answer_with_citations",
        "question": user_query,
        "context": user_content_final,
        "catalog": [
            {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "page": item.get("page"),
                "start_char": item.get("start_char"),
                "end_char": item.get("end_char"),
                "source_type": item.get("source_type"),
                "text_snippet": item.get("text_snippet")
            }
            for item in chunk_catalog
        ],
        "instructions": citations_instructions,
        "constraints": {
            "max_blocks": 4,
            "max_citations_per_block": 2,
            "prefer_snippets": True,
            "prefer_titles": True
        }
    }

    messages = [
        ChatMessage(role="system", content=f"{system_content}\n\n{citations_system}"),
        ChatMessage(role="user", content=json.dumps(citations_payload, ensure_ascii=False))
    ]

    try:
        response = llm_main.chat(messages)
        response_text = response.message.content
    except Exception as e:
        log_bug_event(f"Main LLM call failed: {e}")
        log_bug_event(traceback.format_exc())
        raise
    
    # Debug Logging - Model Response
    with open(os.path.join(debug_dir, "response_log.txt"), "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Query: {user_query}\n")
        f.write(f"---\n{response_text}\n\n")

    structured_payload = parse_json_response(response_text)
    valid_payload = validate_citation_payload(structured_payload, set(chunk_lookup.keys()))

    if valid_payload:
        return valid_payload, evidence_datos, evidence_antecedentes, query_teorica, nodes_datos, chunk_lookup

    return response_text, evidence_datos, evidence_antecedentes, query_teorica, nodes_datos, chunk_lookup

# --- Streamlit UI ---

st.set_page_config(page_title="Conversando con Datos Cualitativos", layout="wide")

# Language selector (stored in session state)
if "lang" not in st.session_state:
    st.session_state.lang = "es"

# Auto-restore indices if they exist from a previous session
indices_restored = try_restore_indices()

# Helper to get current language
lang = st.session_state.lang
t = lambda key, **kwargs: get_text(key, lang, **kwargs)

# Sidebar
with st.sidebar:
    # Language toggle at the top
    col_lang1, col_lang2 = st.columns([1, 1])
    with col_lang1:
        if st.button("🇪🇸 Español", use_container_width=True, type="primary" if lang == "es" else "secondary"):
            st.session_state.lang = "es"
            st.rerun()
    with col_lang2:
        if st.button("🇬🇧 English", use_container_width=True, type="primary" if lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    
    st.divider()
    st.header(t("sidebar_header"))
    
    model_option = st.selectbox(
        t("model_selector"),
        ("moonshotai/kimi-k2-instruct-0905", "openai/gpt-oss-120b")
    )
    
    disciplina = st.text_input(t("discipline"), "Sociología" if lang == "es" else "Sociology")
    perspectiva = st.text_input(t("perspective"), "Feminismo" if lang == "es" else "Feminism")
    tema = st.text_input(t("topic"), "Experiencias sobre maternidad" if lang == "es" else "Motherhood experiences")

    # Advanced Settings Expander
    with st.expander(t("advanced_settings"), expanded=False):
        st.caption(t("settings_note"))
        
        # Initialize session state for settings if not exists
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 512
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 128
        if "top_k_data" not in st.session_state:
            st.session_state.top_k_data = 20
        if "top_k_theory" not in st.session_state:
            st.session_state.top_k_theory = 15
        if "topic_k" not in st.session_state:
            st.session_state.topic_k = 6
        if "rerank_model" not in st.session_state:
            st.session_state.rerank_model = "rerank-2.5"
        if "rerank_top_n_data" not in st.session_state:
            st.session_state.rerank_top_n_data = 7
        if "rerank_top_n_theory" not in st.session_state:
            st.session_state.rerank_top_n_theory = 5

        st.session_state.topic_k = st.slider(
            t("topic_k"),
            min_value=2, max_value=20,
            value=st.session_state.topic_k,
            help=t("topic_k_help")
        )
        st.session_state.chunk_size = st.slider(
            t("chunk_size"), 
            min_value=256, max_value=2048, 
            value=st.session_state.chunk_size, 
            step=128,
            help=t("chunk_size_help")
        )
        st.session_state.chunk_overlap = st.slider(
            t("chunk_overlap"), 
            min_value=64, max_value=512, 
            value=st.session_state.chunk_overlap, 
            step=64,
            help=t("chunk_overlap_help")
        )
        st.session_state.top_k_data = st.slider(
            t("top_k_data"), 
            min_value=3, max_value=30, 
            value=st.session_state.top_k_data,
            help=t("top_k_help")
        )
        st.session_state.top_k_theory = st.slider(
            t("top_k_theory"), 
            min_value=3, max_value=30, 
            value=st.session_state.top_k_theory,
            help=t("top_k_help")
        )
        st.session_state.rerank_model = st.selectbox(
            t("rerank_model_label"),
            ("rerank-2.5", "rerank-2.5-lite", "rerank-2", "rerank-2-lite"),
            index=("rerank-2.5", "rerank-2.5-lite", "rerank-2", "rerank-2-lite").index(
                st.session_state.rerank_model
            ),
            help=t("rerank_model_help")
        )
        st.caption(t("rerank_ratio_hint"))
        st.session_state.rerank_top_n_data = st.slider(
            t("rerank_top_n_data"),
            min_value=1, max_value=15,
            value=st.session_state.rerank_top_n_data,
            help=t("rerank_top_n_help")
        )
        st.session_state.rerank_top_n_theory = st.slider(
            t("rerank_top_n_theory"),
            min_value=1, max_value=15,
            value=st.session_state.rerank_top_n_theory,
            help=t("rerank_top_n_help")
        )

    st.subheader(t("file_upload"))
    uploaded_datos = st.file_uploader(t("empirical_data"), accept_multiple_files=True, key="uploader_datos")
    uploaded_antecedentes = st.file_uploader(t("literature"), accept_multiple_files=True, key="uploader_antecedentes")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("init_indices")):
            if not os.environ.get("GROQ_KEY"):
                st.error(t("api_key_error"))
            else:
                with st.spinner(t("processing")):
                    reset_directories()
                    save_uploaded_files(uploaded_datos, DATOS_DIR)
                    save_uploaded_files(uploaded_antecedentes, ANTECEDENTES_DIR)
                    
                    index_datos, index_antecedentes, count_d, count_a = initialize_indices()
                    
                    if index_datos and index_antecedentes:
                        st.session_state['index_datos'] = index_datos
                        st.session_state['index_antecedentes'] = index_antecedentes
                        db = get_chroma_client()
                        try:
                            collection_datos = db.get_collection("index_datos")
                            build_topics_for_collection(
                                collection_datos,
                                st.session_state.get("topic_k", 6),
                                lang
                            )
                            llm_label = Groq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_KEY"))
                            generate_topic_labels(
                                collection_datos,
                                llm_label=llm_label,
                                lang=lang,
                                top_k=3
                            )
                            # Refresh index to include updated metadata after topic build
                            refreshed_store = ChromaVectorStore(chroma_collection=collection_datos)
                            st.session_state["index_datos"] = VectorStoreIndex.from_vector_store(
                                refreshed_store
                            )
                        except Exception as e:
                            log_bug_event(f"Topic build failed: {e}")
                            st.warning(get_text("topic_build_error", lang, error=e))
                        st.success(t("indices_created", d=count_d, a=count_a))
                    else:
                        st.error(t("indices_failed"))
    
    with col2:
        if st.button(t("reset_all")):
            if reset_indices_in_session():
                st.session_state.messages = []
                st.success(t("indices_reset"))
                st.rerun()

    if st.button(t("clear_chat")):
        st.session_state.messages = []
        st.rerun()

    # Source Browser
    st.subheader(t("explore_sources"))
    
    if os.path.exists(EXTRACTED_TEXT_DIR):
        txt_files = [f for f in os.listdir(EXTRACTED_TEXT_DIR) if f.endswith('.txt')]
        if txt_files:
            for txt_file in txt_files:
                original_name = txt_file.replace('.txt', '')
                with st.expander(f"📄 {original_name}"):
                    txt_path = os.path.join(EXTRACTED_TEXT_DIR, txt_file)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.text_area(t("content"), content, height=200, disabled=True, key=f"browse_{txt_file}")
        else:
            st.caption(t("no_sources"))
    else:
        st.caption(t("no_sources"))

# Main Interface
st.title(t("main_title"))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []
if "hide_suggestions" not in st.session_state:
    st.session_state.hide_suggestions = False

if 'index_datos' not in st.session_state or 'index_antecedentes' not in st.session_state:
    st.warning(t("indices_not_ready"))

if not st.session_state.messages and not st.session_state.pending_query:
    try:
        if "index_datos" in st.session_state:
            db = get_chroma_client()
            collection_datos = db.get_collection("index_datos")
            llm_bootstrap = Groq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_KEY"))
            bootstrap_context = f"{tema} | {perspectiva} | {disciplina}"
            st.session_state.last_suggestions = generate_bootstrap_suggestions(
                bootstrap_context,
                llm_bootstrap,
                collection_datos,
                lang
            )
        else:
            log_topic_event(t("bootstrap_skipped"))
    except Exception as e:
        log_bug_event(f"Bootstrap suggestions failed: {e}")

def set_pending_query(label, source):
    st.session_state.pending_query = label
    st.session_state.pending_query_source = source
    st.session_state.hide_suggestions = True
    st.session_state.last_suggestions = []
    log_topic_event(f"Suggestion clicked ({source}): '{label[:80]}'")

user_input = st.chat_input(t("chat_placeholder"))

pending_used = False
if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    pending_used = True
    log_topic_event(f"Pending query consumed: '{prompt[:80]}'")
else:
    prompt = user_input


def render_evidence_item(item, idx, prefix, lang):
    """Render a single evidence item with proper formatting."""
    st.markdown(f"**{get_text('source', lang)}:** `{item['source']}` | **{get_text('similarity', lang)}:** {item['score']:.4f}")
    
    formatted_text = format_text_for_display(item['text'])
    st.markdown(
        f"<div style='background-color: #262730; padding: 12px; border-radius: 8px; "
        f"margin-bottom: 12px; line-height: 1.6; font-size: 14px;'>{formatted_text}</div>", 
        unsafe_allow_html=True
    )
    
    safe_name = item['source'].replace("/", "_").replace("\\", "_")
    txt_path = os.path.join(EXTRACTED_TEXT_DIR, f"{safe_name}.txt")
    
    with st.expander(f"{get_text('view_full_source', lang)}: {item['source']}", expanded=False):
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                full_content = f.read()
            st.text_area(get_text("extracted_content", lang), full_content, height=300, disabled=True, key=f"ta_{prefix}_{idx}")
        else:
            st.warning(get_text("text_not_available", lang, source=item['source']))
    
    st.divider()


def format_page_label(value, lang):
    if value is None:
        return ""
    try:
        page_num = int(value)
        return f"{get_text('page_label', lang)} {page_num}"
    except (ValueError, TypeError):
        return f"{get_text('page_label', lang)} {value}"


def render_citation_badge(cite_label):
    return (
        "<span style='background:#2b2f3a;color:#dce3f0;"
        "padding:2px 6px;border-radius:6px;font-size:12px;margin-left:4px;'>"
        f"[{cite_label}]</span>"
    )


def build_citation_details(citation_payload, chunk_lookup, lang):
    details = []
    for cite_id, cite_data in citation_payload.items():
        for chunk_id in cite_data["chunk_ids"]:
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                continue
            title = chunk.get("title") or chunk.get("doc_id") or ""
            page = format_page_label(chunk.get("page"), lang)
            source_type = chunk.get("source_type")
            if source_type:
                source_type = get_text(
                    "source_type_data" if source_type == "data" else "source_type_theory",
                    lang
                )
            details.append({
                "cite_id": cite_id,
                "label": cite_data["label"],
                "chunk_id": chunk_id,
                "title": title,
                "page": page,
                "source_type": source_type,
                "text": chunk.get("text")
            })
    return details


def render_citations_inline(citations_payload, chunk_lookup, lang, show_details=True):
    st.subheader(get_text("citations_title", lang))
    details = build_citation_details(citations_payload["citations"], chunk_lookup, lang) if chunk_lookup else []
    for block in citations_payload["answer_blocks"]:
        cite_tags = "".join([
            render_citation_badge(citations_payload["citations"][cid]["label"])
            for cid in block["citations"]
        ])
        block_text = block["text"].replace("\n", "<br/>")
        st.markdown(
            f"<div style='margin-bottom:10px;'>{block_text} {cite_tags}</div>",
            unsafe_allow_html=True
        )

        if not show_details or not chunk_lookup:
            continue

        block_has_details = False
        for item in details:
            if item["cite_id"] in block["citations"]:
                block_has_details = True
                break
        if not block_has_details:
            continue

        with st.expander(get_text("citations_title", lang), expanded=False):
            for item in details:
                if item["cite_id"] not in block["citations"]:
                    continue
                header_bits = [
                    f"{get_text('citation_label', lang)} {item['label']}",
                    item["title"],
                    item.get("page") or ""
                ]
                if item.get("source_type"):
                    header_bits.append(item["source_type"])
                header = " | ".join([bit for bit in header_bits if bit])
                st.markdown(f"**{header}**")
                if item.get("text"):
                    formatted_text = format_text_for_display(item["text"])
                    st.markdown(
                        "<div style='background-color: #262730; padding: 10px; "
                        "border-radius: 8px; margin-bottom: 10px; font-size: 13px; "
                        "line-height: 1.5;'>"
                        f"{formatted_text}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(get_text("citation_missing", lang, cite_id=item["label"]))


# Display Chat History
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("citations"):
            render_citations_inline(message["citations"], {}, lang, show_details=False)
        else:
            st.markdown(message["content"])
        if "evidence" in message:
            with st.expander(t("evidence_title")):
                tab_datos, tab_teoria = st.tabs([t("tab_data"), t("tab_theory")])
                with tab_datos:
                    for i, item in enumerate(message["evidence"]["datos"]):
                        render_evidence_item(item, i, f"hist_{msg_idx}_dato", lang)

                with tab_teoria:
                    if message["evidence"]["query_teorica"]:
                        st.info(f"{t('bridge_query')}: {message['evidence']['query_teorica']}")
                    for i, item in enumerate(message["evidence"]["antecedentes"]):
                        render_evidence_item(item, i, f"hist_{msg_idx}_ante", lang)

# Suggestions (bootstrap / latest)
if st.session_state.last_suggestions and not st.session_state.hide_suggestions:
    st.subheader(t("suggestions_title"))
    for idx, suggestion in enumerate(st.session_state.last_suggestions):
        label = suggestion.get("text") or ""
        if not label:
            continue
        label_hash = hashlib.md5(label.encode("utf-8")).hexdigest()[:8]
        btn_key = f"boot_suggest_{idx}_{label_hash}"
        st.button(
            label,
            key=btn_key,
            use_container_width=True,
            on_click=set_pending_query,
            args=(label, "bootstrap")
        )

if prompt:
    if pending_used:
        st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'index_datos' not in st.session_state or 'index_antecedentes' not in st.session_state:
        st.error(t("init_indices_first"))
    else:
        with st.chat_message("assistant"):
            with st.spinner(t("analyzing")):
                groq_key = os.getenv("GROQ_KEY")
                llm_main = Groq(model=model_option, api_key=groq_key)
                llm_bridge = Groq(model="openai/gpt-oss-20b", api_key=groq_key)
                llm_suggest = Groq(model="openai/gpt-oss-20b", api_key=groq_key)
                
                chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

                try:
                    response_payload, ev_datos, ev_ante, q_teo, nodes_datos, chunk_lookup = run_sociological_pipeline(
                        prompt, 
                        chat_history_str, 
                        st.session_state['index_datos'], 
                        st.session_state['index_antecedentes'],
                        llm_main,
                        llm_bridge,
                        disciplina,
                        perspectiva,
                        tema,
                        lang  # Pass language to pipeline
                    )
                except Exception as e:
                    log_bug_event(f"Pipeline failed: {e}")
                    log_bug_event(traceback.format_exc())
                    st.error(t("pipeline_error"))
                    st.stop()

                suggestions = []
                try:
                    db = get_chroma_client()
                    collection_datos = db.get_collection("index_datos")
                    log_topic_event(f"Generating suggestions for query='{prompt[:80]}' nodes={len(nodes_datos)}")
                    suggestions = generate_suggestions(
                        prompt,
                        nodes_datos=nodes_datos,
                        llm_suggest=llm_suggest,
                        chroma_collection=collection_datos,
                        lang=lang
                    )
                    st.session_state["last_suggestions"] = suggestions
                    st.session_state.hide_suggestions = False
                except Exception as e:
                    log_bug_event(f"Suggestion generation failed: {e}")
                    st.warning(t("suggestions_error", error=e))
                
                if isinstance(response_payload, dict):
                    render_citations_inline(response_payload, chunk_lookup, lang)
                    response_text = "\n\n".join([block["text"] for block in response_payload["answer_blocks"]])
                else:
                    response_text = response_payload
                    st.warning(t("citations_parse_error"))
                    st.markdown(response_text)

                if suggestions:
                    st.subheader(t("suggestions_title"))
                    log_topic_event(f"Rendering {len(suggestions)} suggestions in UI.")
                    for idx, suggestion in enumerate(suggestions):
                        label = suggestion.get("text") or ""
                        if not label:
                            continue
                        label_hash = hashlib.md5(label.encode("utf-8")).hexdigest()[:8]
                        btn_key = f"suggest_{idx}_{label_hash}"
                        st.button(
                            label,
                            key=btn_key,
                            use_container_width=True,
                            on_click=set_pending_query,
                            args=(label, "post-response")
                        )
                else:
                    log_topic_event("No suggestions to render in UI.")
                
                with st.expander(t("evidence_title")):
                    tab_datos, tab_teoria = st.tabs([t("tab_data"), t("tab_theory")])
                    with tab_datos:
                        for i, item in enumerate(ev_datos):
                            render_evidence_item(item, i, "new_dato", lang)

                    with tab_teoria:
                        st.info(f"{t('bridge_query')}: {q_teo}")
                        for i, item in enumerate(ev_ante):
                            render_evidence_item(item, i, "new_ante", lang)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "citations": response_payload if isinstance(response_payload, dict) else None,
                    "evidence": {
                        "datos": ev_datos,
                        "antecedentes": ev_ante,
                        "query_teorica": q_teo
                    }
                })
