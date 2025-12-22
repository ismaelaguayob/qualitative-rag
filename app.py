import streamlit as st
import os
import shutil
import re
from dotenv import load_dotenv
import chromadb
import time
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CHROMA_PATH = "./chroma_db"
DATA_TEMP_PATH = "./data_temp"
DATOS_DIR = os.path.join(DATA_TEMP_PATH, "datos")
ANTECEDENTES_DIR = os.path.join(DATA_TEMP_PATH, "antecedentes")
EXTRACTED_TEXT_DIR = os.path.join(DATA_TEMP_PATH, "extracted_text")

# --- i18n: Internationalization ---
I18N = {
    "es": {
        "page_title": "Conversando con Datos Cualitativos",
        "sidebar_header": "Configuración del Investigador",
        "model_selector": "Selector de Modelo",
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
        "api_key_error": "Por favor configura GOOGLE_API_KEY en el archivo .env",
        "language": "Idioma",
        "advanced_settings": "⚙️ Ajustes Avanzados",
        "chunk_size": "Tamaño de Chunk (caracteres)",
        "chunk_overlap": "Overlap de Chunk (caracteres)",
        "top_k_data": "Top-K Datos Empíricos",
        "top_k_theory": "Top-K Antecedentes",
        "chunk_size_help": "Tamaño de cada fragmento de texto. Menor = más granular.",
        "chunk_overlap_help": "Solapamiento entre chunks. Mayor = menos pérdida de contexto.",
        "top_k_help": "Cantidad de fragmentos a recuperar por consulta.",
        "settings_note": "⚠️ Reinicializa los índices después de cambiar estos valores.",
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
- Si los datos muestran algo distinto a la teoría, destaca esa tensión.
- Cita siempre las fuentes. Ejemplo: "...se observa un patrón de negación ([Datos]: archivo_noticia.pdf)".""",
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
        "api_key_error": "Please set GOOGLE_API_KEY in the .env file",
        "language": "Language",
        "advanced_settings": "⚙️ Advanced Settings",
        "chunk_size": "Chunk Size (characters)",
        "chunk_overlap": "Chunk Overlap (characters)",
        "top_k_data": "Top-K Empirical Data",
        "top_k_theory": "Top-K Background",
        "chunk_size_help": "Size of each text fragment. Smaller = more granular.",
        "chunk_overlap_help": "Overlap between chunks. Higher = less context loss.",
        "top_k_help": "Number of fragments to retrieve per query.",
        "settings_note": "⚠️ Reinitialize indices after changing these values.",
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
- If the data shows something different from theory, highlight that tension.
- Always cite sources. Example: "...a pattern of denial is observed ([Data]: news_file.pdf)".""",
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

# LlamaIndex Settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

try:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.warning("HF_TOKEN no encontrado en .env. Embeddings pueden fallar.")
    
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="intfloat/multilingual-e5-large",
        token=hf_token,
        timeout=60
    )
except Exception as e:
    st.error(f"Error initializing Embedding Model: {e}")

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
            index_datos = VectorStoreIndex.from_documents(
                datos_docs, storage_context=storage_context_datos
            )

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
             index_antecedentes = VectorStoreIndex.from_documents(
                antecedentes_docs, storage_context=storage_context_antecedentes
            )
            
        return index_datos, index_antecedentes, len(datos_docs), len(antecedentes_docs)

    except Exception as e:
        st.error(f"Error initializing indices: {e}")
        return None, None, 0, 0

def reset_indices_in_session():
    """[FIX 1] Reset indices without deleting the ChromaDB folder."""
    try:
        db = get_chroma_client()
        # Delete collections via API (this doesn't lock the file)
        try:
            db.delete_collection("index_datos")
        except:
            pass
        try:
            db.delete_collection("index_antecedentes")
        except:
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
        st.error(f"Error resetting indices: {e}")
        return False

# --- Pipeline Logic ---

def run_sociological_pipeline(user_query, chat_history, index_datos, index_antecedentes, llm_main, llm_bridge, disciplina, perspectiva, tema, lang="es"):
    
    # Use top_k from session state (or defaults)
    top_k_data = st.session_state.get("top_k_data", 7)
    top_k_theory = st.session_state.get("top_k_theory", 5)
    
    retriever_datos = index_datos.as_retriever(similarity_top_k=top_k_data)
    retriever_antecedentes = index_antecedentes.as_retriever(similarity_top_k=top_k_theory)

    # --- Step A: Grounding ---
    nodes_datos = retriever_datos.retrieve(user_query)
    contexto_datos_str = ""
    evidence_datos = []
    
    for node in nodes_datos:
        text = normalize_extracted_text(node.text)
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_datos_str += f"- {text} (Fuente: {file_name})\n"
        evidence_datos.append({"text": text, "source": file_name, "score": score})

    # --- Step B: Bridging ---
    if not contexto_datos_str:
        contexto_datos_str = "No specific empirical data found."
    
    # Use i18n bridge prompt
    bridge_prompt = get_text("bridge_prompt", lang, query=user_query, context=contexto_datos_str)
    query_teorica_resp = llm_bridge.complete(bridge_prompt)
    query_teorica = query_teorica_resp.text.strip()
    
    # Debug Logging
    debug_dir = os.path.join(DATA_TEMP_PATH, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "query_log.txt"), "a") as f:
        f.write(f"Query: {user_query} | Bridge: {query_teorica}\n")

    # --- Step C: Theorizing ---
    nodes_antecedentes = retriever_antecedentes.retrieve(query_teorica)
    contexto_antecedentes_str = ""
    evidence_antecedentes = []

    for node in nodes_antecedentes:
        text = normalize_extracted_text(node.text)
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_antecedentes_str += f"- {text} (Fuente: {file_name})\n"
        evidence_antecedentes.append({"text": text, "source": file_name, "score": score})

    # --- Step D: Synthesis ---
    # Use i18n system prompt
    system_content = get_text("system_prompt", lang, discipline=disciplina, perspective=perspectiva, topic=tema)

    user_content_final = (
        f"HISTORIAL CHAT:\n{chat_history}\n\n"
        f"[DATOS EMPÍRICOS]:\n{contexto_datos_str}\n\n"
        f"[ANTECEDENTES] (Búsqueda basada en datos: '{query_teorica}'):\n{contexto_antecedentes_str}\n\n"
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
    
    messages = [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content_final)
    ]

    response = llm_main.chat(messages)
    response_text = response.message.content
    
    # Debug Logging - Model Response
    with open(os.path.join(debug_dir, "response_log.txt"), "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Query: {user_query}\n")
        f.write(f"---\n{response_text}\n\n")
    
    return response_text, evidence_datos, evidence_antecedentes, query_teorica

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
        ("gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3-pro-preview")
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
            st.session_state.top_k_data = 7
        if "top_k_theory" not in st.session_state:
            st.session_state.top_k_theory = 5
        
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
            min_value=3, max_value=15, 
            value=st.session_state.top_k_data,
            help=t("top_k_help")
        )
        st.session_state.top_k_theory = st.slider(
            t("top_k_theory"), 
            min_value=3, max_value=15, 
            value=st.session_state.top_k_theory,
            help=t("top_k_help")
        )

    st.subheader(t("file_upload"))
    uploaded_datos = st.file_uploader(t("empirical_data"), accept_multiple_files=True, key="uploader_datos")
    uploaded_antecedentes = st.file_uploader(t("literature"), accept_multiple_files=True, key="uploader_antecedentes")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("init_indices")):
            if not os.environ.get("GOOGLE_API_KEY"):
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


# Display Chat History
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
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

# Chat Input
if prompt := st.chat_input(t("chat_placeholder")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'index_datos' not in st.session_state or 'index_antecedentes' not in st.session_state:
        st.error(t("init_indices_first"))
    else:
        with st.chat_message("assistant"):
            with st.spinner(t("analyzing")):
                llm_main = GoogleGenAI(model=model_option)
                llm_bridge = GoogleGenAI(model="gemma-3-27b-it")
                
                chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

                response_text, ev_datos, ev_ante, q_teo = run_sociological_pipeline(
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
                
                st.markdown(response_text)
                
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
                    "evidence": {
                        "datos": ev_datos,
                        "antecedentes": ev_ante,
                        "query_teorica": q_teo
                    }
                })

