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
        # [FIX 1] Use shared ChromaDB client from session state
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

def run_sociological_pipeline(user_query, chat_history, index_datos, index_antecedentes, llm_main, llm_bridge, disciplana, perspectiva, tema):
    
    retriever_datos = index_datos.as_retriever(similarity_top_k=5)
    retriever_antecedentes = index_antecedentes.as_retriever(similarity_top_k=3)

    # --- Step A: Grounding ---
    nodes_datos = retriever_datos.retrieve(user_query)
    contexto_datos_str = ""
    evidence_datos = []
    
    for node in nodes_datos:
        text = normalize_extracted_text(node.text)  # [FIX 2] Normalize here too
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_datos_str += f"- {text} (Fuente: {file_name})\n"
        evidence_datos.append({"text": text, "source": file_name, "score": score})

    # --- Step B: Bridging ---
    if not contexto_datos_str:
        contexto_datos_str = "No specific empirical data found."
    
    bridge_prompt = f"""TAREA: Genera palabras clave para buscar literatura académica relevante.

CONTEXTO:
- Pregunta del usuario: "{user_query}"
- Datos empíricos encontrados: {contexto_datos_str}

INSTRUCCIONES:
1. Extrae 5-10 conceptos clave de los datos que sean relevantes para la pregunta.
2. Los conceptos deben ser términos académicos/teóricos buscables en literatura científica.
3. Responde ÚNICAMENTE con los conceptos separados por comas.
4. NO incluyas explicaciones, solo la lista de conceptos.

FORMATO DE RESPUESTA:
concepto1, concepto2, concepto3, concepto4, concepto5

RESPUESTA:"""
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
        text = normalize_extracted_text(node.text)  # [FIX 2] Normalize here too
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_antecedentes_str += f"- {text} (Fuente: {file_name})\n"
        evidence_antecedentes.append({"text": text, "source": file_name, "score": score})

    # --- Step D: Synthesis ---
    system_content = f"""
    Eres un investigador experto con un Doctorado en {disciplana}.
    Estás analizando fuentes primarias desde una perspectiva {perspectiva}.
    Tu tema de investigación actual es: {tema}.

    TU OBJETIVO:
    Responder a la consulta del usuario analizando los DATOS EMPÍRICOS proporcionados.

    TUS FUENTES DE INFORMACIÓN:
    1. [DATOS EMPÍRICOS]: Esta es la evidencia textual que debes analizar (noticias, discursos, entrevistas, etc.). Es la verdad absoluta del caso.
    2. [ANTECEDENTES]: Es literatura y teoría para apoyar tu interpretación.

    INSTRUCCIONES DE RESPUESTA:
    - Basa tus afirmaciones principalmente en los [DATOS EMPÍRICOS].
    - Usa los [ANTECEDENTES] para teorizar o dar contexto a lo que encuentras en los datos.
    - Si los datos muestran algo distinto a la teoría, destaca esa tensión.
    - Cita siempre las fuentes. Ejemplo: "...se observa un patrón de negación ([Datos]: archivo_noticia.pdf)".
    """

    user_content_final = (
        f"HISTORIAL CHAT:\n{chat_history}\n\n"
        f"[DATOS EMPÍRICOS]:\n{contexto_datos_str}\n\n"
        f"[ANTECEDENTES] (Búsqueda basada en datos: '{query_teorica}'):\n{contexto_antecedentes_str}\n\n"
        f"PREGUNTA USUARIO: {user_query}\n"
    )
    
    messages = [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content_final)
    ]

    response = llm_main.chat(messages)
    
    return response.message.content, evidence_datos, evidence_antecedentes, query_teorica

# --- Streamlit UI ---

st.set_page_config(page_title="Conversando con Datos Cualitativos", layout="wide")

# Auto-restore indices if they exist from a previous session
indices_restored = try_restore_indices()

# Sidebar
with st.sidebar:
    st.header("Configuración del Investigador")
    
    model_option = st.selectbox(
        "Selector de Modelo",
        ("gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3-pro-preview")
    )
    
    disciplina = st.text_input("Disciplina", "Sociología")
    perspectiva = st.text_input("Perspectiva Teórica", "Feminismo")
    tema = st.text_input("Tema de Investigación", "Experiencias sobre maternidad")

    st.subheader("Carga de Archivos")
    uploaded_datos = st.file_uploader("Datos Empíricos / Fuentes Primarias", accept_multiple_files=True, key="uploader_datos")
    uploaded_antecedentes = st.file_uploader("Antecedentes / Literatura", accept_multiple_files=True, key="uploader_antecedentes")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inicializar Indices"):
            if not os.environ.get("GOOGLE_API_KEY"):
                st.error("Please set GOOGLE_API_KEY in .env file")
            else:
                with st.spinner("Procesando archivos e índices..."):
                    reset_directories()
                    save_uploaded_files(uploaded_datos, DATOS_DIR)
                    save_uploaded_files(uploaded_antecedentes, ANTECEDENTES_DIR)
                    
                    index_datos, index_antecedentes, count_d, count_a = initialize_indices()
                    
                    if index_datos and index_antecedentes:
                        st.session_state['index_datos'] = index_datos
                        st.session_state['index_antecedentes'] = index_antecedentes
                        st.success(f"Indices creados. Datos: {count_d} docs, Antecedentes: {count_a} docs.")
                    else:
                        st.error("Fallo al crear índices.")
    
    with col2:
        # [FIX 1] Use proper reset function that doesn't lock DB
        if st.button("Reiniciar Todo"):
            if reset_indices_in_session():
                st.session_state.messages = []
                st.success("Índices reiniciados. Puedes subir nuevos archivos.")
                st.rerun()

    if st.button("Limpiar Conversación"):
        st.session_state.messages = []
        st.rerun()

    # [FIX 3] Source Browser - Show uploaded files
    st.subheader("📂 Explorar Fuentes")
    
    # Check for extracted text files
    if os.path.exists(EXTRACTED_TEXT_DIR):
        txt_files = [f for f in os.listdir(EXTRACTED_TEXT_DIR) if f.endswith('.txt')]
        if txt_files:
            for txt_file in txt_files:
                original_name = txt_file.replace('.txt', '')
                with st.expander(f"📄 {original_name}"):
                    txt_path = os.path.join(EXTRACTED_TEXT_DIR, txt_file)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.text_area("Contenido", content, height=200, disabled=True, key=f"browse_{txt_file}")
        else:
            st.caption("No hay fuentes indexadas aún.")
    else:
        st.caption("No hay fuentes indexadas aún.")

# Main Interface
st.title("Conversando con los Datos")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_evidence_item(item, idx, prefix):
    """Render a single evidence item with proper formatting."""
    st.markdown(f"**Fuente:** `{item['source']}` | **Similitud:** {item['score']:.4f}")
    
    # [FIX 2] Use normalized and formatted text
    formatted_text = format_text_for_display(item['text'])
    st.markdown(
        f"<div style='background-color: #262730; padding: 12px; border-radius: 8px; "
        f"margin-bottom: 12px; line-height: 1.6; font-size: 14px;'>{formatted_text}</div>", 
        unsafe_allow_html=True
    )
    
    # Full source expander
    safe_name = item['source'].replace("/", "_").replace("\\", "_")
    txt_path = os.path.join(EXTRACTED_TEXT_DIR, f"{safe_name}.txt")
    
    with st.expander(f"📄 Ver Fuente Completa: {item['source']}", expanded=False):
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                full_content = f.read()
            st.text_area("Contenido Extraído", full_content, height=300, disabled=True, key=f"ta_{prefix}_{idx}")
        else:
            st.warning(f"Texto extraído no disponible para {item['source']}. Reinicia los índices.")
    
    st.divider()


# Display Chat History
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("🔍 Revisar Evidencia y Trazabilidad"):
                tab_datos, tab_teoria = st.tabs(["Datos Empíricos", "Antecedentes Teóricos"])
                with tab_datos:
                    for i, item in enumerate(message["evidence"]["datos"]):
                        render_evidence_item(item, i, f"hist_{msg_idx}_dato")

                with tab_teoria:
                    if message["evidence"]["query_teorica"]:
                        st.info(f"Query Puente: {message['evidence']['query_teorica']}")
                    for i, item in enumerate(message["evidence"]["antecedentes"]):
                        render_evidence_item(item, i, f"hist_{msg_idx}_ante")

# Chat Input
if prompt := st.chat_input("Escribe tu pregunta de investigación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'index_datos' not in st.session_state or 'index_antecedentes' not in st.session_state:
        st.error("Por favor inicializa los índices primero desde el panel lateral.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos y triangulando con teoría..."):
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
                    tema
                )
                
                st.markdown(response_text)
                
                with st.expander("🔍 Revisar Evidencia y Trazabilidad"):
                    tab_datos, tab_teoria = st.tabs(["Datos Empíricos", "Antecedentes Teóricos"])
                    with tab_datos:
                        for i, item in enumerate(ev_datos):
                            render_evidence_item(item, i, "new_dato")

                    with tab_teoria:
                        st.info(f"Query Puente: {q_teo}")
                        for i, item in enumerate(ev_ante):
                            render_evidence_item(item, i, "new_ante")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "evidence": {
                        "datos": ev_datos,
                        "antecedentes": ev_ante,
                        "query_teorica": q_teo
                    }
                })
