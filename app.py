import streamlit as st
import os
import shutil
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
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding

# Load environment variables
load_dotenv()

# --- Configuration & Constants ---
CHROMA_PATH = "./chroma_db"
DATA_TEMP_PATH = "./data_temp"
DATOS_DIR = os.path.join(DATA_TEMP_PATH, "datos")
ANTECEDENTES_DIR = os.path.join(DATA_TEMP_PATH, "antecedentes")

# Initialize Directories
os.makedirs(DATOS_DIR, exist_ok=True)
os.makedirs(ANTECEDENTES_DIR, exist_ok=True)

# LlamaIndex Settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

try:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.warning("HF_TOKEN no encontrado en .env. Embeddings pueden fallar.")
    
    # Use Hugging Face Serverless Inference API
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="intfloat/multilingual-e5-large",
        token=hf_token,
        timeout=60
    )
except Exception as e:
    st.error(f"Error initializing Embedding Model: {e}")

# --- Helper Functions ---

def reset_directories():
    """Clear temp directories for fresh start."""
    for folder in [DATOS_DIR, ANTECEDENTES_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

def save_uploaded_files(uploaded_files, target_dir):
    """Save Streamlit uploaded files to disk."""
    if not uploaded_files:
        return
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

def initialize_indices():
    """Ingest documents and create/persist vector indices."""
    try:
        # ChromaDB Client
        db = chromadb.PersistentClient(path=CHROMA_PATH)

        # 1. Index A [DATOS]
        chroma_collection_datos = db.get_or_create_collection("index_datos")
        vector_store_datos = ChromaVectorStore(chroma_collection=chroma_collection_datos)
        storage_context_datos = StorageContext.from_defaults(vector_store=vector_store_datos)
        
        datos_docs = SimpleDirectoryReader(DATOS_DIR).load_data()
        if not datos_docs:
             # Create empty index if no docs, to avoid errors
             index_datos = VectorStoreIndex.from_documents([], storage_context=storage_context_datos)
        else:
            # Create index (this will use the safe embedding model)
            index_datos = VectorStoreIndex.from_documents(
                datos_docs, storage_context=storage_context_datos
            )

        # 2. Index B [ANTECEDENTES]
        chroma_collection_antecedentes = db.get_or_create_collection("index_antecedentes")
        vector_store_antecedentes = ChromaVectorStore(chroma_collection=chroma_collection_antecedentes)
        storage_context_antecedentes = StorageContext.from_defaults(vector_store=vector_store_antecedentes)

        antecedentes_docs = SimpleDirectoryReader(ANTECEDENTES_DIR).load_data()
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

# --- Pipeline Logic ---

def run_sociological_pipeline(user_query, chat_history, index_datos, index_antecedentes, llm_main, llm_bridge, disciplana, perspectiva, tema):
    
    # 3. Retrievers
    retriever_datos = index_datos.as_retriever(similarity_top_k=5)
    retriever_antecedentes = index_antecedentes.as_retriever(similarity_top_k=3)

    # --- Step A: Grounding ---
    nodes_datos = retriever_datos.retrieve(user_query)
    contexto_datos_str = ""
    evidence_datos = []
    
    for node in nodes_datos:
        text = node.text
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_datos_str += f"- {text} (Fuente: {file_name})\n"
        evidence_datos.append({"text": text, "source": file_name, "score": score})

    # --- Step B: Bridging ---
    if not contexto_datos_str:
        contexto_datos_str = "No specific empirical data found."
    
    bridge_prompt = (
        f"El usuario preguntó '{user_query}'. En los datos encontramos: '{contexto_datos_str}'. "
        "Genera una frase de búsqueda técnica para encontrar literatura teórica relevante. "
        "Responde SOLO con la frase de búsqueda."
    )
    query_teorica_resp = llm_bridge.complete(bridge_prompt)
    query_teorica = query_teorica_resp.text.strip()

    # --- Step C: Theorizing ---
    nodes_antecedentes = retriever_antecedentes.retrieve(query_teorica)
    contexto_antecedentes_str = ""
    evidence_antecedentes = []

    for node in nodes_antecedentes:
        text = node.text
        file_name = node.metadata.get('file_name', 'Unknown')
        score = node.score if node.score else 0.0
        contexto_antecedentes_str += f"- {text} (Fuente: {file_name})\n"
        evidence_antecedentes.append({"text": text, "source": file_name, "score": score})

    # --- Step D: Synthesis ---
    system_prompt = f"""
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

    final_prompt = (
        f"{system_prompt}\n\n"
        f"HISTORIAL CHAT:\n{chat_history}\n\n"
        f"[DATOS EMPÍRICOS]:\n{contexto_datos_str}\n\n"
        f"[ANTECEDENTES] (Búsqueda basada en datos: '{query_teorica}'):\n{contexto_antecedentes_str}\n\n"
        f"PREGUNTA USUARIO: {user_query}\n"
        "RESPUESTA:"
    )

    response = llm_main.complete(final_prompt)
    
    return response.text, evidence_datos, evidence_antecedentes, query_teorica

# --- Streamlit UI ---

st.set_page_config(page_title="Conversando con los Datos", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Configuración del Investigador")
    
    model_option = st.selectbox(
        "Selector de Modelo",
        ("models/gemini-2.5-flash-lite", "models/gemini-3-flash-preview", "models/gemini-3-pro-preview")
    )
    
    disciplina = st.text_input("Disciplina", "Sociología")
    perspectiva = st.text_input("Perspectiva Teórica", "Teoría Fundamentada")
    tema = st.text_input("Tema de Investigación", "Análisis de Discurso")

    st.subheader("Carga de Archivos")
    uploaded_datos = st.file_uploader("Datos Empíricos / Fuentes Primarias", accept_multiple_files=True, key="uploader_datos")
    uploaded_antecedentes = st.file_uploader("Antecedentes / Literatura", accept_multiple_files=True, key="uploader_antecedentes")
    
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

# Main Interface
st.title("Conversando con los Datos")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("🔍 Revisar Evidencia y Trazabilidad"):
                tab_datos, tab_teoria = st.tabs(["Datos Empíricos", "Antecedentes Teóricos"])
                with tab_datos:
                    for item in message["evidence"]["datos"]:
                        st.markdown(f"**Fuente:** {item['source']} | **Similitud:** {item['score']:.4f}")
                        st.text(item['text'])
                with tab_teoria:
                    if message["evidence"]["query_teorica"]:
                        st.info(f"Query Puente: {message['evidence']['query_teorica']}")
                    for item in message["evidence"]["antecedentes"]:
                        st.markdown(f"**Fuente:** {item['source']} | **Similitud:** {item['score']:.4f}")
                        st.text(item['text'])

# Chat Input
if prompt := st.chat_input("Escribe tu pregunta de investigación..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if indices are ready
    if 'index_datos' not in st.session_state or 'index_antecedentes' not in st.session_state:
        st.error("Por favor inicializa los índices primero desde el panel lateral.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos y triangulando con teoría..."):
                # LLM Setup
                llm_main = Gemini(model_name=model_option)
                llm_bridge = Gemini(model_name="models/gemini-2.5-flash-lite")
                
                # History as string for context
                # Simple concatenation of last few messages
                chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

                # Run Pipeline
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
                
                # Evidence Display (Immediate)
                with st.expander("🔍 Revisar Evidencia y Trazabilidad"):
                    tab_datos, tab_teoria = st.tabs(["Datos Empíricos", "Antecedentes Teóricos"])
                    with tab_datos:
                        for item in ev_datos:
                            st.markdown(f"**Fuente:** {item['source']} | **Similitud:** {item['score']:.4f}")
                            st.text(item['text'])
                    with tab_teoria:
                        st.info(f"Query Puente: {q_teo}")
                        for item in ev_ante:
                            st.markdown(f"**Fuente:** {item['source']} | **Similitud:** {item['score']:.4f}")
                            st.text(item['text'])
                
                # Append to history with evidence
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "evidence": {
                        "datos": ev_datos,
                        "antecedentes": ev_ante,
                        "query_teorica": q_teo
                    }
                })
