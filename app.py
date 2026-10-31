import os
import time
import json
import tempfile
import urllib.request
import streamlit as st

# LangChain components matching RAGTutorial.ipynb
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Optional LLM integration
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="RAG Tutorial - LangChain & FAISS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .response-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper: Detect Ollama & Models
# ---------------------------------------------------------
def check_ollama_status():
    """Checks if local Ollama daemon is reachable and lists models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "StreamlitRAG"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


ollama_online, installed_models = check_ollama_status()


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading embedding model (`all-MiniLM-L6-v2`)...")
def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Step 3 from RAGTutorial: HuggingFaceEmbeddings"""
    return HuggingFaceEmbeddings(model_name=model_name)


def load_raw_documents(source_option: str, uploaded_file):
    """Step 1 from RAGTutorial: Data loading"""
    if source_option == "Upload custom document (.txt, .pdf)":
        if uploaded_file is None:
            return None, "Please upload a document (.txt or .pdf) using the file uploader above to proceed."
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # If valid PDF signature
        if file_name.lower().endswith(".pdf") and file_bytes.startswith(b"%PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return docs, None
        else:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="replace")
            docs = [Document(page_content=text, metadata={"source": file_name})]
            return docs, None
    else:
        file_path = source_option
        if not os.path.exists(file_path):
            return None, f"File `{file_path}` not found in workspace."
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(file_path, encoding="latin-1")
            docs = loader.load()
        return docs, None


@st.cache_resource(show_spinner="⏳ Indexing document into FAISS vector database...")
def build_vector_store(doc_identifier: str, doc_content_hash: int, chunk_size: int, chunk_overlap: int, _docs, _embeddings):
    """
    Step 2 & Step 4 from RAGTutorial:
    - Text splitting with RecursiveCharacterTextSplitter
    - Embedding and vector store in FAISS DB
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(_docs)
    vector_db = FAISS.from_documents(chunks, _embeddings)
    return vector_db, chunks


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

def set_query(text: str):
    st.session_state["query_text"] = text
    st.session_state["trigger_search"] = True


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ RAG Settings")

    # Document Selector (Step 1)
    available_files = [f for f in ["CPP.txt", "Transformer.txt"] if os.path.exists(f)]
    options = available_files + ["Upload custom document (.txt, .pdf)"]

    selected_source = st.selectbox(
        "📄 Select Document Source",
        options=options,
        index=0,
        help="Matches Step 1: TextLoader('CPP.txt', encoding='utf-8')"
    )

    uploaded_file = None
    if selected_source == "Upload custom document (.txt, .pdf)":
        uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf"])

    st.divider()

    # Step 2: Splitter Settings
    st.subheader("✂️ Chunking Parameters")
    chunk_size = st.slider("Chunk Size", 50, 1000, 200, 25, help="Notebook default: 200 characters")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 20, 5, help="Notebook default: 20 characters overlap")

    st.divider()

    # Step 5: Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 4)

    st.divider()

    # Step 6: LLM Generation
    st.subheader("🤖 LLM Answer Generation")
    
    # Status indicator
    if ollama_online:
        st.success("🟢 Ollama Connected (localhost:11434)")
        default_model = "gemma2:2b" if "gemma2:2b" in installed_models else (installed_models[0] if installed_models else "gemma2:2b")
        selected_model = st.selectbox("Select Ollama Model", options=installed_models or ["gemma2:2b"], index=0)
    else:
        st.warning("🟡 Ollama Offline (Run `ollama serve` in terminal)")
        selected_model = st.text_input("Ollama Model", value="gemma2:2b")

    enable_llm = st.checkbox("Generate synthesized answer with LLM", value=True)

    st.divider()
    if st.button("🔄 Reset & Rebuild Index", use_container_width=True):
        st.cache_resource.clear()
        st.session_state["last_result"] = None
        st.success("Cache cleared!")
        st.rerun()


# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">📚 Retrieval-Augmented Generation (RAG)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Streamlit interface converted from <code>RAGTutorial.ipynb</code> with LangChain, FAISS vector search, and Ollama response generation.</div>', unsafe_allow_html=True)

# Load Embedding Model
embeddings = get_embeddings(model_name="all-MiniLM-L6-v2")

# Load Document
docs, error_msg = load_raw_documents(selected_source, uploaded_file)

if error_msg:
    st.info(error_msg)
    st.stop()

if not docs or len(docs) == 0:
    st.warning("No text content found in the selected document.")
    st.stop()

total_text = "".join([d.page_content for d in docs])
doc_content_hash = hash(total_text)
doc_name = uploaded_file.name if uploaded_file else selected_source

# Build FAISS Vector Store
vector_db, final_documents = build_vector_store(
    doc_identifier=doc_name,
    doc_content_hash=doc_content_hash,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    _docs=docs,
    _embeddings=embeddings
)

# Overview Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Source Document", doc_name)
with m2:
    st.metric("Document Characters", f"{len(total_text):,}")
with m3:
    st.metric("FAISS Chunks Created", f"{len(final_documents):,}")
with m4:
    st.metric("LLM Status", f"Ollama ({selected_model})" if (enable_llm and ollama_online) else "Similarity Search Only")

st.write("")

# Navigation Tabs
tab_qa, tab_chunks, tab_pipeline = st.tabs(["🔍 Ask & Generate Answer", "📑 Chunks Explorer", "📖 Pipeline Architecture"])

# ---------------------------------------------------------
# TAB 1: Search & Q&A
# ---------------------------------------------------------
with tab_qa:
    st.subheader("Ask a Question")

    # Sample query prompts for convenience
    sample_queries = {
        "CPP.txt": [
            "What is C++?",
            "Who developed C++ and when?",
            "What are the main features of C++?",
            "What is operator overloading?",
            "Explain virtual functions in C++"
        ],
        "Transformer.txt": [
            "What is the Transformer architecture?",
            "How does self-attention work?",
            "What is positional encoding?",
            "What are multi-head attention advantages?"
        ]
    }

    if selected_source in sample_queries:
        st.caption("Click any suggestion to ask immediately:")
        cols = st.columns(len(sample_queries[selected_source]))
        for i, q in enumerate(sample_queries[selected_source]):
            cols[i].button(q, key=f"quick_btn_{i}", on_click=set_query, args=(q,), use_container_width=True)

    # Input Form
    with st.form("rag_form", clear_on_submit=False):
        user_input = st.text_input(
            "Enter your question:",
            value=st.session_state.get("query_text", ""),
            placeholder="e.g. Who developed C++ and when?",
            key="rag_input_box"
        )
        col_btn1, col_btn2 = st.columns([5, 1])
        submitted = col_btn1.form_submit_button("🚀 Ask & Generate Answer", type="primary", use_container_width=True)
        cleared = col_btn2.form_submit_button("Clear", use_container_width=True)

    if cleared:
        st.session_state["query_text"] = ""
        st.session_state["trigger_search"] = False
        st.session_state["last_result"] = None
        st.rerun()

    # Determine query to execute
    query_to_run = None
    if submitted and user_input.strip():
        query_to_run = user_input.strip()
        st.session_state["query_text"] = query_to_run
        st.session_state["trigger_search"] = False
    elif st.session_state.get("trigger_search", False) and st.session_state.get("query_text", "").strip():
        query_to_run = st.session_state["query_text"].strip()
        st.session_state["trigger_search"] = False

    # Process Query
    if query_to_run:
        with st.spinner("🔍 Performing FAISS similarity search..."):
            docs_and_scores = vector_db.similarity_search_with_score(query_to_run, k=top_k)

        context_str = "\n\n---\n\n".join([doc.page_content for doc, _ in docs_and_scores])
        answer_text = None

        if enable_llm:
            if not OLLAMA_AVAILABLE:
                st.error("`langchain-ollama` is not installed or importable.")
            elif not ollama_online:
                st.warning("⚠️ Local Ollama service is not reachable at `http://localhost:11434`. Please start Ollama (`ollama serve`) to generate synthesized answers.")
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are a helpful and knowledgeable assistant. "
                        "Use the following retrieved context from the document to answer the user's question clearly and accurately. "
                        "If the answer is not contained in the context, state that the document does not contain this information.\n\n"
                        "Context:\n{context}"
                    )),
                    ("user", "{question}"),
                ])
                llm = OllamaLLM(model=selected_model)
                chain = prompt | llm | StrOutputParser()

                st.markdown("### 🤖 Synthesized Answer")
                answer_container = st.container()
                with answer_container:
                    with st.spinner(f"Generating response using `{selected_model}`..."):
                        try:
                            # Stream answer live to UI
                            def stream_generator():
                                for chunk in chain.stream({"context": context_str, "question": query_to_run}):
                                    yield chunk

                            answer_text = st.write_stream(stream_generator())
                        except Exception as e:
                            st.error(f"Error during Ollama generation: {e}")
                            answer_text = None

        # Store in session state so interaction with expanders doesn't wipe results
        st.session_state["last_result"] = {
            "query": query_to_run,
            "answer": answer_text,
            "docs_and_scores": docs_and_scores,
            "model_used": selected_model if (enable_llm and ollama_online) else None
        }

    # Render previous or current result
    last_res = st.session_state.get("last_result")
    if last_res:
        # If not freshly streamed in this run, display previously generated answer
        if not query_to_run and last_res.get("answer"):
            st.markdown("### 🤖 Synthesized Answer")
            st.info(last_res["answer"])
            if last_res.get("model_used"):
                st.caption(f"Generated by `{last_res['model_used']}` for query: *'{last_res['query']}'*")

        st.markdown(f"### 📄 Retrieved Context Chunks (`k={len(last_res['docs_and_scores'])}`)")
        st.caption("Chunks retrieved from FAISS ordered by semantic relevance (lower L2 distance = closer match):")

        for rank, (doc, score) in enumerate(last_res["docs_and_scores"], start=1):
            with st.expander(f"📌 Chunk #{rank} | FAISS Distance: {score:.4f}", expanded=(rank == 1)):
                st.markdown(f"```text\n{doc.page_content}\n```")
                if doc.metadata:
                    st.caption(f"**Metadata:** `{doc.metadata}`")

    elif not query_to_run:
        st.write("")
        st.info("👆 Enter a question above or click one of the quick suggestions to search the document and generate an answer!")


# ---------------------------------------------------------
# TAB 2: Chunks Explorer
# ---------------------------------------------------------
with tab_chunks:
    st.subheader("📑 Document Splitter Explorer")
    st.markdown(
        f"The raw document was split into **{len(final_documents)} chunks** using `RecursiveCharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})`."
    )

    search_chunk = st.text_input("Filter chunks by keyword:", placeholder="Type to filter chunks...", key="chunk_filter")
    filtered_chunks = [
        (idx, c) for idx, c in enumerate(final_documents)
        if (not search_chunk) or (search_chunk.lower() in c.page_content.lower())
    ]

    st.write(f"Showing **{len(filtered_chunks)}** of **{len(final_documents)}** chunks:")

    max_display = min(len(filtered_chunks), 20)
    for idx, chunk in filtered_chunks[:max_display]:
        with st.expander(f"Chunk #{idx} ({len(chunk.page_content)} chars)"):
            st.code(chunk.page_content, language="text")

    if len(filtered_chunks) > 20:
        st.caption(f"Showing first 20 chunks. Use the keyword filter above to search specific terms.")


# ---------------------------------------------------------
# TAB 3: Pipeline Architecture
# ---------------------------------------------------------
with tab_pipeline:
    st.subheader("📖 Step-by-Step RAG Pipeline Walkthrough")
    st.markdown("""
    This application executes the full Retrieval-Augmented Generation workflow from **`RAGTutorial.ipynb`**:
    """)

    st.markdown("""
    #### 1️⃣ Step 1: Data Loading (`TextLoader`)
    Loads raw text into LangChain `Document` objects:
    ```python
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader("CPP.txt", encoding="utf-8")
    text_documents = loader.load()
    ```

    #### 2️⃣ Step 2: Text Splitting (`RecursiveCharacterTextSplitter`)
    Splits long documents into smaller overlapping passages:
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    finalDocuments = textSplitter.split_documents(text_documents)
    ```

    #### 3️⃣ Step 3: Embeddings (`HuggingFaceEmbeddings`)
    Converts text chunks into dense 384-dimensional vector representations:
    ```python
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ```

    #### 4️⃣ Step 4: Vector Store & Indexing (`FAISS`)
    Builds a fast similarity search index:
    ```python
    from langchain_community.vectorstores import FAISS
    db = FAISS.from_documents(finalDocuments, embeddings)
    ```

    #### 5️⃣ Step 5: Similarity Search (`similarity_search`)
    Finds the most relevant chunks based on vector distance:
    ```python
    docs = db.similarity_search(query)
    for doc in docs:
        print(doc.page_content)
    ```

    #### 6️⃣ Step 6: Augmented Generation (`OllamaLLM`)
    Pipes retrieved context chunks into Ollama (`gemma2:2b`) to generate natural language answers:
    ```python
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    ```
    """)
