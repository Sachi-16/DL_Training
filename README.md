# RAG Tutorial - Streamlit Application

This Streamlit application is converted directly from the [`RAGTutorial.ipynb`](./RAGTutorial.ipynb) notebook. It provides an interactive interface for Retrieval-Augmented Generation (RAG) using **LangChain**, **HuggingFace Embeddings**, and the **FAISS** vector database.

---

## 🚀 Features

- **Document Ingestion (Step 1)**:
  - Preloaded documents (`CPP.txt`, `Transformer.txt`).
  - Upload custom `.txt` or `.pdf` files.
- **Dynamic Chunking (Step 2)**:
  - Adjust `chunk_size` (default: 200) and `chunk_overlap` (default: 20) via sliders.
  - Powered by `RecursiveCharacterTextSplitter`.
- **HuggingFace Embeddings (Step 3)**:
  - Uses `all-MiniLM-L6-v2` dense vector embeddings.
  - Models are cached using `@st.cache_resource` for high performance.
- **Vector Database (Step 4)**:
  - `FAISS` vector index constructed and cached for similarity search.
- **Semantic Search & Retrieval (Step 5)**:
  - Query your document with similarity search.
  - Displays retrieved chunks along with FAISS L2 distance scores.
- **Optional LLM Generation (Step 6)**:
  - Synthesize conversational answers using local Ollama models (e.g., `gemma2:2b`).
- **Chunk Explorer & Architecture Guide**:
  - Inspect split chunks and read the step-by-step breakdown of how RAG works.

---

## 🛠️ How to Run

1. **Activate the Virtual Environment**:
   ```powershell
   .\myenv\Scripts\Activate.ps1
   ```

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **(Optional) Run with Ollama**:
   If you want natural language answer generation:
   ```bash
   ollama run gemma2:2b
   ```
   Then enable the *"Generate synthesized answer with LLM"* checkbox in the sidebar.

