import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#-----------------------------------------------------
# Page Config
#-----------------------------------------------------
st.set_page_config(page_title = "C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")

#---------------------------------------
#Cache Document loading
#---------------------------------------
@st.cache_resource
def load_vectorestore():
    loader = TextLoader("CPP.txt",encoding="utf-8")
    documents = loader.load()

    #split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    #embeddings
    
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"

    )

    #create FAISS vector store
    db = FAISS.from_documents(final_documents,embeddings)
    return db

db=load_vectorestore()

query = st.text_input("Enter your question about C++ :")
if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Retrieved Context :")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)

    





