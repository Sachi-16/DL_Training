import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

#*****************
#page config
#*****************
st.title("C++ RAG Chatbot using Ollama")

#***********************
#load and prosess data
#***********************
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

#***********************
#Load LLM (Ollama)
#***********************
llm = Ollama(model="gemma2:2b")

#************************
#Chat Interface
#************************
user_question = st.text_input("Ask a question about C++ :")

if user_question:
    with st.spinner("Thinking...."):
        docs = db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}
        
        Question:
        {user_question}
        
        Answer:
        """

        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)
        









