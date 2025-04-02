import streamlit as st
from rag_system import get_qa_chain, create_vector
from dotenv import load_dotenv
import os
load_dotenv() 
# google_api_key = "YOUR_API_KEY"
google_api_key= os.getenv("google_api_key")
root_dir = os.getenv("root_dir")

db_path = os.path.join(root_dir, "database", "faiss_index")

st.title("Maharashtra Tourism 🌱")
btn = st.button("Create Knowledgebase")

if btn:
    if not os.path.exists(db_path):
        st.error(f"FAISS index not found at {db_path}. Make sure it exists.")
    else:
        create_vector(db_path=db_path)

question = st.text_input("Question: ")

if question:

    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response.content)
