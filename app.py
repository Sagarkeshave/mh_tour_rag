import streamlit as st
from rag_system import get_qa_chain, create_vector

st.title("Maharashtra Tourism 🌱")
btn = st.button("Create Knowledgebase")
if btn: 
    create_vector()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response.content)
