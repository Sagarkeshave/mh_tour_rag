import os
import streamlit as st
import pickle
import google.generativeai as genai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from api_key import google_api_key
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

genai.configure(api_key=google_api_key)
st.title("Mah Tourism information")
# llm = genai.GenerativeModel('gemini-pro')
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

main_placeholder = st.empty()
file_path = "data/mh_database"

# google_palm_embeddings = GooglePalmEmbeddings(google_api_key=  google_api_key )


# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):

        knowledge_base = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        # create retriever for performing the query
        retriever = knowledge_base.as_retriever(score_threshold=0.7)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        # prompt template to prevent model hallucination
        prompt_template = """
            Human: You are an AI tourist guuide assistant, and provides answers to questions by using fact based and statistical information when possible.
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
            If the answer of quetion is not present in context or quetion is not related to context then state that you don't know, don't try to make up an answer.
            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            Assistant:"""

        # Create a PromptTemplate instance with the defined template and input variables
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
        rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm)
        result = rag_chain.invoke(query)


        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result)

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

