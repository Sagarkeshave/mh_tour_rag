import warnings
warnings.filterwarnings("ignore")

# from langchain.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
import os
import time
from api_key import google_api_key
# from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv() 
# google_api_key = "YOUR_API_KEY"
google_api_key= os.getenv("google_api_key")
root_dir = os.getenv("root_dir")


print(f"google_api_key {google_api_key}")
print(f"root_dir {root_dir}")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=google_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)


# we will create vector database and store it locally and use again (to avoid databasing multiple times)
file_path = "url_2"

urls = ["https://pune.gov.in/tourist-places/",
        "https://nashik.gov.in/tourism/tourists-places/"]

url_2 = ["https://pune.gov.in/tourist-places/", "https://nashik.gov.in/tourism/tourists-places/",
         "https://www.trawell.in/maharashtra", "https://www.trawell.in/maharashtra/best-places-to-visit",
         "https://ahmednagar.nic.in/en/tourist-places/", "https://gadchiroli.gov.in/tourist-places/",
         "https://chanda.nic.in/en/tourist-places/", "https://www.satara.gov.in/en/tourist-places/",
         "https://mumbaicity.gov.in/tourism/tourist-places/",
         "https://www.godigit.com/explore/wildlife-safari/tiger-reserves-in-maharashtra"
         ]


db_path = os.path.join(root_dir, "database", "faiss_index")
os.makedirs(db_path, exist_ok=True)

def create_vector():
    loader = WebBaseLoader(url_2)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['.', ','],
        chunk_size=1000)

    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    print(f"google_api_key {google_api_key}")
    vectorstore_ = FAISS.from_documents(docs, embeddings)

    vectorstore_.save_local(db_path)


# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain():
    knowledge_base = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # create retriever for performing the query.
    retriever = knowledge_base.as_retriever(score_threshold=0.7)

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
    # result = rag_chain.invoke(query)

    return rag_chain


if __name__ == "__main__":
    create_vector()
    time.sleep(3)

    queries = ["Give information about Pune Tourism", "Who is the best cricketer in the world"]

    chain = get_qa_chain()

    for query in queries:
        result = chain.invoke(query)
        print(f"\n\n Output for Query '{query}': {result.content} \n\n")
