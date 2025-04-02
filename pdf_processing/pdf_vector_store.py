from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from api_key import google_api_key
import os

pdf_folder_path = r"C:\Users\SAGAR KESHAVE\PycharmProjects\mh_tour_RAG\pdf_processing\pdfs"

# create embedding
google_palm_embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)


def load_chunk_persist_pdf() -> FAISS:
    # pdf_folder_path = "D:\\diptiman\\dataset\\consent_forms_cleaned"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)

    embeddings = google_palm_embeddings

    vectorstore = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)
    return vectorstore


if __name__ == "__main__":
    # vectorstore = load_chunk_persist_pdf()
    # vectorstore.save_local("PDF_VECTORSTORE")

    vectorstore = FAISS.load_local(r"C:\Users\SAGAR KESHAVE\PycharmProjects\mh_tour_RAG\pdf_processing\PDF_VECTORSTORE",
                                   google_palm_embeddings, allow_dangerous_deserialization=True)
    url_db = FAISS.load_local(r"C:\Users\SAGAR KESHAVE\PycharmProjects\mh_tour_RAG\url_2", google_palm_embeddings,
                              allow_dangerous_deserialization=True)

    vectorstore.merge_from(url_db)

    vectorstore.save_local("Final_knowledge_base")
