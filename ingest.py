from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docx"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory="./db")
    db.persist()
    db=None 

if __name__ == "__main__":
    main()