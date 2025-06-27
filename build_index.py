# build_index.py
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# 🧠 Embeddings
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

# 📄 Load raw text files
DOCUMENT_DIR = "docs"

def load_documents():
    documents = []
    for filename in os.listdir(DOCUMENT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCUMENT_DIR, filename), encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def build_faiss_index():
    docs = load_documents()
    if not docs:
        print("❌ No .txt documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vector_index")
    print("✅ LangChain-compatible FAISS index saved to ./vector_index")

if __name__ == "__main__":
    build_faiss_index()
