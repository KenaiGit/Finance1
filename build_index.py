import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

DOCUMENT_DIR = "docs"
INDEX_FILE = "finance_docs_index.faiss"
MAPPING_FILE = "doc_index_mapping.txt"

def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_files = [f for f in os.listdir(DOCUMENT_DIR) if f.endswith(".txt")]
    embeddings = []
    mapping = []

    for file in doc_files:
        with open(os.path.join(DOCUMENT_DIR, file), encoding="utf-8") as f:
            text = f.read()
            embedding = model.encode([text])[0]
            embeddings.append(embedding)
            mapping.append(file)

    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save index and mapping
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, "w") as f:
        f.write("\n".join(mapping))

    print("✅ FAISS index and mapping created successfully.")

if __name__ == "__main__":
    build_index()
