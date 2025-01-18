import os
from fastapi import FastAPI, Query
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Constants
CONSTITUTION_FILE = "constitution.txt"
EMBEDDING_DIM = 384  # Default for SentenceTransformer models like 'all-MiniLM-L6-v2'
FAISS_INDEX_FILE = "faiss_index.bin"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the embedding model
model = SentenceTransformer(MODEL_NAME)

# Initialize FAISS index
index = None
documents = []

# Load Constitution into FAISS
def load_constitution():
    global index, documents
    if os.path.exists(FAISS_INDEX_FILE):
        # Load pre-built FAISS index
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open("documents.npy", "rb") as f:
            documents = np.load(f, allow_pickle=True).tolist()
        print("FAISS index loaded.")
    else:
        # Create FAISS index and populate it
        with open(CONSTITUTION_FILE, "r", encoding="utf-8") as file:
            sections = file.read().split("\n\n")
            documents = [section.strip() for section in sections if section.strip()]

        embeddings = np.array([model.encode(doc) for doc in documents]).astype("float32")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)  # L2 distance
        index.add(embeddings)

        # Save FAISS index and documents
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open("documents.npy", "wb") as f:
            np.save(f, np.array(documents, dtype=object))
        print("Constitution loaded into FAISS index.")

# Retrieve relevant sections
@app.get("/query")
def query_constitution(query: str = Query(..., description="Your question about the Constitution")):
    try:
        if index is None:
            return {"error": "FAISS index is not initialized."}

        # Generate query embedding
        query_embedding = model.encode(query).astype("float32").reshape(1, -1)

        # Search FAISS index
        distances, indices = index.search(query_embedding, k=3)  # Retrieve top 3 results

        answers = []
        for idx in indices[0]:
            answers.append(documents[idx])

        return {
            "query": query,
            "answers": answers,
        }
    except Exception as e:
        return {"error": str(e)}

# Main script
if __name__ == "__main__":
    load_constitution()
    uvicorn.run(app, host="127.0.0.1", port=8000)
