import faiss
import numpy as np

def build_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    return index

def search (index, query_vector, top_k=5):

    query_vector = np.array(query_vector).astype("float32")

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    return scores, indices