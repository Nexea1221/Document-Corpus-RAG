import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from src.chunking import chunk_document

DEFAULT_FOLDER = "data/documents/agora/fulltext"
DEFAULT_CHUNKS_CACHE = "data/cache/fulltext_chunks.json"
DEFAULT_INDEX_CACHE = "data/cache/fulltext_index.faiss"

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _load_chunks(chunks_cache):
    if os.path.exists(chunks_cache):
        with open(chunks_cache, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_chunks(chunks_cache, chunks):
    os.makedirs(os.path.dirname(chunks_cache), exist_ok=True)
    with open(chunks_cache, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def build_fulltext_chunks(folder=DEFAULT_FOLDER, chunk_size=1000, overlap=200, chunks_cache=DEFAULT_CHUNKS_CACHE, force_rebuild=False):
    if not force_rebuild:
        cached = _load_chunks(chunks_cache)

        if cached is not None:
            return cached
    
    all_chunks = []


    for file in sorted(os.listdir(folder)):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_chunks = chunk_document(doc_id=file, text=text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)

    _save_chunks(chunks_cache, all_chunks)

    return all_chunks

def build_fulltext_index(chunks, index_cache=DEFAULT_INDEX_CACHE, force_rebuild=False):
    if not force_rebuild and os.path.exists(index_cache):
        return faiss.read_index(index_cache)
    
    model = _get_model()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar = True)

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_cache), exist_ok=True)
    faiss.write_index(index, index_cache)

    return index

def search_fulltext(query, folder=DEFAULT_FOLDER, max_docs = 3, chunk_size=1000, overlap=200, chunks_cache=DEFAULT_CHUNKS_CACHE, index_cache=DEFAULT_INDEX_CACHE):

    chunks = build_fulltext_chunks(folder=folder, chunk_size=chunk_size, overlap=overlap, chunks_cache=chunks_cache)

    if not chunks:
        return []

    index = build_fulltext_index(chunks, index_cache=index_cache)

    model = _get_model()
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, max_docs)

    results = []

    for score, idx in zip(scores[0], idxs[0]):

        if idx < 0:  # FAISS returns -1 if there are fewer results than max_docs
            continue

        chunk = chunks[idx]

        results.append({
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "position": f"fulltext chunk {chunk['chunk_index']}",
            "score": float(score),
            "final_score": float(score),
        })

    return results