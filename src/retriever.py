import numpy as np
import faiss

def retrieve(query, model, index, chunks, k=8):

    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)

    scores, idxs = index.search(q, k)

    results = []

    for score, idx in zip(scores[0], idxs[0]):

        chunk = chunks[idx].copy()

        chunk["score"] = float(score)
        chunk["final_score"] = float(score)

        results.append(chunk)

    return results