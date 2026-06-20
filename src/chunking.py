def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunks.append({
            "text": text[start:end],
            "start": start,
            "end": end,
        })


        if end == text_length:
            break

        start += chunk_size - overlap
    
    return chunks

def chunk_document(doc_id, text, chunk_size=1000, overlap=200):
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return [
        {
            "doc_id": doc_id,
            "chunk_index": i,
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
        }
        for i, c in enumerate(raw_chunks)
    ]