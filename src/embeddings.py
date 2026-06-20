from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_embedding_text(chunk):
    
    return f"""
TITLE: {chunk.get('official_name', '')}
CASUAL NAME: {chunk.get('casual_name', '')}

TAGS: {chunk.get('tags', '')}
SUMMARY: {chunk.get('summary', '')}

JURISDICTION: {chunk.get('jurisdiction', '')}
AUTHORITY LEVEL: {chunk.get('parent_authority', '')}

COLLECTION CATEGORIES: {', '.join(chunk.get('collection_names', []))}

# STATUS: {chunk.get('status', '')}

TEXT:
{chunk['text']}
"""

def embed_chunks(chunks):
    texts = [build_embedding_text(c) for c in chunks]
    embeddings = model.encode(texts, show_progress_bar = True)
    return embeddings