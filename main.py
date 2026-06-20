from src.data_loader import load_data
from src.embeddings import embed_chunks, model
from src.faiss_index import build_index
from src.rag_pipeline import answer_question

chunks = load_data("data/documents/agora")
embeddings = embed_chunks(chunks)
index = build_index(embeddings)

print("\nRAG System is Ready\n")

while True:
    try: 
        query = input("Ask a question (or type 'exit' to exit): ")

        if query.lower() == "exit":
            break

        answer, sources= answer_question(query, model, index, chunks)

        print("\nANSWER:\n", answer)

        if not sources:
            print("\nNo Sources Found\n")

        print("\nSOURCES:\n")
        for s in sources:
            print(f"- Doc {s['doc_id']} | Seg {s['position']} | Score {s.get('score', 0):.3f}")

        print("\n" + "-"*50 + "\n")

    except Exception as e:
        print("Error:", e)