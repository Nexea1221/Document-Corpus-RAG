from src.retriever import retrieve
from src.llm import generate_answer
from src.fulltext_search import search_fulltext

def answer_question(question, model, index, chunks):

    retrieved = retrieve(question, model, index, chunks)

    # top_score = retrieved[0]["score"]

    if not retrieved or retrieved[0]["score"] < 0.3:
        fallback = search_fulltext(question)
        retrieved.extend(fallback)

    retrieved = retrieved[:8]

    answer = generate_answer(question, retrieved)

    return answer, retrieved