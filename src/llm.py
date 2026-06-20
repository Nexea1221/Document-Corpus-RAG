import ollama

def generate_answer(question, contexts):

    context_text = "\n\n".join(
        f"""
[Doc {c['doc_id']} | Seg {c['position']}]

TITLE: {c.get('official_name', '')}
CASUAL NAME: {c.get('casual_name', '')}
LINK: {c.get('link', '')}

AUTHORITY NAME: {c.get('authority_name', '')}
JURISDICTION: {c.get('jurisdiction', '')}
PARENT AUTHORITY: {c.get('parent_authority', '')}

COLLECTION CATEGORIES: {', '.join(c.get('collection_names', []))}
COLLECTION DESCRIPTIONS: {', '.join(c.get('collection_descriptions', []))}

TAGS: {c.get('tags', '')}
SUMMARY: {c.get('summary', '')}

NOT AI-RELATED: {c.get('non_ai_related', '')}
NON-OPERATIONAL: {c.get('non_operational', '')}

TEXT:
{c['text']}
"""
        for c in contexts
    )

    prompt = f"""
You are a strict AI Governance QA System.

RULES:
- Use ONLY the provided context. Do not infer, guess, or construct URLs, citations, dates, or any facts not explicitly present in the context.
- If a specific piece of information (such as a link, date, or status) is not present in the context, say so explicitly rather than guessing.
- If the answer is missing or cannot be found, say "I don't know"
- Every factual claim must be followed by its citation in the format [Doc <id> | Seg <position>]. Do not make uncited claims.
- If sources conflict or seem inconsistent, point out the conflict rather than picking one arbitrarily.
- If only part of the question can be answered from context, answer the part you can and explicitly state which part is not covered.


Before answering, check: is every fact in your answer explicitly present in the CONTEXT above? 
If you are about to state a URL, date, or number that is not character-for-character present in the context, do not include it — say "not specified in the provided context" instead

CONTEXT:
{context_text}  

QUESTION:
{question}

ANSWER:
"""
    
    response = ollama.chat(
        model="mistral",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]