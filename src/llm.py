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

COLLECTION CATEGORIES: {', '.join(c.get('collection_categories') or [])}
COLLECTION DESCRIPTIONS: {', '.join(c.get('collection_descriptions') or [])}

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
You are a strict AI Governance QA System. You must follow these rules with no exceptions:

1. Read the CONTEXT below.
2. If the CONTEXT does not contain information that directly answers the QUESTION, you MUST respond with exactly: "I don't know"
3. Do NOT use any knowledge outside the CONTEXT, even if you know the answer.
4. Do NOT explain why the context is irrelevant. Just say "I don't know".
5. Always cite Document ID and segment position when you do answer.

CONTEXT:
{context_text}  

QUESTION:
{question}

Respond with ONLY the answer or "I don't know". Do not add commentary.

ANSWER:
"""
    
    response = ollama.chat(
        model="mistral",
        messages = [
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]