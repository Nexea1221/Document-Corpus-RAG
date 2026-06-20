import pandas as pd
import os

def load_data(path):

    segments = pd.read_csv(os.path.join(path, "segments.csv"))
    documents = pd.read_csv(os.path.join(path, "documents.csv"))
    authorities = pd.read_csv(os.path.join(path, "authorities.csv"))
    collections = pd.read_csv(os.path.join(path, "collections.csv"))

    segments.columns = segments.columns.str.strip()
    documents.columns = documents.columns.str.strip()
    authorities.columns = authorities.columns.str.strip()
    collections.columns = collections.columns.str.strip()

    # print("DOCUMENTS:", documents.columns)
    # print("AUTHORITIES:", authorities.columns)
    # print("SEGMENTS:", segments.columns)
    # print("AUTHORITIES:", collections.columns)

    doc_map = documents.set_index("AGORA ID").to_dict("index")
    auth_map = authorities.set_index("Name").to_dict("index")
    collection_map = collections.set_index("Name").to_dict("index")

    # collection_names = collections["Name"].tolist()
    # collection_description = collections["Description"].tolist()

    chunks = []

    for _, row in segments.iterrows():

        doc_id = row["Document ID"]
        meta = doc_map.get(doc_id, {})

        authority_name = meta.get("Authority", "")
        auth = auth_map.get(authority_name, {})

        raw_collections = meta.get("Collections", "") or ""
        collection_names = [c.strip() for c in str(raw_collections).split(";") if c.strip()]
        collection_descriptions = [collection_map.get(name, {}).get("Description", "") for name in collection_names]

        chunks.append({
            "text": row["Text"],
            "summary": row.get("Summary", ""),
            "tags": row.get("Tags", ""),   

            "doc_id": doc_id,
            "position": row["Segment position"],
            "source": doc_id,

            "official_name": meta.get("Official name", ""),
            "casual_name": meta.get("Casual name", ""),
            "link": meta.get("Link to document", ""),

            "authority_name": auth.get("Name", authority_name),
            "jurisdiction": auth.get("Jurisdiction", ""),
            "parent_authority": auth.get("Parent authority", ""),

            "collection_categories": collection_names,
            "collection_descriptions": collection_descriptions,

            "non_ai_related": row.get("Not AI-related", False),
            "non_operational": row.get("Non-operative", False),
        })

    return chunks