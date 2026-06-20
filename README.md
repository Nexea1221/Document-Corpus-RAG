# Document Corpus RAG Project

A simple **Retrieval-Augmented Generation (RAG)** system that allows users to query about the AI Governance Documents Data from the United States.

The system combines structured metadata with vector search and a local LLM (Ollama with Mistral) to answer questions about AI governance laws, with a fall back to txt files for queries which are missed.

---

## Project Architecture
```
AGORA CSVs
↓
Data Loading (data_loader.py)
↓
Embedding Text Construction (embeddings.py)
↓
Vector Embeddings (Sentence Transformers)
↓
FAISS Vector Index (faiss_index.py)
↓
User Query
↓
Similarity Search (retriever.py)
↓
Retrieved Context
↓
LLM (Ollama)
↓
Final Answer
```

---

## Project Structure
```
agora-rag-assistant/
│
├── data/
│   ├── documents/
│   │   └── agora/
│   │       ├── segments.csv
│   │       ├── documents.csv
│   │       ├── authorities.csv 
│   │       ├── collections.csv
│   │       └── fulltext/ 
│   └── cache/
│       └── fulltext_chunks.json
│
├── src/
│   ├── data_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── fulltext_search.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Nexea1221/Document-Corpus-RAG
cd document-corpus-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Install Ollama
https://ollama.com

### 5. Pull a model from Ollama (Run this in Windows Powershell or Command Prompt)
```bash
ollama run mistral
```

### 6. Download and add dataset (ONLY IF DATA FOLDER IS EMPTY/DOES NOT EXIST)
1. Download the dataset from `https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data` or run this code

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("umerhaddii/ai-governance-documents-data")

print("Path to dataset files:", path)
```

2. Create a data folder, and a documents folder

3. Place the dataset into data/documents/ like shown below:

```
data/documents/agora/
├── segments.csv
├── documents.csv
├── authorities.csv
├── collections.csv
└── fulltext/
    ├── 1.txt
    ├── 2.txt
    └── ...
```

## 7. Run project
```bash
py main.py
```

## How to use
1. Download the dataset from the link provided (Only if data/ folder is empty)
2. Place the dataset into the folder shown above (Only if data/ folder is empty)
3. Run the program
4. Ask a question
5. Get answer

--- 

## Note: Since the assessment mentioned not to include any large datasets in the repo, anyone who uses the repo will have to execute Step 6