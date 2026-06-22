# 📚 AskMA Knowledge Assistant

AskMA is an AI-powered enterprise knowledge assistant designed to help users search and retrieve information from internal MA-IN documentation using Retrieval-Augmented Generation (RAG).

The application combines semantic search, vector embeddings, and Groq LLMs to provide grounded answers based only on uploaded documents.

---

## Features

* Enterprise document search
* Retrieval-Augmented Generation (RAG)
* Chroma Vector Database
* Groq LLM Integration
* Semantic PDF Search
* Source Document References
* Multi-document Knowledge Base
* Streamlit Web Interface

---

## Technology Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Frontend        | Streamlit             |
| LLM             | Groq (Llama 3.3 70B)  |
| Framework       | LangChain             |
| Vector Database | ChromaDB              |
| Embeddings      | Sentence Transformers |
| Document Loader | PyPDF                 |
| Language        | Python                |

---

## Project Structure

enterprise-knowledge-assistant/
│
├── app_enterprise.py
├── ingest.py
├── requirements.txt
│
├── data/
│   └── PDF Documents
│
└── db/
    └── Chroma Vector Database


---

## Installation

Clone repository:


git clone https://github.com/Addyzz10/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant


# Install dependencies:


pip install -r requirements.txt


Create `.env`


GROQ_API_KEY=your_api_key


---

## Build Vector Database

Place PDF documents inside:


data/


# Run:


python ingest.py


This creates the Chroma vector database inside:


db/


---

## Run Application


streamlit run app_enterprise.py

---

## How It Works

1. PDF documents are loaded.
2. Documents are chunked into semantic sections.
3. Embeddings are generated using Sentence Transformers.
4. ChromaDB stores vectors.
5. User submits a query.
6. Relevant chunks are retrieved.
7. Groq LLM generates an answer using only retrieved context.
8. Source documents are displayed alongside the response.

---

## Current Model


Llama 3.3 70B Versatile


Hosted through Groq.

---

## License

For educational and internal enterprise knowledge management purposes.

---

## Author

Aditya Singh
Graduate Apprentice – Data & Business Analyst
Bosch Limited
