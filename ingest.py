from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import shutil
import os

PDF_FOLDER = "data"
DB_DIR = "db"

# ----------------------------------
# DELETE OLD VECTOR DB
# ----------------------------------

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

documents = []

print("Loading PDFs...")

# ----------------------------------
# LOAD ALL PDFS
# ----------------------------------

for file in os.listdir(PDF_FOLDER):

    if file.lower().endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, file)

        print(f"\nLoading: {file}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        print(f"Loaded {len(pages)} pages")

        # Combine all pages
        full_text = "\n".join(
            page.page_content for page in pages
        )

        # Split by Q1., Q2., Q3. etc.
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        chunks = splitter.split_text(full_text)
        
        print(f"Created {len(chunks)} chunks from {file}")

        for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file
                        }
                    )
                )

print(f"\nCreated {len(documents)} chunks")

# ----------------------------------
# EMBEDDINGS
# ----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nCreating Vector DB...")

# ----------------------------------
# CREATE CHROMA DB
# ----------------------------------

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name="langchain"
)

vectordb.persist()

print("\nVector DB Created Successfully!")
print("Done!")