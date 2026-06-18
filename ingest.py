from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
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
        qa_sections = re.split(
            r"(?=Q\d+\.)",
            full_text
        )

        for section in qa_sections:

            section = section.strip()

            if section.startswith("Q"):

                documents.append(
                    Document(
                        page_content=section,
                        metadata={
                            "source": PDF_PATH,
                            "page": page_number
                        }
                    )
                )

print(f"\nCreated {len(documents)} Q&A chunks")

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
    persist_directory=DB_DIR
)

vectordb.persist()

print("\nVector DB Created Successfully!")
print("Done!")