from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise Knowledge Assistant")
st.markdown("Ask questions from your knowledge base")

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:
    st.header("Settings")

    if st.button("Clear Chat"):
        st.session_state.clear()
        st.rerun()

# -------------------------------
# EMBEDDINGS
# -------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# VECTOR DB
# -------------------------------

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -------------------------------
# LLM
# -------------------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# -------------------------------
# STRICT RAG PROMPT
# -------------------------------

prompt_template = """
You are an Enterprise Knowledge Assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
respond with:

"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------------------
# QA CHAIN
# -------------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# -------------------------------
# USER INPUT
# -------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is Dynamic Programming?"
)

# -------------------------------
# ANSWER
# -------------------------------

if question:

    with st.spinner("Searching knowledge base..."):

        result = qa({"query": question})

        answer = result["result"]
        sources = result["source_documents"]

    st.subheader("Answer")
    st.write(answer)

    # ---------------------------
    # SOURCES
    # ---------------------------

    with st.expander("View Sources"):

        for i, doc in enumerate(sources, start=1):

            st.markdown(f"### Source {i}")

            st.write(doc.page_content[:1000])

            st.divider()