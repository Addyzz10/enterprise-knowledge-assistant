from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import streamlit as st
import os
import time

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
}

div[data-testid="stChatMessage"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown("""
# 📚 Enterprise Knowledge Assistant

Ask questions across your enterprise knowledge base and retrieve grounded answers with source citations.
""")
# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def load_vectordb():

    return Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

vectordb = load_vectordb()
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("📄 Loaded Documents")

    try:

        pdfs = set()

        data = vectordb.get()

        for meta in data["metadatas"]:

            if meta and "source" in meta:
                pdfs.add(meta["source"])

        for pdf in sorted(pdfs):
            st.write(f"• {pdf}")

    except Exception:
        st.write("No documents loaded")

    st.subheader("📊 Knowledge Base")

    try:
        chunk_count = vectordb._collection.count()

        st.metric(
            "Chunks Indexed",
            chunk_count
        )

    except Exception:
        pass



# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 20
    }
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# --------------------------------------------------
# STRICT RAG PROMPT
# --------------------------------------------------

prompt_template = """
You are an Enterprise Knowledge Assistant.

Rules:

1. Use ONLY the provided context.
2. Never invent information.
3. If multiple sources contain useful information, combine them.
4. Give a concise but complete answer.
5. If the answer cannot be found in the context, respond exactly:

I don't know.

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

# --------------------------------------------------
# QA CHAIN
# --------------------------------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PROMPT
    }
)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)

# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            docs = retriever.get_relevant_documents(question)

            if not docs:

                answer = "I don't know."

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                st.stop()

            start = time.time()
            result = qa({"query": question})

            answer = result["result"]
            sources = result["source_documents"]
            
            elapsed = round(time.time() - start, 2)

            st.markdown(answer)
            
            st.caption(
                f"⚡ Response generated in {elapsed} sec"
            )

            with st.expander("📄 Sources"):

                for i, doc in enumerate(sources, start=1):

                    source_name = doc.metadata.get(
                        "source",
                        "Unknown PDF"
                    )

                    st.markdown(
                        f"### 📄 {source_name}"
                    )

                    st.text_area(
                        label="",
                        value=doc.page_content[:500],
                        height=180,
                        disabled=True
                    )

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )