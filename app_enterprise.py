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

st.info("""
🚀 Enterprise AI Search

Ask questions from multiple PDFs and receive grounded answers with source references.
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

    import os

    st.write("DB EXISTS:", os.path.exists("db"))

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings,
        collection_name="langchain"
    )

    return db

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

    st.subheader("💡 Example Questions")

    examples = [
        "What is Dynamic Programming?",
        "Explain Inner Join",
        "What is Snowpark?",
        "What is FastAPI?",
        "What is RAG?"
    ]

    for ex in examples:
        st.caption(f"• {ex}")

    st.divider()

    st.success(
        """
        Ask questions across your enterprise documents and get
        grounded answers with source citations.
        """
    )




# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
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

1. Answer ONLY from the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not explicitly present in the context, respond:

I don't know.

4. Do not guess.
5. Do not infer.
6. Do not make assumptions.
7. Keep answers concise and factual.

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

            if len(docs) == 0:

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
        if answer.strip().lower() != "i don't know.":
            with st.expander("📄 Sources"):

                for i, doc in enumerate(sources, start=1):

                    source_name = os.path.basename(
                        doc.metadata.get(
                            "source",
                            "Unknown PDF"
                        )
                    )

                    
                    st.markdown(
                        f"### 📄 {source_name}"
                    )

                    st.text_area(
                        "",
                        doc.page_content[:500],
                        height=150,
                        disabled=True
                    )

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )