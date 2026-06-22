from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key="dummy",
    model_name="llama-3.1-8b-instant"
)

print("SUCCESS")