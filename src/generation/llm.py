from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


RAG_SYSTEM_TEMPLATE = """You are a domain-expert assistant for technical QA.
Answer the user's question strictly based on the following retrieved context.

Rules:
1. Every claim must cite its source chunk inline using [Chunk <id>].
2. If the context does not contain enough evidence, explicitly state:
   "I do not have sufficient information in the provided context to answer this."
3. Do not invent or assume any facts outside the context.

Context:
{context}
"""

def create_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.6-flash",
        
    )



