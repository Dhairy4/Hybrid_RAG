from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

RAG_SYSTEM_TEMPLATE = """
You are an assistant answering questions using retrieved document chunks.

Rules:
1. Answer using only the supplied context.
2. Cite supporting chunks using [Chunk <id>].
3. Do not invent facts or citations.
4. If the context is insufficient, clearly say:
   "I do not have sufficient information in the provided context."
5. Distinguish information found in the documents from your own assumptions.
"""

def create_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.6-flash",
        
    )

    prompt = ChatPromptTemplate.from_messages([

        ("system", RAG_SYSTEM_TEMPLATE),
        (
            "human",
            "context : {context}\n\n"
            "question : {question}\n\n",
        ),
    ])

    return prompt| llm | StrOutputParser()