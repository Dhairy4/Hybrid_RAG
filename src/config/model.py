from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv  
from langfuse.langchain import CallbackHandler
from langfuse import get_client

load_dotenv()  # Load environment variables from .env file

langfuse_client = get_client()
langfuse_handler = CallbackHandler()



llm  = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b"
)



output = llm.invoke("hello", config={"callbacks": [langfuse_handler]})

print(output)