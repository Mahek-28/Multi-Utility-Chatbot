from __future__ import annotations
import tempfile
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any, Annotated, Dict,Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage,SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import operator
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver # MemorySaver store memory in RAM
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests
from langgraph.prebuilt import ToolNode,tools_condition
from  langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool 
from google.api_core.exceptions import ResourceExhausted

# load environment variables from .env file
load_dotenv()

# ====================================
# 1. LLM + embeddings
# ====================================

# ----------------------------
# Primary LLM (Gemini)
# ----------------------------
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# ----------------------------
# Fallback LLM (Groq)
# ----------------------------
groq_llm = ChatGroq(
    api_key=os.getenv('GROK_API_KEY'),
    model='llama-3.3-70b-versatile'
)

embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

class FallbackLLM:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, *args, **kwargs):
        try:
            return self.primary.invoke(*args, **kwargs)

        except ResourceExhausted:
            print("Gemini quota exceeded. Switching to Groq...")
            return self.fallback.invoke(*args, **kwargs)

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("Gemini rate limited. Switching to Groq...")
                return self.fallback.invoke(*args, **kwargs)
            raise

    def stream(self, *args, **kwargs):
        try:
            yield from self.primary.stream(*args, **kwargs)

        except ResourceExhausted:
            print("Gemini quota exceeded. Switching to Groq...")
            yield from self.fallback.stream(*args, **kwargs)

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("Gemini rate limited. Switching to Groq...")
                yield from self.fallback.stream(*args, **kwargs)
            else:
                raise

    def bind_tools(self, tools):
        """Bind tools to the primary LLM."""
        self.primary = self.primary.bind_tools(tools)
        self.fallback = self.fallback.bind_tools(tools)
        return self

llm = FallbackLLM(gemini_llm, groq_llm)

#======================================
# 2. PDF retriever store (per thread)
# =====================================
_THREAD_RETRIEVERS: Dict[str,Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id:Optional[str]):
    """Fetch the retriver for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id:str, filename: Optional[str] = None)-> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("NO bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n"," ",""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type='similarity',search_kwargs={"k":4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename":filename or os.path.basename(temp_path),
            "documkents":len(docs),
            "chunks":len(chunks),
        }

        return{
            "filename":filename or os.path.basename(temp_path),
            "documkents":len(docs),
            "chunks":len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove 
        try:
            os.remove(temp_path)
        except OSError:
            pass

#======================================
# 3. Tools
# =====================================
#Tools
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float,second_num: float,operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    supported operations: add, sub, mul, div
    """ 
    try:
        if operation == 'add':
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {'error':'Division by zero is not allowed'}
            result = first_num/second_num
        else:
            return {'error':f"Unsupported operation '{operation}'"}
        return {'first_num':first_num,"second_num":second_num,'operation':operation,'result':result}
    except Exception as e:
        return {'error':str(e)}


@tool
def get_stock_info(symbol: str) -> dict:
    """
    Get the latest stock information.

    Example:
        AAPL
        TSLA
        MSFT
        NVDA
    """

    symbol = symbol.upper().strip()

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if "Global Quote" not in data or not data["Global Quote"]:
            return {
                "success": False,
                "message": f"No stock found for '{symbol}'."
            }

        quote = data["Global Quote"]

        return {
            "success": True,
            "symbol": quote.get("01. symbol"),
            "price": float(quote.get("05. price", 0)),
            "open": float(quote.get("02. open", 0)),
            "high": float(quote.get("03. high", 0)),
            "low": float(quote.get("04. low", 0)),
            "previous_close": float(quote.get("08. previous close", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": quote.get("10. change percent"),
            "volume": int(float(quote.get("06. volume", 0))),
            "latest_trading_day": quote.get("07. latest trading day")
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

@tool
def rag_tool(query: str, thread_id: Optional[str] = None)->dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return{
            "error":"No document indexed for this chat. Upload a PDF first.",
            "query":query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return{
        "query":query,
        "context":context,
        "metadata":metadata,
        "source_file":_THREAD_METADATA.get(str(thread_id),{}).get("filename"),
    }

tools = [search_tool,get_stock_info, calculator,rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ======================================
# 4. State
# ======================================

class ChatState(TypedDict):
    # BaseMessage includes any message type: HumanMessage, AIMessage, or SystemMessage
    messages: Annotated[list[BaseMessage], add_messages]  # type: ignore

# ======================================
# 5. Nodes
# ======================================

def chat_node(state: ChatState,config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None 
    if config and isinstance(config,dict):
        thread_id = config.get("configurable",{}).get("thread_id")

    system_message = SystemMessage(
        content=(
            f"""
You are a helpful assistant.

If the user asks ANY question about an uploaded PDF,
ALWAYS call the tool `rag_tool`.

Pass:

query = user's question

thread_id = "{thread_id}"

Never answer from memory if a PDF exists.
Always retrieve first.

Use web search only if the question is unrelated to the PDF.
You can also use the web search, stock price, and 
            calculator tools when heplful.
"""
        )
    )

    """Process the current state and invoke the LLM with stored messages."""
    messages = [system_message, *state['messages']]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# =========================================
# 6. Checkpointer
# =========================================

# connect to the SQLite database for persistence
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# persist checkpoint data into SQLite
checkpointer = SqliteSaver(conn=conn)

# ==========================================
# 7. Graph
# ==========================================
graph = StateGraph(ChatState)
# add the chat node to the state graph
graph.add_node('chat_node', chat_node)
graph.add_node('tools',tool_node)

# wire the graph from START to chat_node and from chat_node to END
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools', 'chat_node')

# compile the graph into a chatbot object with persistence
chatbot = graph.compile(checkpointer=checkpointer)

# ==========================
# 7. Helper
# ==========================
def retrieve_all_threads():
    """Return all distinct thread IDs stored in the SQLite checkpoint history."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

def thread_has_document(thread_id: str)->bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id:str)->dict:
    return _THREAD_METADATA.get(str(thread_id),{})
