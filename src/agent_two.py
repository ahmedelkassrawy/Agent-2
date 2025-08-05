from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.tools import tool
import langgraph
from langgraph.graph import MessagesState, StateGraph, START, END
import logging
from typing import Literal, Optional, List, Any
from langchain_core.messages import HumanMessage, AIMessage
import os
import warnings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

# RAG-related imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.models.rag_module import RAGModule
from langgraph.checkpoint.memory import MemorySaver

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

os.environ["GOOGLE_API_KEY"] = "AIzaSyB9o34YFREb_JLj7nXdfNwHfu5Pw9M-Hpw"

class State(MessagesState):
    question : Optional[str] = Field(default=None, description="The question to be answered by the agent.")
    chat_summary: str = Field(default="", description="The summary of the previous conversation.")
    doc_summary: str = Field(default="", description="The summary of the document.")
    query: Optional[str] = Field(default=None, description="The query to be sent to the RAG.")
    document_path: Optional[str] = Field(default=None, description="Path to the document for RAG.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

@tool
def rag_query_tool(question: str, 
                   document_path: str, 
                   model_name: str = "gemini-2.5-flash") -> str:
    """
    Use RAG (Retrieval-Augmented Generation) to answer questions based on a document.
    
    Args:
        question: The question to ask about the document
        document_path: Path to the document (PDF or text file)
        model_name: The LLM model to use (default: gemini-2.5-flash)
    
    Returns:
        The answer based on the document content
    """
    try:
        rag = RAGModule(
            model_name=model_name,
            embeddings_model="sentence-transformers/all-mpnet-base-v2",
            doc_path=document_path
        )
        
        result = rag.initialize_and_ask(question)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return result["answer"]
        
    except Exception as e:
        return f"Error using RAG: {str(e)}"

@tool
def summarize_chat(messages: List[BaseMessage]) -> str:
    """Summarize the chat conversation"""
    if not messages:
        return "No messages to summarize."
    
    # Convert messages to text
    conversation_text = ""
    for msg in messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        conversation_text += f"{role}: {msg.content}\n"
    
    prompt = ChatPromptTemplate.from_template("""
    Summarize the following conversation:
    
    {conversation}
    
    Summary:
    """)

    summary_chain = prompt | llm | StrOutputParser()
    
    summary = summary_chain.invoke({"conversation": conversation_text})
    return summary

@tool 
def summarize_doc(document_path: str) -> str:
    """Summarize the document specified by the path."""
    if not document_path:
        return "No document path provided."
    
    try:
        # Read the document content
        if document_path.endswith('.pdf'):
            loader = PyPDFLoader(document_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
        else:
            # Assume it's a text file
            with open(document_path, 'r', encoding='utf-8') as file:
                content = file.read()
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that summarizes documents.
        Please provide a concise summary of the following document content:
        
        Document Content:
        {content}
        
        Summary:
        """)

        summary_chain = prompt | llm | StrOutputParser()

        summary = summary_chain.invoke({"content": content})
        if not summary:
            return "No summary generated."
        
        return summary
    except Exception as e:
        return f"Error reading or summarizing document: {str(e)}"

def route_request(state: dict) -> Literal["rag_query", "summarize_doc", "summarize_chat", "general_response"]:
    """Router function that uses LLM to determine which tool/node to use based on the user's request"""
    question = state.get("question", "")
    document_path = state.get("document_path", "")
    messages = state.get("messages", [])
    
    # Create a prompt for the LLM to decide routing
    routing_prompt = f"""
    You are a routing assistant that determines which tool should handle a user's request.
    
    Available tools:
    1. "rag_query" - Use this when the user is asking questions about document content and a document is provided
    2. "summarize_doc" - Use this when the user wants to summarize a document and a document path is provided
    3. "summarize_chat" - Use this when the user wants to summarize the conversation/chat history
    4. "general_response" - Use this for general questions that don't require documents or chat summarization
    
    Context:
    - User's question: "{question}"
    - Document path provided: {"Yes" if document_path else "No"}
    - Conversation history available: {"Yes" if len(messages) > 1 else "No"}
    
    Based on the user's request and available context, respond with ONLY ONE of these four options:
    rag_query
    summarize_doc
    summarize_chat
    general_response
    
    Your response must be exactly one of these four words, nothing else.
    """
    
    try:
        # Use LLM to determine routing
        response = llm.invoke(
            [
                {"role": "user",
                "content": routing_prompt}
            ]
        )
        route = response.content.strip().lower()
        
        valid_routes = ["rag_query", 
                        "summarize_doc", 
                        "summarize_chat", 
                        "general_response"]
        
        if route in valid_routes:
            return route
        else:
            # Fallback logic if LLM gives unexpected response
            if document_path and any(word in question.lower() for word in ["what", "how", "why", "when", "where", "explain", "tell me", "describe"]):
                return "rag_query"
            elif "summarize" in question.lower() and ("document" in question.lower() or "doc" in question.lower()) and document_path:
                return "summarize_doc"
            elif "summarize" in question.lower() and ("chat" in question.lower() or "conversation" in question.lower()) and len(messages) > 1:
                return "summarize_chat"
            else:
                return "general_response"
    
    except Exception as e:
        print(f"Error in LLM routing: {e}")
        # Fallback to rule-based routing if LLM fails
        if document_path and any(word in question.lower() for word in ["what", "how", "why", "when", "where", "explain", "tell me", "describe"]):
            return "rag_query"
        elif "summarize" in question.lower() and ("document" in question.lower() or "doc" in question.lower()) and document_path:
            return "summarize_doc"
        elif "summarize" in question.lower() and ("chat" in question.lower() or "conversation" in question.lower()) and len(messages) > 1:
            return "summarize_chat"
        else:
            return "general_response"

def rag_query_node(state: dict) -> dict:
    """Node for handling RAG queries"""
    try:
        question = state.get("question", "")
        document_path = state.get("document_path", "")
        
        if not document_path:
            ai_message = AIMessage(content="I would need a document to answer questions about it. Please provide a document path.")
        else:
            answer = rag_query_tool.invoke({
                "question": question,
                "document_path": document_path
            })
            ai_message = AIMessage(content=answer)
        
        # Ensure messages list exists
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(ai_message)
        
    except Exception as e:
        error_message = AIMessage(content=f"Error processing RAG query: {str(e)}")
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(error_message)
    
    return state

def summarize_doc_node(state: dict) -> dict:
    """Node for handling document summarization"""
    try:
        document_path = state.get("document_path", "")
        
        if not document_path:
            ai_message = AIMessage(content="I would need a document path to summarize it. Please provide a document path.")
        else:
            doc_summary = summarize_doc.invoke({"document_path": document_path})
            ai_message = AIMessage(content=doc_summary)
        
        # Ensure messages list exists
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(ai_message)
        
    except Exception as e:
        error_message = AIMessage(content=f"Error summarizing document: {str(e)}")
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(error_message)
    
    return state

def summarize_chat_node(state: dict) -> dict:
    """Node for handling chat summarization"""
    try:
        messages_to_summarize = state.get("messages", [])
        # Remove the current request message
        if messages_to_summarize:
            messages_to_summarize = messages_to_summarize[:-1]
        
        if not messages_to_summarize:
            ai_message = AIMessage(content="No previous conversation to summarize.")
        else:
            # Convert messages to text
            conversation_text = ""
            for msg in messages_to_summarize:
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                conversation_text += f"{role}: {msg.content}\n"
            
            prompt = ChatPromptTemplate.from_template("""
            Summarize the following conversation:
            
            {conversation}
            
            Summary:
            """)

            summary_chain = prompt | llm | StrOutputParser()
            summary = summary_chain.invoke({"conversation": conversation_text})
            ai_message = AIMessage(content=f"Conversation Summary: {summary}")
        
        # Ensure messages list exists
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(ai_message)
        
    except Exception as e:
        error_message = AIMessage(content=f"Error summarizing conversation: {str(e)}")
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(error_message)
    
    return state

def general_response_node(state: dict) -> dict:
    """Node for handling general responses using LLM"""
    try:
        question = state.get("question", "")
        
        # Use LLM for general responses
        response = llm.invoke(
            [
                {"role": "user", 
                "content": question}
            ]
        )
        ai_message = AIMessage(content=response.content)
        
        # Ensure messages list exists
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(ai_message)
        
    except Exception as e:
        error_message = AIMessage(content=f"Error processing request: {str(e)}")
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(error_message)
    
    return state

workflow = StateGraph(State)


workflow.add_node("rag_query", rag_query_node)
workflow.add_node("summarize_doc", summarize_doc_node)
workflow.add_node("summarize_chat", summarize_chat_node)
workflow.add_node("general_response", general_response_node)


workflow.add_conditional_edges(
    START,
    route_request,
    {
        "rag_query": "rag_query",
        "summarize_doc": "summarize_doc", 
        "summarize_chat": "summarize_chat",
        "general_response": "general_response"
    }
)

workflow.add_edge("rag_query", END)
workflow.add_edge("summarize_doc", END)
workflow.add_edge("summarize_chat", END)
workflow.add_edge("general_response", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

conversation_history = []

while True:
    try:
        config = {"configurable":{"thread_id" : "1"}}
        question = input("Enter your question (or 'exit' to quit): ").strip()
        
        if question.lower() == "exit":
            break
            
        document_path = input("Enter document path (or leave blank if not applicable): ").strip()
        
        # Add the current question to conversation history
        conversation_history.append(HumanMessage(content=question))
        
        # Create initial state as dictionary (LangGraph converts internally)
        initial_state = {
            "messages": conversation_history.copy(),  # Use the persistent conversation history
            "question": question,
            "document_path": document_path if document_path else None
        }

        res = app.invoke(initial_state, config=config)

        if "messages" in res and res["messages"]:
            # Get the last AI message from the result
            latest_message = res["messages"][-1].content
            print(f"AI Response: {latest_message}")
            
            # Update conversation history with the AI response
            # Check if we need to add the AI response (it might already be in conversation_history)
            if not conversation_history or conversation_history[-1].content != latest_message:
                conversation_history.append(AIMessage(content=latest_message))
        else:
            print("No response from AI.")
        
    except Exception as e:
        print(f"Error: {e}")
        break


