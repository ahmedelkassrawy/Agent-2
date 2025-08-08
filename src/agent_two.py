from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.tools import tool
import langgraph
from langgraph.graph import MessagesState, StateGraph, START, END
import logging
from typing import Literal, Optional, List, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import os
import warnings
import requests
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
from models.rag_model import RAGModule
from models.mongo_memory import SimpleMongoMemory
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatLiteLLM

# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGFUSE_PUBLIC_KEY"] ="pk-lf-27f7fa53-b370-46d2-82f0-6f32851dfc92"
os.environ["LANGFUSE_SECRET_KEY"]="sk-lf-c3571355-5d0c-48bb-ac92-c3dfaecea1c2"
os.environ["LANGFUSE_HOST"]="https://cloud.langfuse.com"

# Set API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# LLM Configuration with Proxy Support
def setup_llm():
    """
    Setup LLM with proxy support. Uses LiteLLM proxy if configured,
    otherwise falls back to direct API calls.
    """
    use_proxy = os.getenv('USE_LLM_PROXY', 'false').lower() == 'true'
    proxy_url = os.getenv('LLM_PROXY_URL', 'http://localhost:4000')
    
    if use_proxy:
        try:
            response = requests.get(f"{proxy_url}/health", timeout=5)

            if response.status_code == 200:
                print("🔗 Using LiteLLM Proxy connection")
                return ChatLiteLLM(
                    model="groq-gemma9b",  # Primary model from proxy
                    api_base=proxy_url,
                    temperature=0.7,
                    max_tokens=1000
                )
            else:
                print("⚠️  Proxy not responding, falling back to direct API")
        except Exception as e:
            print(f"⚠️  Proxy connection failed: {e}, falling back to direct API")
    
    
    print("🔗 Using direct API connection")
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not google_key or not groq_key:
        raise ValueError("Missing required API keys. Please set GOOGLE_API_KEY and GROQ_API_KEY environment variables.")
    
    os.environ["GOOGLE_API_KEY"] = google_key
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["GROQ_API_KEY"] = groq_key
    
    try:
        print("🔄 Attempting Gemini model...")
        llm = ChatLiteLLM(
            model="gemini/gemini-1.5-flash",
            temperature=0.7,
            max_tokens=1000
        )
        
        test_response = llm.invoke("Hello")
        print("✅ Gemini model initialized successfully")
        return llm
    except Exception as e:
        print(f"⚠️  Gemini failed: {str(e)[:100]}...")
        print("🔄 Attempting Groq model...")
        try:
            llm = ChatLiteLLM(
                model="groq/llama3-8b-8192",  
                temperature=0.7,
                max_tokens=1000
            )
            
            test_response = llm.invoke("Hello")
            print("✅ Groq model initialized successfully")
            return llm
        except Exception as e2:
            print(f"❌ Both models failed. Gemini: {str(e)[:50]}... | Groq: {str(e2)[:50]}...")
            
            return ChatLiteLLM(
                model="gemini/gemini-1.5-flash", 
                temperature=0.7,
                max_tokens=1000
            )


llm = setup_llm()

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "Up1yTRmr2bZ73o4fzSseBaonBJLaHBN9ZCeVh1xG")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-27f7fa53-b370-46d2-82f0-6f32851dfc92")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-c3571355-5d0c-48bb-ac92-c3dfaecea1c2")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

class State(MessagesState):
    question : Optional[str] = Field(default=None, description="The question to be answered by the agent.")
    chat_summary: str = Field(default="", description="The summary of the previous conversation.")
    doc_summary: str = Field(default="", description="The summary of the document.")
    query: Optional[str] = Field(default=None, description="The query to be sent to the RAG.")
    document_path: Optional[str] = Field(default=None, description="Path to the document for RAG.")
    user_id: Optional[str] = Field(default="user", description="User ID for accessing preferences.")

@tool
def rag_query_tool(question: str, 
                   document_path: str, 
                   model_name: str = "gemini-2.5-flash") -> str:
    """
    Use RAG (Retrieval-Augmented Generation) to answer questions based on a document.
    
    Args:
        question: The question to ask about the document
        document_path: Path to the document (PDF or text file)
        model_name: The LLM model to use (default: gemini-1.5-flash)
    
    Returns:
        The answer based on the document content
    """
    try:
        # Use proxy-aware model name if proxy is enabled
        use_proxy = os.getenv('USE_LLM_PROXY', 'false').lower() == 'true'
        if use_proxy:
            model_name = "groq-gemma9b"  # Use proxy model
        
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
        # PDF Loading
        if document_path.endswith('.pdf'):
            loader = PyPDFLoader(document_path)
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
        else:
            # Text Loading
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
        response = llm.invoke([HumanMessage(content=routing_prompt)])
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
    """Node for handling RAG queries with user preferences"""
    try:
        question = state.get("question", "")
        document_path = state.get("document_path", "")
        user_id = state.get("user_id", "user")
        
        if not document_path:
            ai_message = AIMessage(content="I would need a document to answer questions about it. Please provide a document path.")
        else:
            # Get user preferences to customize RAG response
            user_preferences = mongo_memory.get_user_preferences(user_id)
            
            # Modify question to include preference context if available
            enhanced_question = question
            
            if user_preferences:
                preference_context = f"User preferences: {user_preferences}. "
                enhanced_question = f"{preference_context} Please consider these preferences when answering: {question}"
            
            answer = rag_query_tool.invoke({
                "question": enhanced_question,
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
    """Node for handling general responses using LLM with user preferences"""
    try:
        question = state.get("question", "")
        user_id = state.get("user_id", "user") 

        # User preferences
        user_preferences = mongo_memory.get_user_preferences(user_id)
        
        if user_preferences:
            preference_text = "\n".join([f"- {key}: {value}" for key, value in user_preferences.items()])
            system_prompt = f"""You are a helpful AI assistant. Please consider the user's preferences when responding:

                            User Preferences:
                            {preference_text}

                            Please tailor your response to align with these preferences when relevant. For example:
                            - If the user prefers a certain programming language, suggest solutions in that language
                            - If the user has a preferred communication style, match that tone
                            - If the user has specific interests, relate your answers to those interests when possible

                            User Question: {question}

                            Provide a helpful response that takes into account the user's preferences where applicable."""
        else:
            system_prompt = f"""You are a helpful AI assistant.

                            User Question: {question}

                            Provide a helpful response."""
        
        # Use LLM for general responses with preferences context        
        if user_preferences:
            preference_text = "\n".join([f"- {key}: {value}" for key, value in user_preferences.items()])
            system_content = f"""You are a helpful AI assistant. Please consider the user's preferences when responding:

                                User Preferences:
                                {preference_text}

                                Please tailor your response to align with these preferences when relevant. For example:
                                - If the user prefers a certain programming language, suggest solutions in that language
                                - If the user has a preferred communication style, match that tone
                                - If the user has specific interests, relate your answers to those interests when possible"""
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=question)
            ]
        else:
            messages = [HumanMessage(content=question)]
        
        response = llm.invoke(messages)
        ai_message = AIMessage(content=response.content)
        
        # Ensure messages list exists
        if "messages" not in state:
            state["messages"] = []
        state["messages"].append(ai_message)
        
    except Exception as e:
        error_str = str(e)
        if "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
            error_message = AIMessage(content=f"""❌ API Key Error: The configured API key is invalid or expired.

                                                    Possible solutions:
                                                    1. Check your .env file and ensure GOOGLE_API_KEY or GROQ_API_KEY are set correctly
                                                    2. Get a new API key from:
                                                    - Google AI Studio: https://aistudio.google.com/app/apikey
                                                    - Groq: https://console.groq.com/keys
                                                    3. Set the environment variable: export GOOGLE_API_KEY="your-key-here"

                                                    Technical error: {error_str[:200]}...""")
        elif "litellm.AuthenticationError" in error_str:
            error_message = AIMessage(content=f"""🔑 Authentication Error: Unable to authenticate with the AI service.

                                                Please verify your API keys are:
                                                1. Valid and not expired
                                                2. Have sufficient credits/quota
                                                3. Are set correctly in environment variables

                                                Error details: {error_str[:200]}...""")
        else:
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

mongo_memory = SimpleMongoMemory(
    connection_string="mongodb://localhost:27017/",
    database_name="agent_memory"
)

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def print_memory_stats():
    stats = mongo_memory.get_memory_stats()
    print("\n=== Long Memory Stats ===")
    print(f"Users with preferences: {stats.get('long_memory_users', 0)}")
    print(f"Memory type: {stats.get('memory_type', 'unknown')}")
    print("========================\n")

def print_agent_config():
    """Print current agent configuration"""
    use_proxy = os.getenv('USE_LLM_PROXY', 'false').lower() == 'true'
    proxy_url = os.getenv('LLM_PROXY_URL', 'http://localhost:4000')
    
    print("\n🚀 Agent Two - Enhanced Configuration")
    print("=====================================")
    print("Configuration:")
    if use_proxy:
        print("  • LLM Mode: 🔗 Proxy")
        print(f"  • Proxy URL: {proxy_url}")
        print("  • Primary Model: groq-gemma9b")
        print("  • Fallback: groq-mixtral, gemini-1.5-flash")
    else:
        print("  • LLM Mode: 🔗 Direct API")
        print("  • Model: gemini/gemini-1.5-flash (fallback: groq/llama3-8b-8192)")
    
    print("  • Memory: MongoDB + InMemory")
    print("  • RAG: HuggingFace Embeddings + Chroma")
    print("  • Features: PDF Processing, User Preferences")
    print("=====================================\n")

def save_user_preference(user_id: str, key: str, value: Any):
    mongo_memory.save_user_preference(user_id, key, value)

def get_user_preference(user_id: str, key: str, default: Any = None) -> Any:
    return mongo_memory.get_user_preference(user_id, key, default)

def main():
    """Main interactive function"""
    # Display configuration
    print_agent_config()
    
    # Get thread ID from user or use default
    print("Welcome to Agent-2 with Long Memory Only & Personalized AI!")
    print("Commands:")
    print("  - Type your questions normally (AI will consider your preferences)")
    print("  - Type 'stats' to see memory statistics")
    print("  - Type 'config' to see current configuration")
    print("  - Type 'profile <key> <value>' to set preferences that guide AI responses")
    print("  - Type 'myprofile' to see your current preferences")
    print("  - Type 'exit' to quit")
    print("\nPersonalization Examples:")
    print("  profile language python       # AI will suggest Python solutions")
    print("  profile tone friendly         # AI will use a friendly tone")
    print("  profile expertise beginner    # AI will explain things simply")
    print("  profile interests 'AI, music' # AI will relate answers to your interests")

    # Extract user_id for preferences
    user_id = input("\nEnter your user ID (or press Enter for default 'user'): ").strip()

    if not user_id:
        user_id = "user"

    print(f"Using user ID: {user_id}")


    user_preferences = mongo_memory.get_user_preferences(user_id)

    if user_preferences:
        print(f"Loaded {len(user_preferences)} user preferences")
        print(f"Welcome back! Your preferences: {user_preferences}")

    while True:
        try:
            config = {"configurable": {"thread_id": f"session_{user_id}"}}
            
            user_input = input(f"\n[{user_id}] Enter your question: ").strip()
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "stats":
                print_memory_stats()
                continue
            elif user_input.lower() == "config":
                print_agent_config()
                continue
            elif user_input.lower() == "myprofile":
                prefs = mongo_memory.get_user_preferences(user_id)
                if prefs:
                    print(f"\nYour preferences:")
                    for key, value in prefs.items():
                        print(f"  {key}: {value}")
                else:
                    print("No preferences set yet.")
                continue
            elif user_input.startswith("profile "):
                parts = user_input.split(" ", 2)
                if len(parts) >= 3:
                    key, value = parts[1], parts[2]
                    save_user_preference(user_id, key, value)
                    print(f"✓ Saved preference: {key} = {value}")
                    print("This preference will be considered in future AI responses!")
                else:
                    print("Usage: profile <key> <value>")
                    print("Examples:")
                    print("  profile language python")
                    print("  profile communication_style friendly")
                    print("  profile interests 'AI, coding, music'")
                    print("  profile tone casual")
                continue
                
            document_path = input("Enter document path (or leave blank): ").strip()
            
            initial_state = {
                "messages": [],  
                "question": user_input,
                "document_path": document_path if document_path else None,
                "user_id": user_id  
            }

            res = app.invoke(initial_state, config=config)

            if "messages" in res and res["messages"]:
                # Get the last AI message from the result
                latest_message = res["messages"][-1].content
                print(f"\nAI Response: {latest_message}")
            else:
                print("No response from AI.")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Cleanup
    mongo_memory.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()