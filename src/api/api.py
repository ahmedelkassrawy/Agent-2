from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
import os
import traceback
import time
import asyncio
import httpx
from datetime import datetime

# Add the src directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our agent components
from agent_two import app as agent_app, mongo_memory, State
from models.mongo_memory import SimpleMongoMemory
from langchain_core.messages import HumanMessage, AIMessage
from agent_two import app as agent_app, mongo_memory, State
from models.mongo_memory import SimpleMongoMemory
from langchain_core.messages import HumanMessage, AIMessage

from typing import Annotated
from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

# Initialize FastAPI app
app = FastAPI(
    title="Agent Two API",
    description="API for Agent Two - An intelligent assistant with RAG capabilities and memory",
    version="1.0.0"
)

# Configuration
SECRET_KEY = "your-secret-key-here-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

# Request/Response Models
class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agent")
    user_id: str = Field(default="user", description="User ID for personalization")
    document_path: Optional[str] = Field(None, description="Path to document for RAG queries")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation context")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's response")
    user_id: str = Field(..., description="User ID")
    thread_id: str = Field(..., description="Thread ID used for the conversation")
    timestamp: datetime = Field(..., description="Response timestamp")
    route_used: Optional[str] = Field(None, description="Which processing route was used")

class PreferenceRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    key: str = Field(..., description="Preference key")
    value: Any = Field(..., description="Preference value")

class PreferenceResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    key: str = Field(..., description="Preference key")
    value: Any = Field(..., description="Preference value")
    timestamp: datetime = Field(..., description="When the preference was saved")

class UserPreferencesResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    preferences: Dict[str, Any] = Field(..., description="All user preferences")

class MemoryStatsResponse(BaseModel):
    long_memory_users: int = Field(..., description="Number of users with stored preferences")
    memory_type: str = Field(..., description="Type of memory system")
    total_users:int = Field(..., description="Total number of users in the memory system")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    memory_connected: bool = Field(..., description="Whether MongoDB is connected")

class User(PreferenceResponse):
    username:str
    email: Optional[str] = None
    hashed_password: str
    is_active: bool = Field(default=True)
    created_at :datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = Field(None)

class UserCreate(BaseModel):
    username:str
    email: Optional[str] = Field(None)
    password:str
    user_id: str = Field(..., description="User ID")

class UserLogin(BaseModel):
    username: str = Field(..., description="Username for login")
    password: str = Field(..., description="Password for login")

#### Security
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create an access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(username:str) -> User | None:
    try:
        user_data = mongo_memory.get_user_by_username(username)

        if user_data:
            return User(**user_data)
        return None
    except Exception as e:
        print(f"Error retrieving user {username}: {str(e)}")
        return None

def authenticate_user(username: str, password: str) -> User | None:
    if not username or not password:
        return None
    
    user = get_user(username)

    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    
    return user

def create_user(user_create: UserCreate) -> User | None:
    if mongo_memory.username_exists(user_create.username):
        raise ValueError(f"User {user_create.username} already exists.")
    
    hashed_password = get_password_hash(user_create.password)

    user_data = {
        "username": user_create.username,
        "email":user_create.email,
        "hashed_password":hashed_password,
        "user_id": user_create.user_id,
        "is_active":True,
        "created_at": datetime.now(),
        "last_login": None
    }

    try:
        created_user = mongo_memory.create_user(user_data)

        if created_user:
            return User(**created_user)
    except Exception as e:
        print(f"Error creating user in MongoDB:{str(e)}")
        raise ValueError("Failed to create user in database.")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """Get current authenticated user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Helper function to determine which route was used
def get_route_from_state(state: Dict) -> str:
    """Determine which route was used based on the state"""
    question = state.get("question", "").lower()
    document_path = state.get("document_path", "")
    
    if document_path and any(word in question for word in ["what", "how", "why", "when", "where", "explain", "tell me", "describe"]):
        return "rag_query"
    elif "summarize" in question and ("document" in question or "doc" in question) and document_path:
        return "summarize_doc"
    elif "summarize" in question and ("chat" in question or "conversation" in question):
        return "summarize_chat"
    else:
        return "general_response"

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Agent Two API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        stats = mongo_memory.get_memory_stats()
        memory_connected = bool(stats)
        
        return HealthResponse(
            status="healthy" if memory_connected else "degraded",
            timestamp=datetime.now(),
            memory_connected=memory_connected
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            memory_connected=False
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for interacting with Agent Two"""
    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or f"session_{request.user_id}_{datetime.now().timestamp()}"
        
        # Create configuration for the agent
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create initial state
        initial_state = {
            "messages": [],  # Fresh each time as per the agent design
            "question": request.question,
            "document_path": request.document_path,
            "user_id": request.user_id
        }
        
        # Invoke the agent
        result = agent_app.invoke(initial_state, config=config)
        
        # Extract the response
        if "messages" in result and result["messages"]:
            latest_message = result["messages"][-1].content
        else:
            latest_message = "No response generated."
        
        # Determine which route was used
        route_used = get_route_from_state(initial_state)
        
        return ChatResponse(
            response=latest_message,
            user_id=request.user_id,
            thread_id=thread_id,
            timestamp=datetime.now(),
            route_used=route_used
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post("/preferences", response_model=PreferenceResponse)
async def set_preference(request: PreferenceRequest):
    """Set a user preference"""
    try:
        mongo_memory.save_user_preference(request.user_id, request.key, request.value)
        
        return PreferenceResponse(
            user_id=request.user_id,
            key=request.key,
            value=request.value,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving preference: {str(e)}"
        )

@app.get("/preferences/{user_id}", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: str):
    """Get all preferences for a user"""
    try:
        preferences = mongo_memory.get_user_preferences(user_id)
        
        return UserPreferencesResponse(
            user_id=user_id,
            preferences=preferences
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving preferences: {str(e)}"
        )

@app.get("/preferences/{user_id}/{key}")
async def get_user_preference(user_id: str, key: str, default: Any = None):
    """Get a specific preference for a user"""
    try:
        value = mongo_memory.get_user_preference(user_id, key, default)
        
        return {
            "user_id": user_id,
            "key": key,
            "value": value
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving preference: {str(e)}"
        )

@app.delete("/preferences/{user_id}/{key}")
async def delete_user_preference(user_id: str, key: str):
    """Delete a specific preference for a user"""
    try:
        # Get current preferences
        preferences = mongo_memory.get_user_preferences(user_id)
        
        if key not in preferences:
            raise HTTPException(
                status_code=404,
                detail=f"Preference '{key}' not found for user '{user_id}'"
            )
        
        # Remove the preference by updating without that key
        updated_preferences = {k: v for k, v in preferences.items() if k != key}
        
        # Save updated preferences (this is a workaround since we don't have a delete method)
        # We'll update the document to remove the key
        mongo_memory.long_memory.update_one(
            {"user_id": user_id},
            {"$unset": {f"preferences.{key}": ""}}
        )
        
        return {"message": f"Preference '{key}' deleted for user '{user_id}'"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting preference: {str(e)}"
        )

@app.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats():
    """Get memory system statistics"""
    try:
        stats = mongo_memory.get_memory_stats()
        
        return MemoryStatsResponse(
            long_memory_users=stats.get("long_memory_users", 0),
            memory_type=stats.get("memory_type", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )

##### Authentication Endpoints
@app.post("/register",response_model = User)
async def register_user(user_data: UserCreate):
    try:
        user = create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")
    
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """Login endpoint to get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Update last login time in MongoDB
    mongo_memory.update_user_last_login(user.username)
    
    return Token(access_token=access_token, token_type="bearer")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
