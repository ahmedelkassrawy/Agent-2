"""
Simple MongoDB Long Memory for Agent-2
- Long Memory Only: User profile and preferences (persistent)
- No short-term message storage
"""

from typing import Any, Dict, Optional, Sequence, Tuple, List
from pymongo import MongoClient
from datetime import datetime
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage

class SimpleMongoMemory(BaseCheckpointSaver):
    """
    Simple MongoDB long memory only:
    - Long Memory: User profile and preferences (persistent)
    - No short-term message storage
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "agent_memory"
    ):
        """
        Initialize simple MongoDB long memory.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Name of the database to use
        """
        super().__init__()
        self.connection_string = connection_string
        self.database_name = database_name
        
        # Initialize MongoDB connection
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.long_memory = self.db["long_memory"]    # User preferences
        self.users = self.db["users"]                # User accounts
        
        # Create indexes
        self.long_memory.create_index("user_id")
        self.users.create_index("username", unique=True)
        self.users.create_index("user_id", unique=True)
    
    
    # === LONG MEMORY METHODS ===
    def save_user_preference(self, 
                             user_id: str, 
                             key: str, 
                             value: Any):
        """Save user preference to long memory."""
        try:
            self.long_memory.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"preferences.{key}": value,
                        "updated_at": datetime.now()
                    },
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.now()
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"Error saving user preference: {e}")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences from long memory."""
        try:
            doc = self.long_memory.find_one({"user_id": user_id})
            if doc:
                return doc.get("preferences", {})
            return {}
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {}
    
    def get_user_preference(self, 
                            user_id: str, 
                            key: str, 
                            default: Any = None) -> Any:
        """Get specific user preference."""
        preferences = self.get_user_preferences(user_id)
        return preferences.get(key, default)
    
    # === USER MANAGEMENT METHODS ===
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in MongoDB."""
        try:
            user_data["created_at"] = datetime.now()
            user_data["updated_at"] = datetime.now()
            
            # Insert user
            result = self.users.insert_one(user_data)
            
            # Return the created user
            created_user = self.users.find_one({"_id": result.inserted_id})
            if created_user:
                # Convert ObjectId to string for JSON serialization
                created_user["_id"] = str(created_user["_id"])
            return created_user
            
        except Exception as e:
            print(f"Error creating user: {e}")
            raise e
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username from MongoDB."""
        try:
            user = self.users.find_one({"username": username})
            if user:
                # Convert ObjectId to string for JSON serialization
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id from MongoDB."""
        try:
            user = self.users.find_one({"user_id": user_id})
            if user:
                # Convert ObjectId to string for JSON serialization
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    def update_user_last_login(self, username: str) -> bool:
        """Update user's last login timestamp."""
        try:
            result = self.users.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.now()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating user last login: {e}")
            return False
    
    def username_exists(self, username: str) -> bool:
        """Check if username already exists."""
        try:
            return self.users.count_documents({"username": username}) > 0
        except Exception as e:
            print(f"Error checking username existence: {e}")
            return False
    
    def user_id_exists(self, user_id: str) -> bool:
        """Check if user_id already exists."""
        try:
            return self.users.count_documents({"user_id": user_id}) > 0
        except Exception as e:
            print(f"Error checking user_id existence: {e}")
            return False
    
    # === LANGGRAPH CHECKPOINT INTERFACE ===
    
    def get_tuple(self, config: RunnableConfig) -> Optional[Tuple[str, Checkpoint, CheckpointMetadata]]:
        """Get checkpoint - returns None as we don't store messages."""
        return None
    
    def list(self, config: RunnableConfig, **kwargs) -> Sequence[Tuple[str, Checkpoint, CheckpointMetadata]]:
        """List checkpoints - returns empty as we don't store messages."""
        return []
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """Save checkpoint - does nothing as we don't store messages."""
        return config
    
    # === UTILITY METHODS ===
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics including user counts."""
        try:
            return {
                "long_memory_users": self.long_memory.count_documents({}),
                "total_users": self.users.count_documents({}),
                "memory_type": "MongoDB Long Memory + User Management"
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
