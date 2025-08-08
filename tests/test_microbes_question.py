#!/usr/bin/env python3
"""
Test agent with microbes question
"""

import sys
import os
sys.path.insert(0, '/workspaces/Agent-2')

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB9o34YFREb_JLj7nXdfNwHfu5Pw9M-Hpw"

def test_microbes_question():
    """Test the agent with the microbes question"""
    print("=== Testing Microbes Question ===\n")
    
    try:
        from src.agent_two import app, mongo_memory
        
        user_id = "test_user"
        question = "tell me about microbes"
        
        # Set some preferences for testing
        mongo_memory.save_user_preference(user_id, "tone", "friendly")
        mongo_memory.save_user_preference(user_id, "expertise", "beginner")
        mongo_memory.save_user_preference(user_id, "interests", "biology, science")
        
        print(f"📝 Question: {question}")
        print(f"👤 User: {user_id}")
        
        # Get user preferences
        prefs = mongo_memory.get_user_preferences(user_id)
        print(f"🎯 Preferences: {prefs}")
        
        # Create initial state
        initial_state = {
            "messages": [],
            "question": question,
            "document_path": None,
            "user_id": user_id
        }
        
        config = {"configurable": {"thread_id": f"session_{user_id}"}}
        
        print("\n🤖 Processing with Agent-2...")
        result = app.invoke(initial_state, config=config)
        
        if "messages" in result and result["messages"]:
            response = result["messages"][-1].content
            print(f"\n✅ AI Response:\n{response}")
        else:
            print("❌ No response received")
        
        # Cleanup
        mongo_memory.long_memory.delete_one({"user_id": user_id})
        print("\n🧹 Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_microbes_question()
