#!/usr/bin/env python3
"""
Quick test of Agent-2 with preferences
"""

import sys
import os
sys.path.insert(0, '/workspaces/Agent-2')

def test_agent_simple():
    """Test the agent with a simple question"""
    print("=== Testing Agent-2 with Preferences ===\n")
    
    # Set API key first
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB9o34YFREb_JLj7nXdfNwHfu5Pw9M-Hpw"
    print("✓ API key set")
    
    try:
        # Import and test basic components
        from models.mongo_memory import SimpleMongoMemory
        print("✓ Memory import successful")
        
        # Test memory
        memory = SimpleMongoMemory()
        test_user = "test_agent_user"
        
        # Set a preference
        memory.save_user_preference(test_user, "language", "python")
        memory.save_user_preference(test_user, "tone", "friendly")
        print("✓ Test preferences set")
        
        # Get preferences
        prefs = memory.get_user_preferences(test_user)
        print(f"✓ Retrieved preferences: {prefs}")
        
        # Test LLM components
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Test with correct model name
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
        )
        print("✓ LLM initialized")
        
        # Test basic LLM call
        test_question = "What is Python?"
        
        # Create preference-aware prompt
        preference_text = "\n".join([f"- {key}: {value}" for key, value in prefs.items()])
        system_content = f"""You are a helpful AI assistant. Please consider the user's preferences when responding:

User Preferences:
{preference_text}

Please tailor your response to align with these preferences."""
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=test_question)
        ]
        
        print(f"\n📝 Testing question: {test_question}")
        print(f"🎯 With preferences: {prefs}")
        
        response = llm.invoke(messages)
        print(f"\n🤖 AI Response: {response.content[:200]}...")
        print("✓ LLM call successful with preferences!")
        
        # Cleanup
        memory.long_memory.delete_one({"user_id": test_user})
        memory.close()
        print("\n✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_agent_simple()
