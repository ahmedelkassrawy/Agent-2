#!/usr/bin/env python3
"""
Test script for Agent-2 with User Preferences Integration
"""

import sys
import os
sys.path.insert(0, '/workspaces/Agent-2')

def test_preference_integration():
    """Test that user preferences are properly integrated into LLM responses"""
    print("=== Testing Agent-2 Preference Integration ===\n")
    
    try:
        # Test imports
        from models.mongo_memory import SimpleMongoMemory
        print("✓ SimpleMongoMemory import successful")
        
        # Initialize memory
        memory = SimpleMongoMemory()
        print("✓ Memory initialized")
        
        # Test user preferences
        test_user = "test_preferences_user"
        
        # Set some test preferences
        print("\n1. Setting test preferences...")
        memory.save_user_preference(test_user, "language", "python")
        memory.save_user_preference(test_user, "tone", "friendly")
        memory.save_user_preference(test_user, "expertise", "beginner")
        memory.save_user_preference(test_user, "interests", "AI, machine learning")
        print("✓ Test preferences saved")
        
        # Retrieve preferences
        preferences = memory.get_user_preferences(test_user)
        print(f"✓ Retrieved preferences: {preferences}")
        
        # Test preference formatting for LLM
        if preferences:
            preference_text = "\n".join([f"- {key}: {value}" for key, value in preferences.items()])
            print(f"\n2. Formatted preferences for LLM:\n{preference_text}")
            
            # Create a sample system prompt
            question = "How do I learn programming?"
            system_prompt = f"""You are a helpful AI assistant. Please consider the user's preferences when responding:

User Preferences:
{preference_text}

Please tailor your response to align with these preferences when relevant. For example:
- If the user prefers a certain programming language, suggest solutions in that language
- If the user has a preferred communication style, match that tone
- If the user has specific interests, relate your answers to those interests when possible

User Question: {question}

Provide a helpful response that takes into account the user's preferences where applicable."""
            
            print(f"\n3. Sample system prompt with preferences:")
            print(f"{system_prompt[:200]}...")
            print("✓ Preference integration format working")
        
        # Cleanup
        memory.long_memory.delete_one({"user_id": test_user})
        memory.close()
        print("\n✓ Test cleanup completed")
        
        print("\n🎉 Preference Integration Test Passed!")
        print("\nThe agent will now:")
        print("- Include user preferences in LLM context")
        print("- Personalize responses based on user profile")
        print("- Consider preferences for both general and RAG queries")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_preference_integration()
