#!/usr/bin/env python3
"""
Demo script showing Agent-2 with User Preference Integration
"""

import sys
import os
sys.path.insert(0, '/workspaces/Agent-2')

from models.mongo_memory import SimpleMongoMemory

def demo_preferences():
    """Demonstrate how preferences work"""
    print("🚀 Agent-2 User Preference Integration Demo")
    print("=" * 50)
    
    # Initialize memory
    memory = SimpleMongoMemory()
    demo_user = "demo_user"
    
    print("\n1. Setting User Preferences...")
    preferences = {
        "language": "python",
        "tone": "friendly", 
        "expertise": "intermediate",
        "communication_style": "detailed",
        "interests": "AI, machine learning, data science",
        "output_format": "code examples with explanations"
    }
    
    for key, value in preferences.items():
        memory.save_user_preference(demo_user, key, value)
        print(f"   ✓ {key}: {value}")
    
    print(f"\n2. User Profile for '{demo_user}':")
    stored_prefs = memory.get_user_preferences(demo_user)
    for key, value in stored_prefs.items():
        print(f"   • {key}: {value}")
    
    print("\n3. How AI Will Use These Preferences:")
    print("   📝 When user asks: 'How do I sort a list?'")
    print("   🤖 AI will:")
    print("   - Use Python examples (language preference)")
    print("   - Provide friendly, detailed explanations (tone + communication style)")
    print("   - Include intermediate-level concepts (expertise level)")
    print("   - Show code examples with explanations (output format)")
    print("   - Relate to data science if relevant (interests)")
    
    print("\n4. Sample AI Context (what gets sent to LLM):")
    print("   " + "─" * 45)
    
    preference_text = "\n".join([f"   - {key}: {value}" for key, value in stored_prefs.items()])
    sample_prompt = f"""   System: You are a helpful AI assistant. Consider these user preferences:
   
   User Preferences:
{preference_text}
   
   Tailor your response accordingly...
   
   User Question: How do I sort a list?
   """
    
    print(sample_prompt)
    print("   " + "─" * 45)
    
    print("\n5. Benefits:")
    print("   ✅ Personalized responses every time")
    print("   ✅ Consistent user experience")
    print("   ✅ No need to repeat preferences")
    print("   ✅ Works across all agent features (RAG, general chat, etc.)")
    
    # Cleanup
    memory.long_memory.delete_one({"user_id": demo_user})
    memory.close()
    
    print(f"\n🎯 Ready to use! Run: python src/agent_two.py")

if __name__ == "__main__":
    demo_preferences()
