#!/usr/bin/env python3
"""
Simple test script to verify episodic memory loading.
"""

import json

def test_episodic_memory_loading():
    """Test loading the episodic memory file."""
    try:
        with open('episodic_logs_cinema/episodic_memory.json', 'r') as f:
            data = json.load(f)
        
        print("✅ Successfully loaded episodic memory!")
        print(f"📊 Found {len(data.get('optimization_sessions', []))} optimization sessions")
        
        # Show first few sessions
        for i, session in enumerate(data.get('optimization_sessions', [])[:3]):
            print(f"\nSession {i+1}:")
            print(f"  Original: {session.get('original_prompt', 'N/A')}")
            print(f"  Best: {session.get('final_best_prompt', 'N/A')[:50]}...")
            print(f"  Score: {session.get('final_best_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading episodic memory: {e}")
        return False

if __name__ == "__main__":
    test_episodic_memory_loading()

