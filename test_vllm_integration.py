#!/usr/bin/env python3
"""
Test script for vLLM integration in continuous_trellis_orchestrator_lora_working.py
"""

import sys
import os

# Add the current directory to the path so we can import the vLLM functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the vLLM functions
def test_vllm_functions():
    print("🧪 Testing vLLM integration functions...")
    
    try:
        # Test connection
        print("\n1. Testing vLLM connection...")
        from continuous_trellis_orchestrator_lora_working import test_vllm_connection
        connection_success = test_vllm_connection(11300)
        print(f"   Connection test: {'✅ SUCCESS' if connection_success else '❌ FAILED'}")
        
        if connection_success:
            # Test optimization with different methods
            test_prompt = "a golden statue"
            
            print(f"\n2. Testing vLLM optimization with prompt: '{test_prompt}'")
            
            # Test system chat method
            print("\n   Testing system_chat method...")
            from continuous_trellis_orchestrator_lora_working import query_vllm_with_system_prompt_chat
            result = query_vllm_with_system_prompt_chat(test_prompt, 11300)
            if result:
                print(f"   ✅ System chat optimization: '{result[:50]}...'")
            else:
                print("   ❌ System chat optimization failed")
            
            # Test system completions method
            print("\n   Testing system_completions method...")
            from continuous_trellis_orchestrator_lora_working import query_vllm_with_system_prompt_completions
            result = query_vllm_with_system_prompt_completions(test_prompt, 11300)
            if result:
                print(f"   ✅ System completions optimization: '{result[:50]}...'")
            else:
                print("   ❌ System completions optimization failed")
            
            # Test no system method
            print("\n   Testing no_system method...")
            from continuous_trellis_orchestrator_lora_working import query_vllm_no_system_prompt
            result = query_vllm_no_system_prompt(test_prompt, 11300)
            if result:
                print(f"   ✅ No system optimization: '{result[:50]}...'")
            else:
                print("   ❌ No system optimization failed")
            
            # Test the main optimization function
            print("\n3. Testing main vLLM optimization function...")
            from continuous_trellis_orchestrator_lora_working import optimize_prompt_with_vllm
            
            for priority in ['system_chat', 'system_completions', 'no_system']:
                print(f"\n   Testing priority: {priority}")
                result = optimize_prompt_with_vllm(test_prompt, 11300, priority)
                if result:
                    print(f"   ✅ {priority} optimization: '{result[:50]}...'")
                else:
                    print(f"   ❌ {priority} optimization failed")
        else:
            print("   ⚠️ Skipping optimization tests due to connection failure")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the same directory as continuous_trellis_orchestrator_lora_working.py")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_functions()

