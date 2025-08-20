#!/usr/bin/env python3
"""
Test vLLM Connectivity and LLM Behavior

This script tests if vLLM is working and if the LLM can follow instructions properly.
"""

import requests
import json

def test_vllm_connectivity():
    """Test if vLLM is accessible"""
    print("🔍 Testing vLLM Connectivity...")
    
    # Test different vLLM ports
    ports = [9000, 9001, 9002, 9004]
    
    for port in ports:
        try:
            url = f"http://localhost:{port}/health"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ Port {port}: vLLM is running and healthy")
                return port
            else:
                print(f"   ⚠️ Port {port}: vLLM responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Port {port}: Connection failed")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Port {port}: Request timed out")
        except Exception as e:
            print(f"   ❌ Port {port}: Error: {e}")
    
    return None

def test_vllm_completion(port):
    """Test if vLLM can complete a simple prompt"""
    print(f"\n🧪 Testing vLLM Completion on port {port}...")
    
    url = f"http://localhost:{port}/v1/completions"
    
    # Simple test prompt
    payload = {
        "model": "llama-3-2-3b-it",
        "prompt": "Complete this sentence: The cat is",
        "max_tokens": 20,
        "temperature": 0.1,
        "stop": ["."]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            completion = result['choices'][0]['text']
            print(f"   ✅ LLM Response: '{completion}'")
            return True
        else:
            print(f"   ❌ LLM Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ LLM Request failed: {e}")
        return False

def test_vllm_chat(port):
    """Test if vLLM can handle chat format with system prompts"""
    print(f"\n💬 Testing vLLM Chat on port {port}...")
    
    url = f"http://localhost:{port}/v1/chat/completions"
    
    # Test with a system prompt that should be easy to follow
    payload = {
        "model": "llama-3-2-3b-it",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond with exactly 'Hello World' and nothing else."
            },
            {
                "role": "user", 
                "content": "What is 2+2?"
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"   ✅ LLM Response: '{message}'")
            
            # Check if it followed instructions
            if "Hello World" in message:
                print(f"   ✅ LLM followed system prompt correctly")
                return True
            else:
                print(f"   ⚠️ LLM did not follow system prompt exactly")
                return False
        else:
            print(f"   ❌ LLM Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ LLM Request failed: {e}")
        return False

def test_instruction_following(port):
    """Test if the LLM can follow complex instructions"""
    print(f"\n📝 Testing LLM Instruction Following on port {port}...")
    
    url = f"http://localhost:{port}/v1/chat/completions"
    
    # Test with the exact system prompt we use in reproducibility
    system_prompt = """You are a prompt assembly agent. Your task is to reconstruct a high-quality 3D model prompt from a structured set of components.

**CRITICAL INSTRUCTIONS:**
1. ALWAYS start with the `core_subject` - this is the main object and should NEVER be changed
2. Integrate the phrases from the `enhancements` list to describe the subject
3. Combine these components into a single, coherent, and descriptive sentence
4. Ensure the prompt flows naturally and maintains the quality of the original components

**CRITICAL CONSTRAINTS:**
- The `core_subject` MUST remain exactly as provided - do not change it
- The final output must end with `, white background`
- Do not invent hyper-specific details the 3D model cannot render
- Provide only the final prompt without explanation
- Keep it concise but closely related to the original prompt's intent
- Use enhancements to improve quality, not to change the subject

**Components to use:**
{
  "core_subject": "sandstone cat sculpture",
  "enhancements": {
    "quality_adjectives": ["intricate", "bubbly"],
    "material_details": ["sandstone", "glass"],
    "light_interaction": [],
    "context": []
  }
}

**Reconstructed Prompt:**"""
    
    payload = {
        "model": "llama-3-2-3b-it",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"   ✅ LLM Response: '{message}'")
            
            # Check if it followed the critical instructions
            if "sandstone cat sculpture" in message.lower():
                print(f"   ✅ LLM preserved the core subject")
            else:
                print(f"   ❌ LLM did NOT preserve the core subject")
                
            if "white background" in message.lower():
                print(f"   ✅ LLM added white background")
            else:
                print(f"   ❌ LLM did NOT add white background")
                
            return True
        else:
            print(f"   ❌ LLM Error: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ LLM Request failed: {e}")
        return False

def main():
    """Run all connectivity tests"""
    print("🚀 vLLM Connectivity and LLM Behavior Test")
    print("=" * 60)
    
    # Test connectivity
    working_port = test_vllm_connectivity()
    
    if working_port is None:
        print("\n❌ No working vLLM ports found!")
        return
    
    print(f"\n✅ Found working vLLM on port {working_port}")
    
    # Test basic completion
    completion_works = test_vllm_completion(working_port)
    
    # Test chat format
    chat_works = test_vllm_chat(working_port)
    
    # Test instruction following
    instruction_works = test_instruction_following(working_port)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   vLLM Connectivity: ✅ (port {working_port})")
    print(f"   Basic Completion: {'✅' if completion_works else '❌'}")
    print(f"   Chat Format: {'✅' if chat_works else '❌'}")
    print(f"   Instruction Following: {'✅' if instruction_works else '❌'}")
    
    if completion_works and chat_works and instruction_works:
        print(f"\n🎯 All tests passed! vLLM is working correctly.")
        print(f"   The issue with reproducibility might be in the prompt format or model behavior.")
    else:
        print(f"\n⚠️ Some tests failed. vLLM has issues that need to be resolved.")
        print(f"   Check the model configuration and ensure it's properly loaded.")

if __name__ == "__main__":
    main()
