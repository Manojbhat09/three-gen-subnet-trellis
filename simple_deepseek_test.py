#!/usr/bin/env python3
"""
Simple DeepSeek Test
===================
Debug why DeepSeek is returning empty responses in conversation mode
"""

import requests
import json

def test_deepseek_simple():
    """Test DeepSeek with simple single message"""
    
    print("🔍 Testing DeepSeek with simple message...")
    
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "Generate a prompt starting with 'wbgmsst, ' for hexagonal prism steel structure"}],
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
            "num_predict": 200,
            "repeat_penalty": 1.15
        }
    }
    
    response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
    result = response.json()
    
    print(f"Response status: {response.status_code}")
    print(f"Content length: {len(result['message']['content'])}")
    print(f"Content: {result['message']['content']}")

def test_deepseek_conversation():
    """Test DeepSeek with conversation history"""
    
    print("\n🔍 Testing DeepSeek with conversation...")
    
    conversation = [
        {"role": "user", "content": "You are a prompt optimizer. Generate prompts starting with 'wbgmsst, ' and ending with ', white background'."},
        {"role": "assistant", "content": "I understand! I'll generate prompts that start with 'wbgmsst, ' and end with ', white background'. What would you like me to create a prompt for?"},
        {"role": "user", "content": "Generate a prompt for hexagonal prism steel structure"}
    ]
    
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": conversation,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
            "num_predict": 200,
            "repeat_penalty": 1.15
        }
    }
    
    response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
    result = response.json()
    
    print(f"Response status: {response.status_code}")
    print(f"Content length: {len(result['message']['content'])}")
    print(f"Content: {result['message']['content']}")

def test_deepseek_long_conversation():
    """Test DeepSeek with longer conversation"""
    
    print("\n🔍 Testing DeepSeek with long conversation...")
    
    # Simulate a long conversation like in our optimizer
    conversation = []
    for i in range(10):
        conversation.append({"role": "user", "content": f"This is message {i+1}. Generate a prompt for hexagonal prism."})
        conversation.append({"role": "assistant", "content": f"wbgmsst, ultra-precision hexagonal prism steel structure attempt {i+1}, white background"})
    
    conversation.append({"role": "user", "content": "Generate your next optimized prompt:"})
    
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": conversation,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
            "num_predict": 200,
            "repeat_penalty": 1.15
        }
    }
    
    print(f"Conversation length: {len(conversation)} messages")
    print(f"Total conversation chars: {sum(len(msg['content']) for msg in conversation)}")
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
        result = response.json()
        
        print(f"Response status: {response.status_code}")
        print(f"Content length: {len(result['message']['content'])}")
        print(f"Content: {result['message']['content'][:200]}{'...' if len(result['message']['content']) > 200 else ''}")
    except Exception as e:
        print(f"ERROR: {e}")

def test_deepseek_with_stops():
    """Test if stop tokens are causing issues"""
    
    print("\n🔍 Testing DeepSeek without stop tokens...")
    
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": "Generate a prompt starting with 'wbgmsst, ' for hexagonal prism steel structure"}],
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
            "num_predict": 200,
            "repeat_penalty": 1.15
            # No stop tokens
        }
    }
    
    response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
    result = response.json()
    
    print(f"Response status: {response.status_code}")
    print(f"Content length: {len(result['message']['content'])}")
    print(f"Content: {result['message']['content']}")

if __name__ == "__main__":
    test_deepseek_simple()
    test_deepseek_conversation()
    test_deepseek_long_conversation()
    test_deepseek_with_stops() 