#!/usr/bin/env python3
"""
Debug AI Responses - See what DeepSeek is actually saying
"""
import requests
import json

def query_ai(messages):
    """Simple AI query"""
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=data, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"

def test_ai_decision_making():
    """Test what AI actually responds with"""
    
    strategies = ["raw", "material_focus", "geometric_focus", "basic_description", "enhanced_clarity", "concrete_object", "professional_render", "high_quality"]
    
    context = """OPTIMIZATION CONTEXT:
Prompt: "hexagonal prism steel structure" (Category: technical_description)
Baseline: 0.566
Attempt: 3/6
Targets: Min 0.6 | Target 0.9 | Ultra 0.96

RECENT ATTEMPTS:
Attempt 1: enhanced_clarity -> 0.522 (-0.044)
Attempt 2: enhanced_clarity -> 0.276 (-0.291)

STRATEGY PERFORMANCE (for technical_description):
Limited historical data

ANTI-REPETITION:
Recently used strategies to avoid: enhanced_clarity
Recently used decision types: SELECT_STRATEGY"""

    system_prompt = f"""You are a CREATIVE PROMPTER AI focused on breaking through optimization barriers.

Your approach:
- Generate completely new custom prompts when stuck
- Think outside the box with creative interpretations
- Combine concepts in novel ways
- Take calculated risks for breakthrough results

You are an expert 3D prompt optimization AI.

{context}

AVAILABLE STRATEGIES: {strategies}

Your goal is to reach the targets through smart decisions. Respond in this format:
DECISION: [CUSTOM_PROMPT|SELECT_STRATEGY|STRATEGY_SEQUENCE|EARLY_STOP]
REASONING: [Your analysis and justification]
CONFIDENCE: [0.1 to 1.0]
EXPECTED_IMPROVEMENT: [0.0 to 0.5]
CONTENT: [Custom prompt OR strategy name OR strategy list OR stop reason]

Important: Avoid repeating recently failed approaches. Be creative and strategic."""

    user_message = "Based on the context, make your optimization decision for attempt 3. Think about what approach would be most effective given the current situation."
    
    print("🔍 TESTING AI DECISION MAKING")
    print("=" * 60)
    print("📝 System Prompt (excerpt):")
    print(system_prompt[-200:])
    print("\n📝 User Message:")
    print(user_message)
    print("\n🤖 AI Response:")
    print("=" * 60)
    
    for i in range(3):
        print(f"\n--- Response {i+1} ---")
        response = query_ai([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])
        print(response)
        print("-" * 40)

if __name__ == "__main__":
    test_ai_decision_making() 