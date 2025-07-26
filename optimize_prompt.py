#!/usr/bin/env python3
"""
Prompt Optimizer - Production Inference Script
==============================================
🚀 Fast, one-shot prompt optimization for production use
⚡ Loads golden examples and optimizes prompts instantly
📈 No training loops, no complex state - just fast results

Usage:
    python optimize_prompt.py "your prompt here"
    
Or import and use programmatically:
    from optimize_prompt import PromptOptimizer
    optimizer = PromptOptimizer()
    result = optimizer.optimize("your prompt")
"""

import json
import requests
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

class PromptOptimizer:
    """Fast, lightweight prompt optimizer for production"""
    
    def __init__(self, golden_examples_file: str = "golden_examples.json", 
                 ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.golden_examples = self._load_golden_examples(golden_examples_file)
        
        self._test_connection()
        print(f"⚡ PROMPT OPTIMIZER READY (Loaded {len(self.golden_examples)} golden examples)")
    
    def _test_connection(self):
        """Test LLaMA connection"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ LLaMA 3.2 unavailable: {e}")
            return False
    
    def _load_golden_examples(self, examples_file: str) -> List[Dict]:
        """Load golden examples from file"""
        if Path(examples_file).exists():
            with open(examples_file, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ Golden examples file not found: {examples_file}")
            print(f"💡 Run 'python train_optimizer.py' first to generate golden examples")
            return self._get_fallback_examples()
    
    def _get_fallback_examples(self) -> List[Dict]:
        """Minimal fallback examples if no golden file exists"""
        return [
            {
                "original": "tall glass of layered lemonade",
                "optimized": "wbgmsst, crystal-clear artisanal glass of layered lemonade with golden citrus highlights, pristine transparency, white background",
                "score": 0.90,
                "category": "beverages",
                "principle": "Enhanced with clarity and transparency details"
            },
            {
                "original": "emerald pendant",
                "optimized": "wbgmsst, exquisite handcrafted emerald pendant with flawless gemstone clarity, luxurious finish, white background",
                "score": 0.93,
                "category": "jewelry", 
                "principle": "Added luxury materials and premium quality"
            }
        ]
    
    def optimize(self, user_prompt: str, temperature: float = 0.7) -> Dict:
        """
        Optimize a prompt in one shot
        
        Args:
            user_prompt: The prompt to optimize
            temperature: Creativity level (0.1-1.0)
        
        Returns:
            Dict with optimized_prompt, confidence, processing_time
        """
        start_time = time.time()
        
        print(f"🎯 Optimizing: '{user_prompt}'")
        
        # Build the golden system prompt
        system_prompt = self._build_golden_system_prompt()
        
        # Build user prompt
        user_optimization_prompt = f"""OPTIMIZE THIS PROMPT: "{user_prompt}"

Requirements:
- Must start with "wbgmsst," and end with ", white background"
- Enhance with contextually appropriate details
- Maintain the core object identity
- Add material, quality, and visual enhancements
- Make it vivid and specific for 3D generation

Provide only the optimized prompt - no explanations."""
        
        try:
            # Single LLaMA call for optimization
            optimized_prompt = self._query_llama(system_prompt, user_optimization_prompt, temperature)
            confidence = self._calculate_confidence(user_prompt, optimized_prompt)
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            optimized_prompt = self._generate_fallback(user_prompt)
            confidence = 0.6
        
        processing_time = time.time() - start_time
        
        result = {
            'original_prompt': user_prompt,
            'optimized_prompt': optimized_prompt,
            'confidence': confidence,
            'processing_time': processing_time
        }
        
        print(f"✅ Result: {optimized_prompt}")
        print(f"⏱️  Time: {processing_time:.2f}s | Confidence: {confidence:.1%}")
        
        return result
    
    def _build_golden_system_prompt(self) -> str:
        """Build system prompt with golden examples"""
        
        prompt = """You are an expert prompt optimizer for 3D asset generation. Your task is to enhance prompts to be more detailed and visually rich for maximum scores.

OPTIMIZATION RULES:
1. ALWAYS start with "wbgmsst," and end with ", white background"
2. Keep the core object intact - enhance, don't replace
3. Add contextually appropriate material and quality details
4. Use specific, vivid descriptors that fit the object type
5. Don't use technical terms for organic objects or cute terms for industrial objects

Here are PROVEN HIGH-SCORING EXAMPLES. Apply the same enhancement principles:

"""
        
        # Add golden examples by category
        categories = {}
        for example in self.golden_examples:
            cat = example['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(example)
        
        # Show top examples from each category
        for category, examples in categories.items():
            if examples:
                prompt += f"\n--- {category.upper()} ---\n"
                # Sort by score and take top 2
                top_examples = sorted(examples, key=lambda x: x['score'], reverse=True)[:2]
                for ex in top_examples:
                    prompt += f"Original: \"{ex['original']}\"\n"
                    prompt += f"Optimized: \"{ex['optimized']}\" (Score: {ex['score']:.2f})\n\n"
        
        prompt += """CRITICAL: Your enhancement must be APPROPRIATE for the object type. Study the examples above and apply similar enhancement principles to the user's prompt.

Only output the optimized prompt - no analysis or explanations."""
        
        return prompt
    
    def _query_llama(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Query LLaMA for optimization"""
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 150  # Keep it concise
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=20)
        response.raise_for_status()
        
        content = response.json()["message"]["content"].strip()
        return self._clean_prompt(content)
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and validate prompt format"""
        # Remove quotes and extra text
        lines = prompt.split('\n')
        
        # Find the line with wbgmsst
        for line in lines:
            if 'wbgmsst' in line.lower():
                prompt = line.strip()
                break
        else:
            # If no wbgmsst found, use the first non-empty line
            for line in lines:
                if line.strip():
                    prompt = line.strip()
                    break
        
        prompt = prompt.replace('"', '').strip()
        
        # Ensure proper format
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        
        if not prompt.endswith('white background'):
            if prompt.endswith(','):
                prompt += " white background"
            else:
                prompt += ", white background"
        
        return prompt
    
    def _calculate_confidence(self, original: str, optimized: str) -> float:
        """Calculate confidence based on enhancement quality"""
        confidence = 0.5
        
        # Length increase suggests more detail
        length_ratio = len(optimized) / max(len(original), 1)
        if length_ratio > 2:
            confidence += 0.2
        elif length_ratio > 1.5:
            confidence += 0.1
        
        # Check for quality indicators
        quality_words = ['artisanal', 'precision', 'premium', 'exquisite', 'crystal-clear', 'flawless', 'luxury']
        quality_count = sum(1 for word in quality_words if word.lower() in optimized.lower())
        confidence += min(quality_count * 0.05, 0.2)
        
        # Proper format
        if optimized.startswith('wbgmsst') and optimized.endswith('white background'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_fallback(self, user_prompt: str) -> str:
        """Generate fallback if LLaMA fails"""
        prompt_lower = user_prompt.lower()
        
        if any(word in prompt_lower for word in ['glass', 'drink', 'beverage']):
            return f"wbgmsst, crystal-clear artisanal {user_prompt}, pristine presentation, white background"
        elif any(word in prompt_lower for word in ['jewelry', 'pendant', 'gemstone']):
            return f"wbgmsst, exquisite luxury {user_prompt}, flawless craftsmanship, white background"
        else:
            return f"wbgmsst, premium quality {user_prompt}, masterwork finish, white background"

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python optimize_prompt.py \"your prompt here\"")
        print("\nExample:")
        print("python optimize_prompt.py \"tall glass of layered lemonade\"")
        return
    
    user_prompt = sys.argv[1]
    
    print("🚀 PROMPT OPTIMIZER - PRODUCTION INFERENCE")
    print("=" * 50)
    
    try:
        optimizer = PromptOptimizer()
        result = optimizer.optimize(user_prompt)
        
        print(f"\n📋 OPTIMIZATION COMPLETE:")
        print(f"   Original: {result['original_prompt']}")
        print(f"   Optimized: {result['optimized_prompt']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Time: {result['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 