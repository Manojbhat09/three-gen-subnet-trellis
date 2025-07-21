#!/usr/bin/env python3
"""
AI Strategic Optimizer Demo - 3-Step Vision
===========================================
Demonstrates your vision:
1. AI analyzes prompt for potential issues 
2. Selects strategic modifications from learned classes
3. Applies ultra-targeting for 0.96+ scores

No failure detection - proactive optimization through understanding.
"""

import requests
import json
import subprocess
import sys
from typing import Dict, List, Tuple

class AIStrategicOptimizerDemo:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        
        # Strategy Classes - discovered effective patterns
        self.strategy_classes = {
            "AUTHORITY_BOOST": ["aerospace-grade", "military-spec", "defense-grade"],
            "PROCESS_EXCELLENCE": ["precision-engineered", "ultra-precision", "ultra-detailed"],
            "TECHNICAL_SPEC": ["ultra-high technical specification", "advanced engineering design"]
        }
        
        # Ultra patterns that consistently hit 0.96+
        self.ultra_patterns = [
            "defense-grade ultra-precision {target} ultra-high technical specification",
            "military-spec ultra-detailed {target} advanced engineering design"
        ]
        
        print("🚀 AI STRATEGIC OPTIMIZER DEMO")
        print("�� Vision: Analyze → Strategize → Ultra-Target")
        print("=" * 60)

    def step1_ai_analysis(self, prompt: str) -> Dict:
        """Step 1: AI analyzes prompt proactively"""
        
        print(f"🔍 STEP 1: AI ANALYSIS of '{prompt}'")
        
        analysis_prompt = f"""Analyze this 3D generation prompt for optimization:

PROMPT: "{prompt}"

Identify:
1. Issues that might cause low scores (generic terms, missing quality indicators)
2. Opportunities to add premium descriptors 
3. Best strategy to optimize this type of object

Respond with:
ISSUES: [problems you see]
OPPORTUNITIES: [where to improve] 
STRATEGY: [best approach]
CONFIDENCE: [0.0-1.0]

Analysis:"""

        try:
            response = requests.post(f"{self.ollama_url}/api/chat", 
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 200}
                }, timeout=30)
            
            ai_analysis = response.json()["message"]["content"]
            print(f"   🤖 AI Analysis: {ai_analysis[:200]}...")
            
            return {"analysis": ai_analysis, "confidence": 0.8}
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return {"analysis": "Generic prompt needs premium descriptors", "confidence": 0.6}

    def step2_strategic_selection(self, prompt: str, analysis: Dict) -> List[str]:
        """Step 2: Select strategic modifications"""
        
        print(f"⚡ STEP 2: STRATEGIC MODIFICATION SELECTION")
        
        # Based on analysis, select best strategies
        selected_strategies = []
        
        # If analysis mentions authority/quality issues -> AUTHORITY_BOOST
        if any(word in analysis["analysis"].lower() for word in ["generic", "quality", "authority"]):
            selected_strategies.extend(self.strategy_classes["AUTHORITY_BOOST"])
        
        # If analysis mentions precision/process -> PROCESS_EXCELLENCE  
        if any(word in analysis["analysis"].lower() for word in ["precision", "process", "detailed"]):
            selected_strategies.extend(self.strategy_classes["PROCESS_EXCELLENCE"])
        
        # Always add technical specification for high scores
        selected_strategies.extend(self.strategy_classes["TECHNICAL_SPEC"])
        
        # Generate strategic variations
        strategic_prompts = []
        for strategy in selected_strategies[:3]:  # Top 3 strategies
            strategic_prompt = f"wbgmsst, {strategy} {prompt}, white background"
            strategic_prompts.append(strategic_prompt)
        
        print(f"   📋 Selected {len(strategic_prompts)} strategic modifications")
        for i, sp in enumerate(strategic_prompts, 1):
            print(f"      {i}. {sp}")
        
        return strategic_prompts

    def step3_ultra_targeting(self, strategic_prompts: List[str], original_prompt: str) -> Dict:
        """Step 3: Ultra-targeting for 0.96+ scores"""
        
        print(f"🏆 STEP 3: ULTRA-TARGETING OPTIMIZATION")
        
        best_prompt = ""
        best_score = 0.0
        
        # Test strategic prompts
        for i, prompt in enumerate(strategic_prompts, 1):
            print(f"   🔧 Testing strategic prompt {i}...")
            score = self.run_validation(prompt)
            print(f"      📊 Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_prompt = prompt
            
            if score >= 0.96:
                print(f"      🎉 ULTRA ACHIEVED!")
                return {"prompt": prompt, "score": score, "ultra": True}
        
        # If not ultra, try proven ultra patterns
        if best_score < 0.96:
            print(f"   🚀 Applying proven ultra patterns...")
            for pattern in self.ultra_patterns:
                ultra_prompt = f"wbgmsst, {pattern.format(target=original_prompt)}, white background"
                score = self.run_validation(ultra_prompt)
                print(f"      🔬 Ultra pattern: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_prompt = ultra_prompt
                
                if score >= 0.96:
                    print(f"      🏆 ULTRA ACHIEVED WITH PATTERN!")
                    return {"prompt": ultra_prompt, "score": score, "ultra": True}
        
        return {"prompt": best_prompt, "score": best_score, "ultra": False}

    def run_validation(self, prompt: str) -> float:
        """Run validation and return score"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
        except:
            return 0.0

    def optimize_with_vision(self, prompt: str) -> Dict:
        """Main optimization implementing the 3-step vision"""
        
        print(f"\n🎯 OPTIMIZING: '{prompt}'")
        print("=" * 80)
        
        # Step 1: AI Analysis
        analysis = self.step1_ai_analysis(prompt)
        
        # Step 2: Strategic Selection
        strategic_prompts = self.step2_strategic_selection(prompt, analysis)
        
        # Step 3: Ultra-Targeting
        result = self.step3_ultra_targeting(strategic_prompts, prompt)
        
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"   🎯 Final Score: {result['score']:.3f}")
        print(f"   🏆 Ultra Achieved: {'YES' if result['ultra'] else 'NO'}")
        print(f"   📝 Best Prompt: {result['prompt']}")
        
        return {
            "original": prompt,
            "optimized": result['prompt'],
            "score": result['score'],
            "ultra": result['ultra']
        }

def main():
    optimizer = AIStrategicOptimizerDemo()
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping"
    ]
    
    results = []
    for prompt in test_prompts:
        result = optimizer.optimize_with_vision(prompt)
        results.append(result)
    
    print(f"\n🎓 DEMO COMPLETE")
    print("=" * 60)
    
    ultra_count = sum(1 for r in results if r['ultra'])
    print(f"📊 {len(results)} tests, {ultra_count} ultra achieved")
    
    for r in results:
        status = "🏆" if r['ultra'] else "📊"
        print(f"{status} {r['score']:.3f}: {r['original']}")

if __name__ == "__main__":
    main()
