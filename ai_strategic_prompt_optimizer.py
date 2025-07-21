#!/usr/bin/env python3
"""
AI Strategic Prompt Optimizer - 3-Step Vision Implementation
===========================================================
1. Analyze prompt for potential generation issues proactively
2. Select strategic modifications from learned classes  
3. Apply ultra-targeting optimizations for 0.96+ scores
4. Store successful logic/policies for reuse on similar prompts

No failure detection needed - optimization through understanding.
"""

import requests
import json
import time
import subprocess
import sys
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import statistics
import re

@dataclass
class PromptAnalysis:
    original_prompt: str
    identified_issues: List[str]
    optimization_opportunities: List[str]
    prompt_category: str
    recommended_strategies: List[str]
    confidence: float

@dataclass
class StrategyApplication:
    strategy_name: str
    modifications_made: List[str]
    resulting_prompt: str
    validation_score: float
    ultra_potential: bool

class AIStrategicOptimizer:
    """AI-driven strategic prompt optimizer"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "deepseek-r1:1.5b"
        self.ultra_target = ultra_target
        
        # Strategy Classes - learned effective patterns
        self.strategy_classes = {
            "AUTHORITY_BOOST": {
                "descriptors": ["aerospace-grade", "military-spec", "defense-grade", "aviation-standard"],
                "purpose": "Add authoritative quality indicators",
                "ultra_potential": True
            },
            "PROCESS_EXCELLENCE": {
                "descriptors": ["precision-engineered", "ultra-precision", "ultra-detailed", "masterpiece-quality"],
                "purpose": "Emphasize manufacturing excellence",
                "ultra_potential": True
            },
            "TECHNICAL_SPECIFICATION": {
                "descriptors": ["ultra-high technical specification", "advanced engineering design", "premium manufacturing excellence"],
                "purpose": "Add technical depth",
                "ultra_potential": True
            }
        }
        
        # Ultra patterns discovered from previous testing
        self.proven_ultra_patterns = [
            "defense-grade ultra-precision {target} ultra-high technical specification",
            "military-spec ultra-detailed {target} advanced engineering design", 
            "aerospace-grade precision-engineered {target} ultra-high technical specification"
        ]
        
        print(f"🚀 AI STRATEGIC PROMPT OPTIMIZER")
        print(f"🎯 Vision: Proactive optimization through AI understanding")
        print(f"⚡ Strategy: Analyze → Strategize → Ultra-Target → Learn")
        print("=" * 80)

    def step1_analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Step 1: AI analyzes prompt for potential issues and opportunities"""
        
        print(f"🔍 STEP 1: AI PROMPT ANALYSIS")
        
        analysis_request = f"""ANALYZE PROMPT FOR 3D GENERATION OPTIMIZATION

TARGET: "{prompt}"

Analyze this prompt and identify:

1. POTENTIAL ISSUES that might cause low scores:
   - Generic descriptors lacking premium quality
   - Missing technical specifications
   - Weak authority/process language
   
2. OPTIMIZATION OPPORTUNITIES:
   - Where to add authority (aerospace-grade, military-spec)
   - Where to add process excellence (precision-engineered, ultra-detailed)
   - Technical specification improvements

3. STRATEGIC RECOMMENDATIONS:
   - Best strategy classes for this object type
   - Expected score improvement potential

RESPOND FORMAT:
ISSUES: [list issues]
OPPORTUNITIES: [list opportunities] 
CATEGORY: [technical/organic/artistic]
STRATEGIES: [recommended strategy classes]
CONFIDENCE: [0.0-1.0]

ANALYSIS:"""

        try:
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": analysis_request}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 300}
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=45)
            ai_response = response.json()["message"]["content"]
            
            # Parse AI response
            analysis = self._parse_analysis(ai_response, prompt)
            
            print(f"   🚨 Issues: {', '.join(analysis.identified_issues)}")
            print(f"   🎯 Opportunities: {', '.join(analysis.optimization_opportunities)}")
            print(f"   📊 Confidence: {analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            # Fallback analysis
            return PromptAnalysis(
                original_prompt=prompt,
                identified_issues=["Generic descriptors"],
                optimization_opportunities=["Add authority", "Add technical precision"],
                prompt_category="technical",
                recommended_strategies=["AUTHORITY_BOOST", "PROCESS_EXCELLENCE"],
                confidence=0.6
            )

    def _parse_analysis(self, ai_response: str, prompt: str) -> PromptAnalysis:
        """Parse AI analysis response"""
        
        # Extract key sections
        issues = re.search(r'ISSUES:\s*(.+?)(?=OPPORTUNITIES:|$)', ai_response, re.DOTALL)
        opportunities = re.search(r'OPPORTUNITIES:\s*(.+?)(?=CATEGORY:|$)', ai_response, re.DOTALL)
        category = re.search(r'CATEGORY:\s*(.+?)(?=STRATEGIES:|$)', ai_response, re.DOTALL)
        strategies = re.search(r'STRATEGIES:\s*(.+?)(?=CONFIDENCE:|$)', ai_response, re.DOTALL)
        confidence = re.search(r'CONFIDENCE:\s*(.+?)$', ai_response, re.DOTALL)
        
        return PromptAnalysis(
            original_prompt=prompt,
            identified_issues=self._parse_list(issues.group(1) if issues else ""),
            optimization_opportunities=self._parse_list(opportunities.group(1) if opportunities else ""),
            prompt_category=category.group(1).strip() if category else "technical",
            recommended_strategies=self._parse_list(strategies.group(1) if strategies else ""),
            confidence=float(confidence.group(1).strip()) if confidence else 0.7
        )

    def _parse_list(self, text: str) -> List[str]:
        """Parse list from text"""
        items = re.split(r'[,\n\-•]', text.strip())
        return [item.strip() for item in items if item.strip()]

    def step2_strategic_modifications(self, analysis: PromptAnalysis) -> List[StrategyApplication]:
        """Step 2: Select and apply strategic modifications"""
        
        print(f"⚡ STEP 2: STRATEGIC MODIFICATIONS")
        
        applications = []
        
        # Select strategies based on AI recommendations
        selected_strategies = []
        for strategy in analysis.recommended_strategies:
            if strategy in self.strategy_classes:
                selected_strategies.append(strategy)
        
        # Default to high-effectiveness strategies if none recommended
        if not selected_strategies:
            selected_strategies = ["AUTHORITY_BOOST", "PROCESS_EXCELLENCE"]
        
        print(f"   📋 Selected strategies: {', '.join(selected_strategies)}")
        
        # Apply each strategy
        for strategy_name in selected_strategies:
            strategy_info = self.strategy_classes[strategy_name]
            
            # Try best descriptor from this strategy class
            descriptor = strategy_info["descriptors"][0]
            modified_prompt = self._apply_descriptor(analysis.original_prompt, descriptor)
            
            application = StrategyApplication(
                strategy_name=strategy_name,
                modifications_made=[descriptor],
                resulting_prompt=modified_prompt,
                validation_score=0.0,
                ultra_potential=strategy_info["ultra_potential"]
            )
            applications.append(application)
        
        print(f"   ✅ Generated {len(applications)} strategic applications")
        return applications

    def _apply_descriptor(self, prompt: str, descriptor: str) -> str:
        """Apply descriptor to create optimized prompt"""
        
        # Try ultra patterns first for known high-potential descriptors
        if descriptor in ["defense-grade", "military-spec", "aerospace-grade"]:
            import random
            pattern = random.choice(self.proven_ultra_patterns)
            result = f"wbgmsst, {pattern.format(target=prompt)}, white background"
            if len(result) <= 150:
                return result
        
        # Standard application
        return f"wbgmsst, {descriptor} {prompt}, ultra-high technical specification, white background"

    def step3_ultra_targeting(self, applications: List[StrategyApplication]) -> StrategyApplication:
        """Step 3: Ultra-targeting optimization for 0.96+ scores"""
        
        print(f"🏆 STEP 3: ULTRA-TARGETING OPTIMIZATION")
        
        best_app = None
        best_score = 0.0
        
        # Test each application
        for i, app in enumerate(applications, 1):
            print(f"   🔧 Testing {i}/{len(applications)}: {app.strategy_name}")
            print(f"      📝 {app.resulting_prompt}")
            
            score, _ = self.run_validation(app.resulting_prompt)
            app.validation_score = score
            
            print(f"      📊 Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_app = app
            
            if score >= self.ultra_target:
                print(f"      🎉 ULTRA ACHIEVED!")
                return app
        
        # If no ultra achieved, try ultra pattern boost
        if best_score < self.ultra_target:
            print(f"   🚀 Applying ultra pattern boost...")
            ultra_app = self._try_ultra_patterns(applications[0].resulting_prompt)
            if ultra_app and ultra_app.validation_score > best_score:
                best_app = ultra_app
        
        print(f"   🏆 Best score achieved: {best_app.validation_score:.3f}")
        return best_app

    def _try_ultra_patterns(self, base_prompt: str) -> Optional[StrategyApplication]:
        """Try proven ultra patterns"""
        
        # Extract target from base prompt
        target = base_prompt.split(',')[1].split(',')[0].strip()
        if target.startswith('wbgmsst'):
            return None
        
        best_app = None
        best_score = 0.0
        
        for pattern in self.proven_ultra_patterns:
            ultra_prompt = f"wbgmsst, {pattern.format(target=target)}, white background"
            score, _ = self.run_validation(ultra_prompt)
            
            print(f"      🔬 Ultra pattern: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_app = StrategyApplication(
                    strategy_name="ULTRA_PATTERN",
                    modifications_made=[pattern],
                    resulting_prompt=ultra_prompt,
                    validation_score=score,
                    ultra_potential=True
                )
        
        return best_app

    def run_validation(self, prompt: str) -> Tuple[float, float]:
        """Run validation using accurate validator"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return 0.0, 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0), data.get("demo_fidelity_score", 0.0)
        
        except Exception:
            return 0.0, 0.0

    def optimize_strategically(self, prompt: str) -> Dict:
        """Main strategic optimization implementing 3-step vision"""
        
        print(f"\n🎯 STRATEGIC OPTIMIZATION: '{prompt}'")
        print("=" * 80)
        
        # Step 1: AI Analysis
        analysis = self.step1_analyze_prompt(prompt)
        
        # Step 2: Strategic Modifications  
        applications = self.step2_strategic_modifications(analysis)
        
        # Step 3: Ultra-Targeting
        best_application = self.step3_ultra_targeting(applications)
        
        # Results
        ultra_achieved = best_application.validation_score >= self.ultra_target
        
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"   🎯 Final Score: {best_application.validation_score:.3f}")
        print(f"   🏆 Ultra Achieved: {'YES' if ultra_achieved else 'NO'}")
        print(f"   📝 Best Prompt: {best_application.resulting_prompt}")
        
        return {
            "original_prompt": prompt,
            "final_score": best_application.validation_score,
            "ultra_achieved": ultra_achieved,
            "optimized_prompt": best_application.resulting_prompt,
            "strategy_used": best_application.strategy_name
        }

def main():
    """Test the strategic optimizer"""
    
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping", 
        "transparent glass sphere with reflections",
        "ornate wooden sculpture"
    ]
    
    optimizer = AIStrategicOptimizer(ultra_target=0.96)
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_prompts)} {'='*20}")
        result = optimizer.optimize_strategically(prompt)
        results.append(result)
        
        if i < len(test_prompts):
            time.sleep(2)
    
    # Summary
    print(f"\n🎓 STRATEGIC OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    ultra_count = sum(1 for r in results if r['ultra_achieved'])
    avg_score = statistics.mean([r['final_score'] for r in results])
    
    print(f"📊 Results: {len(results)} tests, {ultra_count} ultra, avg {avg_score:.3f}")
    
    for result in sorted(results, key=lambda x: x['final_score'], reverse=True):
        status = "🏆" if result['ultra_achieved'] else "📊"
        print(f"   {status} {result['final_score']:.3f}: {result['original_prompt']}")

if __name__ == "__main__":
    main() 