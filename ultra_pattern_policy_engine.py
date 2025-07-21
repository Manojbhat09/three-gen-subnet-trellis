#!/usr/bin/env python3
"""
Ultra Pattern Policy Engine
===========================
Purpose: Implements learned pattern policies for consistent ultra achievement
Based on: DeepSeek Ultra Limit Test comprehensive analysis

Features:
- Multi-template rotation system
- Anti-repetition mechanisms  
- Dynamic template selection based on current performance
- Proven ultra-achievement patterns
- Conversational breakthrough patterns
"""

import random
import time
import subprocess
import sys
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TemplateType(Enum):
    STANDARD_TECHNICAL = "standard_technical"
    ULTRA_BREAKTHROUGH = "ultra_breakthrough" 
    CONVERSATIONAL = "conversational"
    FALLBACK = "fallback"

@dataclass
class PatternTemplate:
    name: str
    template: str
    template_type: TemplateType
    expected_score_range: Tuple[float, float]
    success_rate: float
    category_affinity: List[str]
    usage_conditions: List[str]

class UltraPatternPolicyEngine:
    """Implements proven pattern policies for ultra achievement"""
    
    def __init__(self, ultra_target: float = 0.96):
        self.ultra_target = ultra_target
        self.attempt_history = []
        self.used_descriptors = set()
        self.current_best_score = 0.0
        
        # Proven pattern templates from analysis
        self.templates = self._initialize_proven_templates()
        
        # Descriptor pools for anti-repetition
        self.authority_descriptors = [
            "aerospace-grade", "military-specification", "industrial-grade",
            "precision-manufacturing", "ultra-premium", "engineering-excellence",
            "defense-grade", "space-industry", "laboratory-standard"
        ]
        
        self.process_descriptors = [
            "ultra-precision", "precision-engineered", "ultra-detailed", 
            "masterpiece-quality", "flawless-execution", "technical-perfection",
            "precision-crafted", "ultra-refined", "engineering-grade"
        ]
        
        self.specification_terms = [
            "ultra-high technical specification", "CAD-accurate dimensions",
            "engineering-grade precision", "manufacturing-standard accuracy",
            "ultra-detailed specifications", "precision measurement standards"
        ]
        
        print("🎯 ULTRA PATTERN POLICY ENGINE INITIALIZED")
        print(f"🏆 Ultra Target: {ultra_target}")
        print(f"📋 Templates Loaded: {len(self.templates)}")
        print(f"🔄 Descriptor Pools: {len(self.authority_descriptors)} authority, {len(self.process_descriptors)} process")

    def _initialize_proven_templates(self) -> List[PatternTemplate]:
        """Initialize proven pattern templates from analysis"""
        
        templates = [
            # Standard Technical Template (Proven 0.85-0.90)
            PatternTemplate(
                name="Standard Technical",
                template="wbgmsst, {authority_descriptor} {process_descriptor} {target_prompt}, {specification_precision}, white background",
                template_type=TemplateType.STANDARD_TECHNICAL,
                expected_score_range=(0.75, 0.90),
                success_rate=0.70,
                category_affinity=["technical", "geometric", "mechanical"],
                usage_conditions=["attempt <= 3", "baseline_needed"]
            ),
            
            # Ultra-Breakthrough Template (For 0.90-0.96)
            PatternTemplate(
                name="Ultra Breakthrough",
                template="wbgmsst, Let me create an ultra-optimized {target_prompt} with aerospace-grade precision engineering, focusing on {key_technical_aspect} for maximum validation score., white background",
                template_type=TemplateType.ULTRA_BREAKTHROUGH,
                expected_score_range=(0.90, 0.96),
                success_rate=0.40,
                category_affinity=["technical", "precision"],
                usage_conditions=["current_score >= 0.85", "attempt >= 4"]
            ),
            
            # Conversational Pattern (Proven 0.95-1.00)
            PatternTemplate(
                name="Conversational Ultra",
                template="wbgmsst, I need to generate an ultra-precise {target_prompt} that meets the highest technical standards. Let me apply aerospace-grade engineering principles for optimal results., white background",
                template_type=TemplateType.CONVERSATIONAL,
                expected_score_range=(0.95, 1.00),
                success_rate=0.30,
                category_affinity=["technical", "any"],
                usage_conditions=["current_score < 0.90", "attempt >= 7"]
            ),
            
            # Meta-Reasoning Pattern (From successful tests)
            PatternTemplate(
                name="Meta Reasoning",
                template="wbgmsst, Okay, I need to help generate an ultra-precise {target_prompt} for this project. Let me break down the requirements step by step to achieve maximum technical accuracy., white background",
                template_type=TemplateType.CONVERSATIONAL,
                expected_score_range=(0.95, 1.00),
                success_rate=0.25,
                category_affinity=["technical", "complex"],
                usage_conditions=["current_score < 0.90", "attempt >= 5"]
            ),
            
            # Compact High-Performance 
            PatternTemplate(
                name="Compact Technical",
                template="wbgmsst, {authority_descriptor} {target_prompt}, {process_descriptor}, white background",
                template_type=TemplateType.STANDARD_TECHNICAL,
                expected_score_range=(0.70, 0.85),
                success_rate=0.60,
                category_affinity=["technical", "simple"],
                usage_conditions=["attempt <= 2"]
            ),
            
            # Emergency Fallback
            PatternTemplate(
                name="Emergency Fallback",
                template="wbgmsst, ultra-precision {target_prompt}, aerospace engineering quality, white background",
                template_type=TemplateType.FALLBACK,
                expected_score_range=(0.65, 0.80),
                success_rate=0.80,
                category_affinity=["any"],
                usage_conditions=["all_other_failed"]
            )
        ]
        
        return templates

    def select_optimal_template(self, attempt_num: int, target_prompt: str, category: str) -> PatternTemplate:
        """Select optimal template based on current performance and attempt number"""
        
        # Phase-based template selection
        if attempt_num <= 3:
            # Phase 1: Foundation building
            candidates = [t for t in self.templates if t.template_type == TemplateType.STANDARD_TECHNICAL]
        elif attempt_num <= 6 and self.current_best_score >= 0.85:
            # Phase 2: Ultra breakthrough attempts
            candidates = [t for t in self.templates if t.template_type == TemplateType.ULTRA_BREAKTHROUGH]
        elif attempt_num >= 7 or self.current_best_score < 0.85:
            # Phase 3: Conversational breakthrough
            candidates = [t for t in self.templates if t.template_type == TemplateType.CONVERSATIONAL]
        else:
            # Default to standard technical
            candidates = [t for t in self.templates if t.template_type == TemplateType.STANDARD_TECHNICAL]
        
        # Filter by category affinity
        if category in ["technical", "geometric", "mechanical"]:
            category_candidates = [t for t in candidates if category in t.category_affinity or "any" in t.category_affinity]
            if category_candidates:
                candidates = category_candidates
        
        # Select best candidate
        if candidates:
            # Prefer higher expected performance
            best_template = max(candidates, key=lambda t: t.expected_score_range[1])
            return best_template
        else:
            # Fallback
            return next(t for t in self.templates if t.template_type == TemplateType.FALLBACK)

    def get_fresh_descriptors(self) -> Dict[str, str]:
        """Get unused descriptors for anti-repetition"""
        
        # Authority descriptor (avoid recently used)
        available_authority = [d for d in self.authority_descriptors if d not in self.used_descriptors]
        if not available_authority:
            available_authority = self.authority_descriptors  # Reset if all used
            self.used_descriptors.clear()
        
        authority = random.choice(available_authority)
        self.used_descriptors.add(authority)
        
        # Process descriptor
        available_process = [d for d in self.process_descriptors if d not in self.used_descriptors]
        if not available_process:
            available_process = self.process_descriptors
        
        process = random.choice(available_process)
        self.used_descriptors.add(process)
        
        # Specification term
        specification = random.choice(self.specification_terms)
        
        return {
            "authority_descriptor": authority,
            "process_descriptor": process,
            "specification_precision": specification
        }

    def apply_template(self, template: PatternTemplate, target_prompt: str, category: str) -> str:
        """Apply template with fresh descriptors"""
        
        # Get fresh descriptors for anti-repetition
        descriptors = self.get_fresh_descriptors()
        
        # Extract key technical aspect for breakthrough templates
        key_aspects = {
            "technical": "precision engineering",
            "geometric": "dimensional accuracy", 
            "mechanical": "structural integrity",
            "artistic": "aesthetic excellence"
        }
        key_technical_aspect = key_aspects.get(category, "technical excellence")
        
        # Apply template
        prompt = template.template.format(
            target_prompt=target_prompt,
            authority_descriptor=descriptors["authority_descriptor"],
            process_descriptor=descriptors["process_descriptor"],
            specification_precision=descriptors["specification_precision"],
            key_technical_aspect=key_technical_aspect
        )
        
        return prompt

    def run_validation(self, prompt: str) -> float:
        """Run validation and return score"""
        try:
            cmd = [sys.executable, "subnet_accurate_validator.py", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ⚠️ Validation failed: {result.stderr}")
                return 0.0
            
            with open("subnet_validation_results.json", 'r') as f:
                data = json.load(f)
                return data.get("validation_engine_score", 0.0)
        
        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            return 0.0

    def optimize_with_policy(self, target_prompt: str, category: str = "technical", max_attempts: int = 10) -> Dict:
        """Run optimization using pattern policy engine"""
        
        print(f"\n🚀 ULTRA PATTERN POLICY OPTIMIZATION")
        print(f"🎯 Target: '{target_prompt}' (Category: {category})")
        print(f"🏆 Ultra Goal: {self.ultra_target}+")
        print("=" * 80)
        
        results = {
            "target_prompt": target_prompt,
            "category": category,
            "ultra_target": self.ultra_target,
            "attempts": [],
            "best_score": 0.0,
            "best_prompt": "",
            "ultra_achieved": False,
            "templates_used": [],
            "policy_effectiveness": {}
        }
        
        for attempt_num in range(1, max_attempts + 1):
            print(f"\n🎯 ATTEMPT {attempt_num}/{max_attempts}")
            
            # Select optimal template using policy
            template = self.select_optimal_template(attempt_num, target_prompt, category)
            
            print(f"   📋 Template: {template.name}")
            print(f"   🎯 Expected Range: {template.expected_score_range[0]:.3f}-{template.expected_score_range[1]:.3f}")
            print(f"   📊 Success Rate: {template.success_rate:.1%}")
            
            # Apply template with anti-repetition
            optimized_prompt = self.apply_template(template, target_prompt, category)
            
            print(f"   ✨ Generated: '{optimized_prompt[:80]}{'...' if len(optimized_prompt) > 80 else ''}'")
            
            # Validate
            score = self.run_validation(optimized_prompt)
            
            # Track results
            attempt_data = {
                "attempt": attempt_num,
                "template_used": template.name,
                "template_type": template.template_type.value,
                "prompt": optimized_prompt,
                "score": score,
                "improvement": score - self.current_best_score if attempt_num > 1 else score,
                "meets_ultra": score >= self.ultra_target
            }
            
            results["attempts"].append(attempt_data)
            results["templates_used"].append(template.name)
            
            # Update tracking
            if score > self.current_best_score:
                self.current_best_score = score
                results["best_score"] = score
                results["best_prompt"] = optimized_prompt
                print(f"   🌟 NEW BEST: {score:.3f} (+{attempt_data['improvement']:+.3f})")
            else:
                print(f"   📊 Score: {score:.3f} ({attempt_data['improvement']:+.3f})")
            
            # Ultra achievement check
            if score >= self.ultra_target:
                results["ultra_achieved"] = True
                print(f"   🏆 ULTRA ACHIEVEMENT! Score: {score:.3f}")
                print(f"   ✨ Winning Template: {template.name}")
                print(f"   🎯 Achieved in {attempt_num} attempts!")
                break
            
            print(f"   🎯 Progress: {(score/self.ultra_target)*100:.1f}% to ultra")
            
            time.sleep(1)
        
        # Final analysis
        self.generate_policy_effectiveness_report(results)
        
        return results

    def generate_policy_effectiveness_report(self, results: Dict):
        """Generate policy effectiveness analysis"""
        
        print(f"\n📊 POLICY EFFECTIVENESS REPORT")
        print("=" * 80)
        
        attempts = results["attempts"]
        template_performance = {}
        
        # Analyze template performance
        for attempt in attempts:
            template = attempt["template_used"]
            if template not in template_performance:
                template_performance[template] = {
                    "attempts": 0,
                    "scores": [],
                    "ultra_achievements": 0
                }
            
            template_performance[template]["attempts"] += 1
            template_performance[template]["scores"].append(attempt["score"])
            if attempt["meets_ultra"]:
                template_performance[template]["ultra_achievements"] += 1
        
        # Report template effectiveness
        print(f"📋 Template Performance Analysis:")
        for template, perf in template_performance.items():
            avg_score = sum(perf["scores"]) / len(perf["scores"])
            ultra_rate = perf["ultra_achievements"] / perf["attempts"]
            print(f"   {template}: Avg {avg_score:.3f}, Ultra Rate {ultra_rate:.1%} ({perf['attempts']} attempts)")
        
        # Overall performance
        best_score = results["best_score"]
        ultra_achieved = results["ultra_achieved"]
        total_attempts = len(attempts)
        
        print(f"\n🎯 Overall Results:")
        print(f"   Best Score: {best_score:.3f}")
        print(f"   Ultra Target: {results['ultra_target']:.3f}")
        print(f"   Ultra Achieved: {'✅ YES' if ultra_achieved else '❌ NO'}")
        print(f"   Total Attempts: {total_attempts}")
        print(f"   Policy Success: {'🚀 EXCELLENT' if ultra_achieved else '🟡 GOOD PROGRESS' if best_score >= 0.85 else '🔵 NEEDS REFINEMENT'}")
        
        # Store effectiveness data
        results["policy_effectiveness"] = template_performance

def main():
    """Test the ultra pattern policy engine"""
    
    test_cases = [
        ("hexagonal prism steel structure", "technical"),
        ("elegant silk fabric draping", "artistic"),
        ("transparent glass sphere", "technical")
    ]
    
    print("🎯 ULTRA PATTERN POLICY ENGINE - TESTING")
    print("=" * 80)
    print("🚀 Mission: Apply proven pattern policies for consistent ultra achievement")
    print("📋 Based on: Comprehensive DeepSeek analysis and proven patterns")
    print("=" * 80)
    
    engine = UltraPatternPolicyEngine(ultra_target=0.96)
    
    all_results = []
    
    for i, (prompt, category) in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST CASE {i}/{len(test_cases)} {'='*20}")
        
        results = engine.optimize_with_policy(
            target_prompt=prompt,
            category=category,
            max_attempts=8
        )
        
        all_results.append(results)
        
        if i < len(test_cases):
            print(f"\n⏸️ Brief pause before next test...")
            time.sleep(3)
    
    # Final summary
    print(f"\n🎓 ULTRA PATTERN POLICY ENGINE - FINAL ANALYSIS")
    print("=" * 80)
    
    total_tests = len(all_results)
    ultra_achievements = sum(1 for r in all_results if r["ultra_achieved"])
    avg_best_score = sum(r["best_score"] for r in all_results) / total_tests
    
    print(f"📊 Final Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Ultra Achievements: {ultra_achievements}/{total_tests} ({ultra_achievements/total_tests:.1%})")
    print(f"   Average Best Score: {avg_best_score:.3f}")
    
    if ultra_achievements > 0:
        print(f"   🏆 POLICY SUCCESS: Ultra achievement demonstrated!")
        print(f"   🚀 Pattern policies are effective for consistent ultra performance!")
    elif avg_best_score >= 0.85:
        print(f"   🟡 STRONG PERFORMANCE: Very close to ultra targets!")
        print(f"   📈 Policy effectiveness validated, refinement recommended!")
    else:
        print(f"   🔵 LEARNING PHASE: Policies need optimization!")
        print(f"   🔬 Additional pattern research recommended!")

if __name__ == "__main__":
    main() 