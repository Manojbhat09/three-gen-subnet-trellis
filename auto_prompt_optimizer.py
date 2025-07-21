#!/usr/bin/env python3
"""
Automated Prompt Optimization Pipeline for Subnet 17
Purpose: Integrate prompt optimization into the mining workflow
"""
import json
import time
import subprocess
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from risk_analyzer import PromptRiskAnalyzer
from advanced_prompt_optimizer import AdvancedPromptOptimizer

@dataclass
class MiningOptimizationResult:
    """Result of mining-ready prompt optimization"""
    original_prompt: str
    final_prompt: str
    optimization_applied: bool
    predicted_score: float
    risk_level: str
    validation_score: Optional[float] = None  # Actual score after generation
    generation_success: bool = False
    time_taken: float = 0.0

class AutoPromptOptimizer:
    """Automated prompt optimization for mining pipeline"""
    
    def __init__(self, config: Dict = None):
        self.risk_analyzer = PromptRiskAnalyzer()
        self.optimizer = AdvancedPromptOptimizer()
        
        # Configuration
        self.config = config or {
            "risk_threshold": 0.6,  # Optimize if predicted score < this
            "enable_validation": True,  # Test optimized prompts locally
            "validation_timeout": 180,  # Max time for local validation
            "optimization_strategies": ["comprehensive", "aggressive"],
            "min_confidence": 0.5,  # Minimum confidence to apply optimization
            "preserve_intent": True,  # Try to preserve semantic intent
        }
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "optimizations_applied": 0,
            "optimizations_successful": 0,
            "average_improvement": 0.0,
            "processing_time_total": 0.0
        }
    
    def should_optimize(self, prompt: str) -> Tuple[bool, str]:
        """Determine if a prompt should be optimized"""
        assessment = self.risk_analyzer.analyze_prompt(prompt)
        
        # Check if predicted score is below threshold
        max_predicted_score = assessment.predicted_score_range[1]
        
        if max_predicted_score < self.config["risk_threshold"]:
            return True, f"Predicted score {max_predicted_score:.2f} below threshold {self.config['risk_threshold']}"
        
        # Check risk level
        if assessment.risk_level in ["HIGH", "CRITICAL"]:
            return True, f"Risk level: {assessment.risk_level}"
        
        return False, "Prompt appears safe"
    
    def optimize_for_mining(self, prompt: str) -> MiningOptimizationResult:
        """Optimize a prompt for mining with full validation"""
        start_time = time.time()
        
        self.stats["total_processed"] += 1
        
        # Initial risk assessment
        should_opt, reason = self.should_optimize(prompt)
        
        if not should_opt:
            # Prompt is already good, no optimization needed
            assessment = self.risk_analyzer.analyze_prompt(prompt)
            predicted_score = sum(assessment.predicted_score_range) / 2
            
            result = MiningOptimizationResult(
                original_prompt=prompt,
                final_prompt=prompt,
                optimization_applied=False,
                predicted_score=predicted_score,
                risk_level=assessment.risk_level,
                time_taken=time.time() - start_time
            )
            
            print(f"✅ Prompt OK: '{prompt}' (predicted: {predicted_score:.2f})")
            return result
        
        print(f"🔧 Optimizing: '{prompt}' ({reason})")
        
        # Try different optimization strategies
        best_result = None
        best_score = -1
        
        for strategy in self.config["optimization_strategies"]:
            optimization_result = self.optimizer.optimize_prompt(prompt, strategy)
            
            if optimization_result.confidence >= self.config["min_confidence"]:
                # Assess optimized prompt
                opt_assessment = self.risk_analyzer.analyze_prompt(optimization_result.optimized_prompt)
                opt_predicted_score = sum(opt_assessment.predicted_score_range) / 2
                
                if opt_predicted_score > best_score:
                    best_score = opt_predicted_score
                    best_result = optimization_result
        
        if best_result is None:
            # No good optimization found, use original
            assessment = self.risk_analyzer.analyze_prompt(prompt)
            predicted_score = sum(assessment.predicted_score_range) / 2
            
            result = MiningOptimizationResult(
                original_prompt=prompt,
                final_prompt=prompt,
                optimization_applied=False,
                predicted_score=predicted_score,
                risk_level=assessment.risk_level,
                time_taken=time.time() - start_time
            )
            
            print(f"⚠️ No good optimization found, using original")
            return result
        
        # Use best optimization
        self.stats["optimizations_applied"] += 1
        
        optimized_prompt = best_result.optimized_prompt
        opt_assessment = self.risk_analyzer.analyze_prompt(optimized_prompt)
        predicted_score = sum(opt_assessment.predicted_score_range) / 2
        
        result = MiningOptimizationResult(
            original_prompt=prompt,
            final_prompt=optimized_prompt,
            optimization_applied=True,
            predicted_score=predicted_score,
            risk_level=opt_assessment.risk_level,
            time_taken=time.time() - start_time
        )
        
        print(f"✨ Optimized: '{optimized_prompt}' (predicted: {predicted_score:.2f})")
        
        # Optional: Validate the optimization locally
        if self.config["enable_validation"]:
            actual_score = self.validate_prompt_locally(optimized_prompt)
            if actual_score is not None:
                result.validation_score = actual_score
                result.generation_success = True
                
                # Update statistics
                if actual_score > 0.6:
                    self.stats["optimizations_successful"] += 1
                
                print(f"📊 Validated: {actual_score:.3f}")
        
        self.stats["processing_time_total"] += result.time_taken
        return result
    
    def validate_prompt_locally(self, prompt: str) -> Optional[float]:
        """Validate an optimized prompt using local validation"""
        try:
            # Use our working simple_local_validator.py
            result = subprocess.run([
                'python3', 'simple_local_validator.py', prompt
            ], capture_output=True, text=True, timeout=self.config["validation_timeout"])
            
            if result.returncode == 0:
                # Parse score from output
                lines = result.stdout.strip().split('\n')
                score_lines = [line for line in lines if 'Final Score:' in line]
                if score_lines:
                    score = float(score_lines[0].split('Final Score:')[1].strip())
                    return score
            
            return None
            
        except Exception as e:
            print(f"⚠️ Validation error: {e}")
            return None
    
    def batch_optimize_for_mining(self, prompts: List[str]) -> List[MiningOptimizationResult]:
        """Optimize multiple prompts for mining"""
        results = []
        
        print(f"🚀 BATCH PROMPT OPTIMIZATION")
        print(f"Processing {len(prompts)} prompts...")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            result = self.optimize_for_mining(prompt)
            results.append(result)
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.5)
        
        # Generate summary
        self.print_batch_summary(results)
        return results
    
    def print_batch_summary(self, results: List[MiningOptimizationResult]):
        """Print summary of batch optimization"""
        print(f"\n📊 BATCH OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        total = len(results)
        optimized = len([r for r in results if r.optimization_applied])
        validated = len([r for r in results if r.validation_score is not None])
        successful = len([r for r in results if r.validation_score and r.validation_score > 0.6])
        
        print(f"Total Prompts: {total}")
        print(f"Optimizations Applied: {optimized} ({optimized/total*100:.1f}%)")
        print(f"Locally Validated: {validated} ({validated/total*100:.1f}%)")
        if validated > 0:
            success_pct = successful/validated*100
        else:
            success_pct = 0
        print(f"Validation Successful: {successful}/{validated} ({success_pct:.1f}%)")
        
        # Score improvements
        improvements = []
        for result in results:
            if result.optimization_applied and result.validation_score:
                # Estimate original score for comparison
                orig_assessment = self.risk_analyzer.analyze_prompt(result.original_prompt)
                orig_predicted = sum(orig_assessment.predicted_score_range) / 2
                improvement = result.validation_score - orig_predicted
                improvements.append(improvement)
        
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Average Improvement: +{avg_improvement:.3f}")
        
        # Risk level distribution
        risk_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for result in results:
            risk_levels[result.risk_level] += 1
        
        print(f"\nFinal Risk Distribution:")
        for level, count in risk_levels.items():
            print(f"   {level}: {count} ({count/total*100:.1f}%)")
        
        # Show best optimizations
        best_optimizations = [r for r in results if r.optimization_applied and r.validation_score and r.validation_score > 0.7]
        if best_optimizations:
            print(f"\n✨ BEST OPTIMIZATIONS:")
            for result in sorted(best_optimizations, key=lambda x: x.validation_score, reverse=True)[:5]:
                print(f"   Score: {result.validation_score:.3f} | '{result.final_prompt}'")
    
    def get_stats(self) -> Dict:
        """Get optimization statistics"""
        if self.stats["optimizations_applied"] > 0:
            success_rate = self.stats["optimizations_successful"] / self.stats["optimizations_applied"]
        else:
            success_rate = 0.0
        
        if self.stats["total_processed"] > 0:
            avg_time = self.stats["processing_time_total"] / self.stats["total_processed"]
        else:
            avg_time = 0.0
        
        return {
            **self.stats,
            "optimization_success_rate": success_rate,
            "average_processing_time": avg_time
        }

def test_auto_optimizer():
    """Test the automated optimizer with real problematic prompts"""
    
    # Test prompts including known problematic ones
    test_prompts = [
        # Known problematic
        "glass jug filled juice",
        "silver chalice with leafy vine pattern", 
        "transparent invisible object floating",
        "quantum mechanical probability cloud",
        "thing with parts and stuff",
        "abstract conceptual entity",
        
        # Mixed risk
        "crystal vase with flowers",
        "wooden chair carved details",
        "blue ceramic bowl",
        "metal robot toy",
        
        # Should be safe
        "red sports car",
        "wooden table",
        "plastic bottle",
        "stone statue"
    ]
    
    # Initialize optimizer with validation enabled
    optimizer = AutoPromptOptimizer({
        "risk_threshold": 0.6,
        "enable_validation": True,
        "validation_timeout": 120,
        "optimization_strategies": ["comprehensive", "aggressive"],
        "min_confidence": 0.5
    })
    
    # Run batch optimization
    results = optimizer.batch_optimize_for_mining(test_prompts)
    
    # Save results
    output_data = {
        "test_timestamp": time.time(),
        "config": optimizer.config,
        "results": [
            {
                "original_prompt": r.original_prompt,
                "final_prompt": r.final_prompt,
                "optimization_applied": r.optimization_applied,
                "predicted_score": r.predicted_score,
                "risk_level": r.risk_level,
                "validation_score": r.validation_score,
                "generation_success": r.generation_success,
                "time_taken": r.time_taken
            }
            for r in results
        ],
        "statistics": optimizer.get_stats()
    }
    
    with open("auto_optimization_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Complete results saved to: auto_optimization_results.json")
    
    # Show final statistics
    stats = optimizer.get_stats()
    print(f"\n📈 FINAL STATISTICS:")
    print(f"   Total Processed: {stats['total_processed']}")
    print(f"   Optimizations Applied: {stats['optimizations_applied']}")
    print(f"   Success Rate: {stats['optimization_success_rate']*100:.1f}%")
    print(f"   Average Processing Time: {stats['average_processing_time']:.2f}s")

if __name__ == "__main__":
    test_auto_optimizer() 