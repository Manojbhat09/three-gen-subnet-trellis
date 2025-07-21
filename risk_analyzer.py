#!/usr/bin/env python3
"""
Prompt Risk Analyzer for Subnet 17
Purpose: Predict which prompts will score below the 0.6 threshold and identify risk factors
"""
import re
import json
import time
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RiskAssessment:
    """Risk assessment result for a prompt"""
    prompt: str
    predicted_score_range: Tuple[float, float]  # (min, max)
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_factors: List[str]
    recommendations: List[str]
    optimization_priority: int  # 1-10, higher = more urgent

class PromptRiskAnalyzer:
    """Analyzes prompts to predict validation failure risk"""
    
    def __init__(self):
        # Problematic material patterns (high risk of transparency/rendering issues)
        self.problematic_materials = {
            r'\bglass\b': 0.8,
            r'\bcrystal\b': 0.7,
            r'\btransparent\b': 0.9,
            r'\btranslucent\b': 0.8,
            r'\bclear\b': 0.6,
            r'\bliquid\b': 0.7,
            r'\bwater\b': 0.6,
            r'\bjuice\b': 0.5,
            r'\bfluid\b': 0.6,
            r'\bmirror\b': 0.7,
            r'\bdiamond\b': 0.8,
            r'\bemerald\b': 0.7,
            r'\bruby\b': 0.7,
            r'\bsapphire\b': 0.7,
            r'\bgem\b': 0.6,
            r'\bjewel\b': 0.6,
        }
        
        # Complex scene patterns (multiple objects, relationships)
        self.complexity_patterns = {
            r'\bwith\s+\w+\s+(pattern|design|details?)\b': 0.7,
            r'\bholding\b': 0.5,
            r'\bbeside\b': 0.4,
            r'\bnext\s+to\b': 0.4,
            r'\band\s+\w+': 0.3,  # Multiple objects
            r'\bfeaturing\b': 0.5,
            r'\badorned\s+with\b': 0.6,
            r'\bdecorated\s+with\b': 0.6,
        }
        
        # Abstract/vague concept patterns
        self.abstraction_patterns = {
            r'\babstract\b': 0.9,
            r'\bconcept\b': 0.8,
            r'\bessence\b': 0.9,
            r'\bmystical\b': 0.7,
            r'\bspiritual\b': 0.7,
            r'\benergy\b': 0.6,
            r'\bquantum\b': 0.8,
            r'\bineffable\b': 0.9,
            r'\bconceptual\b': 0.8,
            r'\bphilosophical\b': 0.8,
        }
        
        # Grammar/clarity issues
        self.grammar_patterns = {
            r'^[a-z]+\s+[a-z]+\s+filled\s+[a-z]+$': 0.8,  # "glass jug filled juice"
            r'\bthing\s+with\b': 0.9,
            r'\bstuff\b': 0.8,
            r'\bobject\s+floating\b': 0.6,
            r'\bparts\s+and\s+stuff\b': 0.9,
        }
        
        # Positive indicators (reduce risk)
        self.positive_patterns = {
            r'\bwooden?\b': -0.3,
            r'\bmetal\b': -0.2,
            r'\bstone\b': -0.2,
            r'\bceramic\b': -0.2,
            r'\bplastic\b': -0.2,
            r'\bchair\b': -0.3,
            r'\btable\b': -0.3,
            r'\bcar\b': -0.3,
            r'\bhouse\b': -0.3,
            r'\btoy\b': -0.2,
            r'\btool\b': -0.2,
        }
        
        # Load historical data if available
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical validation data to improve predictions"""
        try:
            if Path("zero_score_analysis_results.json").exists():
                with open("zero_score_analysis_results.json", "r") as f:
                    self.historical_data = json.load(f)
            else:
                self.historical_data = None
        except Exception:
            self.historical_data = None
    
    def analyze_prompt(self, prompt: str) -> RiskAssessment:
        """Perform comprehensive risk analysis on a prompt"""
        
        risk_score = 0.0
        risk_factors = []
        recommendations = []
        
        prompt_lower = prompt.lower()
        
        # 1. Material Risk Analysis
        material_risk = 0.0
        for pattern, weight in self.problematic_materials.items():
            if re.search(pattern, prompt_lower):
                material_risk += weight
                material_name = pattern.replace(r'\b', '').replace('\\', '')
                risk_factors.append(f"Problematic material: {material_name}")
                recommendations.append(f"Replace {material_name} with solid materials (wood, metal, ceramic)")
        
        # 2. Complexity Risk Analysis  
        complexity_risk = 0.0
        for pattern, weight in self.complexity_patterns.items():
            if re.search(pattern, prompt_lower):
                complexity_risk += weight
                risk_factors.append("Complex scene with multiple elements/relationships")
                recommendations.append("Simplify to single object focus")
        
        # 3. Abstraction Risk Analysis
        abstraction_risk = 0.0
        for pattern, weight in self.abstraction_patterns.items():
            if re.search(pattern, prompt_lower):
                abstraction_risk += weight
                risk_factors.append("Abstract or conceptual language")
                recommendations.append("Use concrete, physical object descriptions")
        
        # 4. Grammar/Clarity Risk Analysis
        grammar_risk = 0.0
        for pattern, weight in self.grammar_patterns.items():
            if re.search(pattern, prompt_lower):
                grammar_risk += weight
                risk_factors.append("Grammar or clarity issues")
                recommendations.append("Improve grammatical structure and clarity")
        
        # 5. Positive Indicators (reduce total risk)
        positive_adjustment = 0.0
        for pattern, weight in self.positive_patterns.items():
            if re.search(pattern, prompt_lower):
                positive_adjustment += weight
        
        # Calculate total risk score
        total_risk = material_risk + complexity_risk + abstraction_risk + grammar_risk
        adjusted_risk = max(0.0, total_risk + positive_adjustment)
        
        # Predict score range based on risk
        if adjusted_risk <= 0.2:
            predicted_range = (0.7, 0.9)
            risk_level = "LOW"
            priority = 1
        elif adjusted_risk <= 0.5:
            predicted_range = (0.5, 0.8)
            risk_level = "MEDIUM"
            priority = 3
        elif adjusted_risk <= 0.8:
            predicted_range = (0.3, 0.6)
            risk_level = "HIGH"
            priority = 7
        else:
            predicted_range = (0.0, 0.4)
            risk_level = "CRITICAL"
            priority = 10
        
        # Add specific recommendations based on risk level
        if risk_level in ["HIGH", "CRITICAL"]:
            if not recommendations:
                recommendations.append("Major optimization needed - consider complete rewrite")
            recommendations.append("Test locally before subnet submission")
        
        return RiskAssessment(
            prompt=prompt,
            predicted_score_range=predicted_range,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            optimization_priority=priority
        )
    
    def analyze_batch(self, prompts: List[str]) -> List[RiskAssessment]:
        """Analyze multiple prompts"""
        return [self.analyze_prompt(prompt) for prompt in prompts]
    
    def generate_report(self, assessments: List[RiskAssessment]) -> Dict:
        """Generate summary report of risk assessments"""
        
        total_prompts = len(assessments)
        risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        at_risk_prompts = []  # Predicted score < 0.6
        safe_prompts = []     # Predicted score >= 0.6
        
        for assessment in assessments:
            risk_distribution[assessment.risk_level] += 1
            
            if assessment.predicted_score_range[1] < 0.6:  # Even best case < 0.6
                at_risk_prompts.append(assessment)
            elif assessment.predicted_score_range[0] >= 0.6:  # Even worst case >= 0.6
                safe_prompts.append(assessment)
        
        # Calculate percentages
        risk_percentages = {
            level: (count / total_prompts) * 100 
            for level, count in risk_distribution.items()
        }
        
        return {
            "total_prompts": total_prompts,
            "risk_distribution": risk_distribution,
            "risk_percentages": risk_percentages,
            "at_risk_count": len(at_risk_prompts),
            "safe_count": len(safe_prompts),
            "optimization_needed": len(at_risk_prompts),
            "at_risk_prompts": [a.prompt for a in at_risk_prompts],
            "safe_prompts": [a.prompt for a in safe_prompts],
            "average_priority": sum(a.optimization_priority for a in assessments) / total_prompts
        }

def main():
    """Test the risk analyzer with sample prompts"""
    
    analyzer = PromptRiskAnalyzer()
    
    # Test prompts from our previous analysis
    test_prompts = [
        "glass jug filled juice",
        "silver chalice with leafy vine pattern", 
        "transparent invisible object floating",
        "quantum mechanical probability cloud",
        "wooden chair with carved details",
        "a blue ceramic vase",
        "metal lantern with brass finish",
        "plastic toy robot",
        "stone statue of a lion",
        "abstract conceptual entity",
        "thing with parts and stuff",
        "crystal formation with light rays",
        "polished wooden table",
        "red sports car"
    ]
    
    print("🔍 PROMPT RISK ANALYSIS")
    print("=" * 80)
    
    # Analyze each prompt
    assessments = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Analyzing: '{prompt}'")
        assessment = analyzer.analyze_prompt(prompt)
        assessments.append(assessment)
        
        print(f"   📊 Risk Level: {assessment.risk_level}")
        print(f"   📈 Predicted Score: {assessment.predicted_score_range[0]:.2f} - {assessment.predicted_score_range[1]:.2f}")
        print(f"   ⚡ Priority: {assessment.optimization_priority}/10")
        
        if assessment.risk_factors:
            print(f"   ⚠️  Risk Factors:")
            for factor in assessment.risk_factors:
                print(f"      • {factor}")
        
        if assessment.recommendations:
            print(f"   💡 Recommendations:")
            for rec in assessment.recommendations:
                print(f"      • {rec}")
    
    # Generate summary report
    report = analyzer.generate_report(assessments)
    
    print(f"\n📊 SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Prompts Analyzed: {report['total_prompts']}")
    print(f"Risk Distribution:")
    for level, count in report['risk_distribution'].items():
        percentage = report['risk_percentages'][level]
        print(f"   {level}: {count} prompts ({percentage:.1f}%)")
    
    print(f"\nOptimization Needed: {report['optimization_needed']} prompts")
    print(f"Safe Prompts: {report['safe_count']} prompts")
    print(f"Average Priority: {report['average_priority']:.1f}/10")
    
    if report['at_risk_prompts']:
        print(f"\n⚠️  HIGH-RISK PROMPTS (likely to score < 0.6):")
        for prompt in report['at_risk_prompts']:
            print(f"   • {prompt}")
    
    # Save detailed analysis
    output_data = {
        "analysis_timestamp": time.time(),
        "assessments": [
            {
                "prompt": a.prompt,
                "predicted_score_range": a.predicted_score_range,
                "risk_level": a.risk_level,
                "risk_factors": a.risk_factors,
                "recommendations": a.recommendations,
                "optimization_priority": a.optimization_priority
            }
            for a in assessments
        ],
        "summary_report": report
    }
    
    with open("prompt_risk_analysis.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: prompt_risk_analysis.json")

if __name__ == "__main__":
    main() 