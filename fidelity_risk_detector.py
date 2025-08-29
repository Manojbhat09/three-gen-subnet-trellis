#!/usr/bin/env python3
"""
Fidelity Risk Detector - Analyzes prompts to predict likelihood of 0.0000 fidelity score.
Uses patterns from 171 high fidelity prompts vs 147 zero fidelity prompts.
"""

import json
import re
import statistics
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import math


class FidelityRiskDetector:
    """Analyzes prompts to predict fidelity failure risk."""

    def __init__(self):
        self.high_fidelity_patterns = {}
        self.zero_fidelity_patterns = {}
        self.atmospheric_keywords = {
            'breathtaking', 'mesmerizing', 'romantic', 'moonlit', 'ethereal', 'delicate',
            'tender', 'whispering', 'dancing', 'flowing', 'shimmering', 'glowing',
            'mystical', 'enchanted', 'serene', 'tranquil', 'natural', 'beautiful',
            'elegant', 'graceful', 'poetic', 'atmospheric', 'dreamlike', 'soft',
            'gentle', 'subtle', 'delicate', 'tender', 'velvety', 'silken'
        }

        self.technical_keywords = {
            'detailed', 'photorealistic', '3d', 'model', 'texture', 'polygon',
            'realistic', 'precise', 'accurate', 'sharp', 'defined', 'polished',
            'sturdy', 'durable', 'robust', 'sleek', 'streamlined', 'ergonomic',
            'compact', 'efficient', 'optimized', 'professional', 'industrial'
        }

        self.load_training_data()

    def load_training_data(self):
        """Load and analyze training data from both datasets."""
        try:
            # Load high fidelity prompts
            with open('high_fidelity_prompts_0.75.json', 'r') as f:
                high_fidelity_data = json.load(f)

            # Load zero fidelity prompts
            with open('zero_fidelity_prompts.txt', 'r') as f:
                content = f.read()
                # Extract prompts from the list format
                zero_fidelity_prompts = []
                for line in content.split('\n'):
                    if line.strip().startswith('"') and line.strip().endswith('",'):
                        prompt = line.strip()[1:-2]  # Remove quotes and comma
                        zero_fidelity_prompts.append(prompt)

            print(f"Loaded {len(high_fidelity_data)} high fidelity prompts")
            print(f"Loaded {len(zero_fidelity_prompts)} zero fidelity prompts")

            # Analyze patterns
            self.analyze_patterns(high_fidelity_data, zero_fidelity_prompts)

        except FileNotFoundError as e:
            print(f"Training data not found: {e}")
            print("Please ensure both 'high_fidelity_prompts_0.75.json' and 'zero_fidelity_prompts.txt' exist")

    def analyze_patterns(self, high_fidelity_data: List[Dict], zero_fidelity_prompts: List[str]):
        """Analyze patterns between successful and failed prompts."""

        # Analyze high fidelity patterns
        high_prompts = [item['original_prompt'] for item in high_fidelity_data]
        self.high_fidelity_patterns = self.analyze_prompt_set(high_prompts, "High Fidelity")

        # Analyze zero fidelity patterns
        self.zero_fidelity_patterns = self.analyze_prompt_set(zero_fidelity_prompts, "Zero Fidelity")

        # Calculate risk factors
        self.calculate_risk_factors()

    def analyze_prompt_set(self, prompts: List[str], label: str) -> Dict:
        """Analyze a set of prompts and return statistical patterns."""
        patterns = {
            'word_counts': [],
            'unique_words': set(),
            'avg_word_length': [],
            'atmospheric_word_count': [],
            'technical_word_count': [],
            'complexity_score': [],
            'sentence_count': []
        }

        for prompt in prompts:
            words = prompt.split()
            patterns['word_counts'].append(len(words))

            # Unique words
            patterns['unique_words'].update(words)

            # Average word length
            if words:
                avg_length = sum(len(word) for word in words) / len(words)
                patterns['avg_word_length'].append(avg_length)

            # Atmospheric and technical word counts
            atmospheric_count = sum(1 for word in words if word.lower() in self.atmospheric_keywords)
            technical_count = sum(1 for word in words if word.lower() in self.technical_keywords)
            patterns['atmospheric_word_count'].append(atmospheric_count)
            patterns['technical_word_count'].append(technical_count)

            # Complexity score (unique words / total words)
            if words:
                complexity = len(set(words)) / len(words)
                patterns['complexity_score'].append(complexity)

            # Sentence count (rough approximation)
            sentences = len(re.split(r'[.!?]+', prompt))
            patterns['sentence_count'].append(sentences)

        # Calculate statistics
        stats = {}
        for key, values in patterns.items():
            if key == 'unique_words':
                stats[key] = len(values)
            elif values:
                stats[f'{key}_mean'] = statistics.mean(values)
                stats[f'{key}_median'] = statistics.median(values)
                stats[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0
                stats[f'{key}_min'] = min(values)
                stats[f'{key}_max'] = max(values)

        print(f"\n📊 {label} Prompt Analysis:")
        print(f"   Total prompts: {len(prompts)}")
        print(f"   Word count - Mean: {stats.get('word_counts_mean', 0):.1f}, Median: {stats.get('word_counts_median', 0):.1f}")
        print(f"   Atmospheric words - Mean: {stats.get('atmospheric_word_count_mean', 0):.2f}")
        print(f"   Technical words - Mean: {stats.get('technical_word_count_mean', 0):.2f}")
        print(f"   Complexity score - Mean: {stats.get('complexity_score_mean', 0):.3f}")

        return stats

    def calculate_risk_factors(self):
        """Calculate risk factors based on pattern differences."""
        self.risk_factors = {}

        # Word count risk
        high_word_mean = self.high_fidelity_patterns.get('word_counts_mean', 5)
        zero_word_mean = self.zero_fidelity_patterns.get('word_counts_mean', 10)

        # Higher word count increases risk
        if zero_word_mean > high_word_mean:
            self.risk_factors['word_count_multiplier'] = zero_word_mean / high_word_mean
        else:
            self.risk_factors['word_count_multiplier'] = 1.0

        # Atmospheric word risk
        high_atmos_mean = self.high_fidelity_patterns.get('atmospheric_word_count_mean', 0)
        zero_atmos_mean = self.zero_fidelity_patterns.get('atmospheric_word_count_mean', 0)

        if zero_atmos_mean > high_atmos_mean:
            self.risk_factors['atmospheric_risk'] = zero_atmos_mean / max(high_atmos_mean, 0.1)
        else:
            self.risk_factors['atmospheric_risk'] = 1.0

        # Technical word protection
        high_tech_mean = self.high_fidelity_patterns.get('technical_word_count_mean', 0)
        zero_tech_mean = self.zero_fidelity_patterns.get('technical_word_count_mean', 0)

        if high_tech_mean > zero_tech_mean:
            self.risk_factors['technical_protection'] = high_tech_mean / max(zero_tech_mean, 0.1)
        else:
            self.risk_factors['technical_protection'] = 1.0

        # Complexity risk
        high_complex = self.high_fidelity_patterns.get('complexity_score_mean', 0.8)
        zero_complex = self.zero_fidelity_patterns.get('complexity_score_mean', 0.9)

        if zero_complex > high_complex:
            self.risk_factors['complexity_risk'] = zero_complex / high_complex
        else:
            self.risk_factors['complexity_risk'] = 1.0

        print("\n🎯 Risk Factors Calculated:")
        print(f"   Word count risk: {self.risk_factors['word_count_multiplier']:.2f}")
        print(f"   Atmospheric risk: {self.risk_factors['atmospheric_risk']:.2f}")
        print(f"   Technical protection: {self.risk_factors['technical_protection']:.2f}")
        print(f"   Complexity risk: {self.risk_factors['complexity_risk']:.2f}")
    def predict_risk(self, prompt: str) -> Dict:
        """Predict the risk of a prompt getting 0.0000 fidelity score."""
        words = prompt.split()
        word_count = len(words)

        # Count atmospheric and technical words
        atmospheric_count = sum(1 for word in words if word.lower() in self.atmospheric_keywords)
        technical_count = sum(1 for word in words if word.lower() in self.technical_keywords)

        # Calculate complexity
        if words:
            complexity = len(set(words)) / len(words)
            avg_word_length = sum(len(word) for word in words) / len(words)
        else:
            complexity = 0
            avg_word_length = 0

        # Calculate risk scores
        risk_scores = {}

        # Word count risk
        expected_high_words = self.high_fidelity_patterns.get('word_counts_mean', 5)
        word_count_risk = min(word_count / expected_high_words, 3.0)  # Cap at 3x
        risk_scores['word_count'] = word_count_risk

        # Atmospheric risk
        if atmospheric_count > 0:
            atmos_ratio = atmospheric_count / len(words)
            expected_high_atmos = self.high_fidelity_patterns.get('atmospheric_word_count_mean', 0) / self.high_fidelity_patterns.get('word_counts_mean', 5)
            atmospheric_risk = atmos_ratio / max(expected_high_atmos, 0.01)
            risk_scores['atmospheric'] = min(atmospheric_risk, 5.0)  # Cap at 5x
        else:
            risk_scores['atmospheric'] = 0.1  # Low risk if no atmospheric words

        # Technical protection
        if technical_count > 0:
            tech_ratio = technical_count / len(words)
            risk_scores['technical'] = 1 / (tech_ratio + 0.1)  # Lower risk with more technical words
        else:
            risk_scores['technical'] = 2.0  # Higher risk without technical words

        # Complexity risk
        expected_high_complex = self.high_fidelity_patterns.get('complexity_score_mean', 0.8)
        complexity_risk = complexity / expected_high_complex
        risk_scores['complexity'] = complexity_risk

        # Overall risk score (weighted average)
        weights = {
            'word_count': 0.25,
            'atmospheric': 0.35,
            'technical': 0.25,
            'complexity': 0.15
        }

        overall_risk = sum(risk_scores[category] * weight for category, weight in weights.items())

        # Risk level classification
        if overall_risk < 1.2:
            risk_level = "LOW"
            confidence = "High"
        elif overall_risk < 2.0:
            risk_level = "MEDIUM"
            confidence = "Medium"
        elif overall_risk < 3.0:
            risk_level = "HIGH"
            confidence = "Medium"
        else:
            risk_level = "VERY HIGH"
            confidence = "High"

        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'confidence': confidence,
            'detailed_scores': risk_scores,
            'word_count': word_count,
            'atmospheric_words': atmospheric_count,
            'technical_words': technical_count,
            'complexity_score': complexity,
            'recommendations': self.generate_recommendations(risk_scores, prompt)
        }

    def generate_recommendations(self, risk_scores: Dict, prompt: str) -> List[str]:
        """Generate specific recommendations to reduce risk."""
        recommendations = []

        if risk_scores['word_count'] > 1.5:
            recommendations.append("Consider shortening the prompt - successful prompts average 5 words")

        if risk_scores['atmospheric'] > 2.0:
            recommendations.append("Reduce atmospheric/poetic language - use more direct, technical descriptions")

        if risk_scores['technical'] > 1.5:
            recommendations.append("Add technical keywords like 'detailed', 'realistic', 'precise'")

        if risk_scores['complexity'] > 1.3:
            recommendations.append("Simplify language - use more common, straightforward words")

        if not recommendations:
            recommendations.append("Prompt looks good - low risk of failure")

        return recommendations


def main():
    print("🔍 Fidelity Risk Detector")
    print("=" * 50)

    # Initialize detector
    detector = FidelityRiskDetector()

    # Test with some example prompts
    test_prompts = [
        "big square yellow computer mouse",  # Likely LOW risk
        "a vibrant hummingbird with iridescent green and black feathers, perched delicately on a rose petal",  # HIGH risk
        "ornate dragon scale shoulder guards with sharp spines",  # LOW risk
        "breathtaking photorealistic hummingbird with mesmerizing plumage in a romantic moonlit scene",  # VERY HIGH risk
        "sleek black leather sofa",  # MEDIUM risk
        "red crystal pendant",  # LOW risk
    ]

    print("\n🧪 Testing Risk Predictions:")
    print("-" * 50)

    for prompt in test_prompts:
        result = detector.predict_risk(prompt)
        print(f"\n📝 Prompt: '{prompt}'")
        print(f"   Overall Risk: {result['overall_risk_score']:.2f}")
        print(f"   Risk Level: {result['risk_level']} ({result['confidence']} confidence)")
        print(f"   Word Count: {result['word_count']}")
        print(f"   Atmospheric Words: {result['atmospheric_words']}")
        print(f"   Technical Words: {result['technical_words']}")
        print(f"   Recommendations:")
        for rec in result['recommendations']:
            print(f"     • {rec}")


if __name__ == "__main__":
    main()
