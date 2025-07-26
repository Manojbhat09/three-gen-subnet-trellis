#!/usr/bin/env python3
"""
LLaMA Prompt Optimizer v5.0 - INFERENCE-FOCUSED REDESIGN
========================================================
✅ ONE-SHOT inference for real-world usage
✅ Golden examples from successful prompts
✅ LLM-based pattern extraction and meta-learning
✅ Fast, practical optimization without complex RL training
✅ Contextually appropriate enhancements
✅ Comprehensive logging and analytics

Revolutionary shift: From training-focused to inference-focused architecture
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import statistics
import datetime

@dataclass
class OptimizationResult:
    """Result of prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    optimization_principle: str
    confidence_score: float
    processing_time: float
    timestamp: float

@dataclass
class GoldenExample:
    """High-scoring prompt example for few-shot learning"""
    original: str
    optimized: str
    score: float
    category: str
    principle: str

class PromptOptimizer:
    """One-shot prompt optimizer for inference"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", 
                 golden_examples_file: Optional[str] = None,
                 log_dir: str = "optimizer_logs"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Load golden examples
        self.golden_examples = self._load_golden_examples(golden_examples_file)
        
        # Initialize logging
        self.optimization_log = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._test_connection()
        print("🚀 PROMPT OPTIMIZER V5.0 INITIALIZED")
        print(f"📚 Loaded {len(self.golden_examples)} golden examples")
    
    def _test_connection(self):
        """Test LLaMA connection"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ LLaMA 3.2 Connected")
            else:
                raise Exception(f"Connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def _load_golden_examples(self, examples_file: Optional[str]) -> List[GoldenExample]:
        """Load golden examples from file or use defaults"""
        if examples_file and Path(examples_file).exists():
            with open(examples_file, 'r') as f:
                data = json.load(f)
                return [GoldenExample(**item) for item in data]
        else:
            return self._get_default_golden_examples()
    
    def _get_default_golden_examples(self) -> List[GoldenExample]:
        """Default set of high-scoring golden examples across categories"""
        return [
            GoldenExample(
                original="tall glass of layered lemonade",
                optimized="wbgmsst, crystal-clear artisanal glass vessel filled with vibrant layered lemonade, golden citrus highlights and pristine transparency, professional beverage photography, white background",
                score=0.92,
                category="beverages",
                principle="Enhanced with clarity, color, and professional presentation details"
            ),
            GoldenExample(
                original="sapphire-studded sharp spear",
                optimized="wbgmsst, exquisite sapphire-studded ceremonial spear with mirror-polished obsidian shaft, intricate metalwork and flawless gemstone settings, masterwork craftsmanship, white background",
                score=0.94,
                category="weapons",
                principle="Added precious materials, craftsmanship quality, and refined details"
            ),
            GoldenExample(
                original="cupcake with chocolate icing",
                optimized="wbgmsst, gourmet artisanal cupcake with lustrous dark chocolate ganache, smooth bakery-quality finish and elegant presentation, professional food styling, white background",
                score=0.89,
                category="food",
                principle="Enhanced with culinary quality, texture, and professional presentation"
            ),
            GoldenExample(
                original="emerald pendant",
                optimized="wbgmsst, exquisite handcrafted emerald pendant with flawless gemstone clarity, precious metal setting and luxurious jewelry-grade finish, fine jewelry craftsmanship, white background",
                score=0.93,
                category="jewelry",
                principle="Emphasized luxury materials, clarity, and fine craftsmanship"
            ),
            GoldenExample(
                original="transparent glass sphere",
                optimized="wbgmsst, flawlessly clear crystal sphere with perfect optical clarity, brilliant light refraction and pristine transparency, precision-cut optical glass, white background",
                score=0.91,
                category="glass",
                principle="Focused on optical properties, clarity, and precision manufacturing"
            ),
            GoldenExample(
                original="elegant silk fabric draping",
                optimized="wbgmsst, luxurious silk fabric with elegant flowing drape, lustrous sheen and premium textile quality, artisanal weaving and sophisticated texture, white background",
                score=0.88,
                category="textiles",
                principle="Enhanced with luxury quality, texture, and artisanal craftsmanship"
            ),
            GoldenExample(
                original="small round blue creature",
                optimized="wbgmsst, charming round blue creature with distinctive features, polished animated character design and appealing cartoon aesthetics, professional 3D character art, white background",
                score=0.87,
                category="characters",
                principle="Added character design quality, aesthetic appeal, and professional finish"
            )
        ]
    
    def optimize_prompt(self, user_prompt: str, temperature: float = 0.7) -> OptimizationResult:
        """
        One-shot prompt optimization for inference
        
        Args:
            user_prompt: Original prompt to optimize
            temperature: Creativity level (0.1-1.0)
        
        Returns:
            OptimizationResult with optimized prompt and metadata
        """
        start_time = time.time()
        
        print(f"\n🎯 OPTIMIZING: '{user_prompt}'")
        
        # Build optimization system prompt
        system_prompt = self._build_optimization_system_prompt()
        
        # Build user prompt with context
        user_optimization_prompt = self._build_user_optimization_prompt(user_prompt)
        
        try:
            # Get optimized prompt from LLaMA
            optimized_prompt = self._query_llama_for_optimization(
                system_prompt, user_optimization_prompt, temperature
            )
            
            # Extract optimization principle
            principle = self._extract_optimization_principle(user_prompt, optimized_prompt)
            
            # Calculate confidence (placeholder - could be enhanced)
            confidence = self._calculate_confidence(user_prompt, optimized_prompt)
            
            processing_time = time.time() - start_time
            
            result = OptimizationResult(
                original_prompt=user_prompt,
                optimized_prompt=optimized_prompt,
                optimization_principle=principle,
                confidence_score=confidence,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
            # Log the result
            self._log_optimization_result(result)
            
            print(f"✅ OPTIMIZED: {optimized_prompt}")
            print(f"📈 Principle: {principle}")
            print(f"⏱️  Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            # Return fallback result
            fallback_prompt = self._generate_fallback_optimization(user_prompt)
            processing_time = time.time() - start_time
            
            return OptimizationResult(
                original_prompt=user_prompt,
                optimized_prompt=fallback_prompt,
                optimization_principle="Fallback enhancement with quality and detail improvements",
                confidence_score=0.6,
                processing_time=processing_time,
                timestamp=time.time()
            )
    
    def _build_optimization_system_prompt(self) -> str:
        """Build the golden system prompt for optimization"""
        
        base_prompt = """You are an expert prompt engineer for 3D asset and image generation models. Your task is to take a user's prompt and rewrite it to be more detailed, specific, and visually rich to achieve maximum validation scores.

OPTIMIZATION PRINCIPLES:
1. Analyze the core object and enhance it with contextually appropriate details
2. Add material & texture descriptions (polished, matte, rough, smooth, glistening, etc.)
3. Include quality & craftsmanship details (pristine, handcrafted, precision-engineered, etc.)
4. Enhance with appropriate lighting and context
5. Add evocative adjectives that fit the object type
6. ALWAYS start with "wbgmsst," and end with ", white background"
7. Keep the core object intact - enhance, don't replace

CRITICAL: Your enhancement must be APPROPRIATE for the object type. Don't use technical terms for food/drinks or cutesy terms for industrial objects.

Here are GOLDEN EXAMPLES of excellent, high-scoring optimizations. Apply the same principles of contextual detail and specificity:

"""
        
        # Add golden examples grouped by category
        categories = {}
        for example in self.golden_examples:
            if example.category not in categories:
                categories[example.category] = []
            categories[example.category].append(example)
        
        for category, examples in categories.items():
            base_prompt += f"\n--- {category.upper()} EXAMPLES ---\n"
            for ex in examples[:2]:  # Limit to prevent prompt bloat
                base_prompt += f"Original: \"{ex.original}\"\n"
                base_prompt += f"Optimized: \"{ex.optimized}\" (Score: {ex.score:.2f})\n"
                base_prompt += f"Principle: {ex.principle}\n\n"
        
        base_prompt += """
RESPONSE FORMAT:
ANALYSIS: [Brief analysis of the object and optimization opportunities]
OPTIMIZED_PROMPT: [Your enhanced prompt - ready to use]
REASONING: [Why this enhancement is appropriate and effective]

Remember: Enhance contextually, maintain object integrity, maximize visual richness."""

        return base_prompt
    
    def _build_user_optimization_prompt(self, user_prompt: str) -> str:
        """Build user prompt for optimization"""
        return f"""OPTIMIZE THIS PROMPT: "{user_prompt}"

Requirements:
- Must start with "wbgmsst," and end with ", white background"
- Enhance with contextually appropriate details
- Maintain the core object identity
- Add material, quality, and visual enhancements
- Make it vivid and specific for 3D generation

Provide your optimization following the format above."""
    
    def _query_llama_for_optimization(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Query LLaMA for prompt optimization"""
        
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
                "num_predict": 300
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        response.raise_for_status()
        
        content = response.json()["message"]["content"].strip()
        
        # Extract optimized prompt from response
        return self._extract_optimized_prompt(content)
    
    def _extract_optimized_prompt(self, response: str) -> str:
        """Extract the optimized prompt from LLaMA response"""
        lines = response.split('\n')
        
        # Look for OPTIMIZED_PROMPT line
        for line in lines:
            if line.strip().startswith('OPTIMIZED_PROMPT:'):
                prompt = line.split('OPTIMIZED_PROMPT:', 1)[1].strip()
                return self._clean_prompt(prompt)
        
        # Fallback: find wbgmsst line
        for line in lines:
            if 'wbgmsst' in line.lower():
                return self._clean_prompt(line.strip())
        
        # If nothing found, return the whole response cleaned
        return self._clean_prompt(response)
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and validate prompt format"""
        prompt = prompt.replace('"', '').strip()
        
        if not prompt.startswith('wbgmsst'):
            prompt = f"wbgmsst, {prompt}"
        
        if not prompt.endswith('white background'):
            if prompt.endswith(','):
                prompt += " white background"
            else:
                prompt += ", white background"
        
        return prompt
    
    def _extract_optimization_principle(self, original: str, optimized: str) -> str:
        """Use LLaMA to extract the optimization principle"""
        
        system_prompt = """You are an expert at analyzing prompt optimization techniques. Given an original prompt and its optimized version, extract the general optimization principle that was applied.

Focus on the TECHNIQUE, not the specific words. Describe what kind of enhancement was made in one concise sentence.

Examples:
- "Enhanced with material quality and craftsmanship details"
- "Added sensory details and professional presentation elements"
- "Emphasized luxury qualities and fine details"
- "Included technical precision and quality specifications"
"""

        user_prompt = f"""Original: "{original}"
Optimized: "{optimized}"

What optimization principle was applied? Provide one sentence describing the general technique."""

        try:
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=15)
            response.raise_for_status()
            
            principle = response.json()["message"]["content"].strip()
            return principle[:200]  # Limit length
            
        except Exception as e:
            print(f"   ⚠️ Principle extraction failed: {e}")
            return "Enhanced with quality and detail improvements"
    
    def _calculate_confidence(self, original: str, optimized: str) -> float:
        """Calculate confidence score based on enhancement quality"""
        
        # Simple heuristics for confidence
        confidence = 0.5
        
        # Length increase suggests more detail
        length_ratio = len(optimized) / max(len(original), 1)
        if length_ratio > 2:
            confidence += 0.2
        elif length_ratio > 1.5:
            confidence += 0.1
        
        # Check for quality indicators
        quality_words = ['artisanal', 'precision', 'premium', 'exquisite', 'masterwork', 
                        'crystal-clear', 'flawless', 'luxury', 'professional']
        quality_count = sum(1 for word in quality_words if word.lower() in optimized.lower())
        confidence += min(quality_count * 0.05, 0.2)
        
        # Check proper format
        if optimized.startswith('wbgmsst') and optimized.endswith('white background'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_fallback_optimization(self, user_prompt: str) -> str:
        """Generate fallback optimization if LLaMA fails"""
        prompt_lower = user_prompt.lower()
        
        # Context-aware fallback based on object type
        if any(word in prompt_lower for word in ['glass', 'drink', 'beverage', 'juice', 'wine']):
            return f"wbgmsst, crystal-clear artisanal {user_prompt}, pristine transparency and elegant presentation, white background"
        elif any(word in prompt_lower for word in ['jewelry', 'pendant', 'ring', 'gemstone']):
            return f"wbgmsst, exquisite luxury {user_prompt}, flawless craftsmanship and premium materials, white background"
        elif any(word in prompt_lower for word in ['food', 'cake', 'cupcake', 'bread']):
            return f"wbgmsst, gourmet artisanal {user_prompt}, professional culinary presentation, white background"
        else:
            return f"wbgmsst, premium quality {user_prompt}, masterwork craftsmanship and refined details, white background"
    
    def _log_optimization_result(self, result: OptimizationResult):
        """Log optimization result to CSV"""
        
        log_file = self.log_dir / f"optimization_log_{self.session_id}.csv"
        file_exists = log_file.exists()
        
        with open(log_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "original_prompt", "optimized_prompt", 
                    "optimization_principle", "confidence_score", "processing_time"
                ])
            
            writer.writerow([
                datetime.datetime.fromtimestamp(result.timestamp).isoformat(),
                result.original_prompt,
                result.optimized_prompt,
                result.optimization_principle,
                result.confidence_score,
                result.processing_time
            ])
        
        # Also keep in memory for session analytics
        self.optimization_log.append(result)
    
    def get_session_analytics(self) -> Dict:
        """Get analytics for current session"""
        if not self.optimization_log:
            return {"message": "No optimizations performed yet"}
        
        processing_times = [r.processing_time for r in self.optimization_log]
        confidence_scores = [r.confidence_score for r in self.optimization_log]
        
        return {
            "total_optimizations": len(self.optimization_log),
            "avg_processing_time": statistics.mean(processing_times),
            "avg_confidence": statistics.mean(confidence_scores),
            "session_duration": time.time() - self.optimization_log[0].timestamp,
            "top_principles": self._get_top_principles()
        }
    
    def _get_top_principles(self) -> List[str]:
        """Get most common optimization principles"""
        principles = [r.optimization_principle for r in self.optimization_log]
        # Simple frequency count (could be enhanced)
        unique_principles = list(set(principles))
        return unique_principles[:5]
    
    def save_session_golden_examples(self, min_confidence: float = 0.8):
        """Save high-confidence optimizations as new golden examples"""
        
        high_quality = [r for r in self.optimization_log if r.confidence_score >= min_confidence]
        
        if high_quality:
            golden_file = self.log_dir / f"golden_examples_{self.session_id}.json"
            
            golden_data = []
            for result in high_quality:
                golden_data.append({
                    "original": result.original_prompt,
                    "optimized": result.optimized_prompt,
                    "score": result.confidence_score,
                    "category": "inferred",  # Could be enhanced with categorization
                    "principle": result.optimization_principle
                })
            
            with open(golden_file, 'w') as f:
                json.dump(golden_data, f, indent=2)
            
            print(f"💎 Saved {len(golden_data)} high-quality examples to {golden_file}")
            return golden_file
        
        return None


class BatchOptimizer:
    """Batch optimization for multiple prompts"""
    
    def __init__(self, optimizer: PromptOptimizer):
        self.optimizer = optimizer
    
    def optimize_batch(self, prompts: List[str], temperature: float = 0.7) -> List[OptimizationResult]:
        """Optimize multiple prompts in batch"""
        
        print(f"\n🔄 BATCH OPTIMIZATION: {len(prompts)} prompts")
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]", end=" ")
            result = self.optimizer.optimize_prompt(prompt, temperature)
            results.append(result)
        
        # Save batch results
        self._save_batch_results(results)
        
        return results
    
    def _save_batch_results(self, results: List[OptimizationResult]):
        """Save batch results to file"""
        
        batch_file = self.optimizer.log_dir / f"batch_results_{self.optimizer.session_id}.json"
        
        batch_data = [asdict(result) for result in results]
        
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"\n💾 Batch results saved to {batch_file}")


def main():
    """Demo and testing"""
    print("🚀 LLAMA PROMPT OPTIMIZER V5.0 - INFERENCE-FOCUSED")
    print("="*60)
    print("✅ One-shot optimization for real-world usage")
    print("✅ Golden examples from successful patterns")
    print("✅ LLM-based meta-learning and principle extraction")
    print("✅ Fast, practical optimization without complex training")
    print("="*60)
    
    try:
        # Initialize optimizer
        optimizer = PromptOptimizer()
        
        # Test prompts
        test_prompts = [
            "tall glass of layered lemonade",
            "cylindrical glass of bubbly lemonade", 
            "sapphire-studded sharp spear",
            "emerald pendant",
            "bottle of red wine with cork in it",
            "crystal staff with swirling light",
            "small round blue creature with long nose and pointed ears",
            "cupcake with chocolate icing on top"
        ]
        
        print(f"\n🧪 TESTING WITH {len(test_prompts)} PROMPTS")
        
        # Individual optimizations
        results = []
        for prompt in test_prompts[:3]:  # Test first 3
            result = optimizer.optimize_prompt(prompt)
            results.append(result)
        
        # Batch optimization for remaining
        if len(test_prompts) > 3:
            batch_optimizer = BatchOptimizer(optimizer)
            batch_results = batch_optimizer.optimize_batch(test_prompts[3:])
            results.extend(batch_results)
        
        # Session analytics
        analytics = optimizer.get_session_analytics()
        print(f"\n📊 SESSION ANALYTICS:")
        print(f"   Total optimizations: {analytics['total_optimizations']}")
        print(f"   Average time: {analytics['avg_processing_time']:.2f}s")
        print(f"   Average confidence: {analytics['avg_confidence']:.2f}")
        
        # Save golden examples
        golden_file = optimizer.save_session_golden_examples(min_confidence=0.7)
        if golden_file:
            print(f"   Golden examples saved: {golden_file}")
        
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print(f"📈 Average confidence: {analytics['avg_confidence']:.1%}")
        print(f"⚡ Average speed: {analytics['avg_processing_time']:.2f}s per prompt")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 