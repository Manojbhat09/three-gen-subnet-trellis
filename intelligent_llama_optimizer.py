#!/usr/bin/env python3
"""
Intelligent LLaMA 3.2 Prompt Optimizer
=====================================
Gives LLaMA 3.2 full control over prompt optimization with:
- Intelligent analysis of each prompt's unique requirements
- Learning from successful optimizations 
- Custom optimization strategies per prompt type
- Memory of what works for different categories
- Adaptive improvement based on feedback

This replaces the rigid RL patterns with intelligent, contextual optimization.
"""

import requests
import json
import time
import hashlib
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

@dataclass
class OptimizationMemory:
    """Memory of successful optimization"""
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    score_achieved: float
    prompt_category: str
    key_improvements: List[str]
    timestamp: float

@dataclass
class OptimizationResult:
    """Result of intelligent optimization"""
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    reasoning: str
    confidence: float
    predicted_score: float
    optimization_time: float
    category_detected: str
    key_changes: List[str]
    memory_used: List[str]

class PromptCategorizer:
    """Intelligent prompt categorization"""
    
    @staticmethod
    def categorize_prompt(prompt: str) -> str:
        """Categorize prompt for targeted optimization"""
        prompt_lower = prompt.lower()
        
        # Material-based categories
        if any(word in prompt_lower for word in ['steel', 'metal', 'iron', 'aluminum', 'copper']):
            return 'metal_objects'
        elif any(word in prompt_lower for word in ['fabric', 'cloth', 'silk', 'cotton', 'textile']):
            return 'fabric_materials'
        elif any(word in prompt_lower for word in ['glass', 'crystal', 'transparent', 'clear']):
            return 'transparent_objects'
        elif any(word in prompt_lower for word in ['wood', 'wooden', 'timber', 'oak', 'pine']):
            return 'wooden_objects'
        elif any(word in prompt_lower for word in ['plastic', 'polymer', 'synthetic']):
            return 'synthetic_materials'
        
        # Shape-based categories
        elif any(word in prompt_lower for word in ['sphere', 'ball', 'globe', 'round']):
            return 'spherical_objects'
        elif any(word in prompt_lower for word in ['cube', 'box', 'rectangular', 'square']):
            return 'geometric_shapes'
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pipe', 'rod']):
            return 'cylindrical_objects'
        
        # Function-based categories
        elif any(word in prompt_lower for word in ['tool', 'instrument', 'device', 'equipment']):
            return 'tools_instruments'
        elif any(word in prompt_lower for word in ['furniture', 'chair', 'table', 'desk']):
            return 'furniture'
        elif any(word in prompt_lower for word in ['vehicle', 'car', 'truck', 'machine']):
            return 'vehicles_machines'
        elif any(word in prompt_lower for word in ['art', 'sculpture', 'statue', 'decorative']):
            return 'artistic_objects'
        
        return 'general_objects'

class LearningMemoryDB:
    """Persistent learning memory database"""
    
    def __init__(self, db_path: str = "llama_optimization_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt TEXT NOT NULL,
                optimized_prompt TEXT NOT NULL,
                strategy_used TEXT NOT NULL,
                score_achieved REAL NOT NULL,
                prompt_category TEXT NOT NULL,
                key_improvements TEXT NOT NULL,  -- JSON list
                timestamp REAL NOT NULL,
                prompt_hash TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS category_strategies (
                category TEXT PRIMARY KEY,
                successful_strategies TEXT NOT NULL,  -- JSON list
                avg_score REAL NOT NULL,
                best_score REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_memory(self, memory: OptimizationMemory):
        """Save successful optimization to memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prompt_hash = hashlib.sha256(memory.original_prompt.encode()).hexdigest()[:16]
        
        cursor.execute('''
            INSERT INTO optimization_memories 
            (original_prompt, optimized_prompt, strategy_used, score_achieved, 
             prompt_category, key_improvements, timestamp, prompt_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.original_prompt, memory.optimized_prompt, memory.strategy_used,
            memory.score_achieved, memory.prompt_category, json.dumps(memory.key_improvements),
            memory.timestamp, prompt_hash
        ))
        
        # Update category strategies
        self._update_category_strategies(cursor, memory)
        
        conn.commit()
        conn.close()
    
    def _update_category_strategies(self, cursor, memory: OptimizationMemory):
        """Update category-specific strategy statistics"""
        # Get existing stats
        cursor.execute(
            'SELECT successful_strategies, avg_score, best_score, sample_count FROM category_strategies WHERE category = ?',
            (memory.prompt_category,)
        )
        result = cursor.fetchone()
        
        if result:
            strategies = json.loads(result[0])
            avg_score = result[1]
            best_score = result[2]
            sample_count = result[3]
            
            # Update strategy count
            if memory.strategy_used not in strategies:
                strategies[memory.strategy_used] = 0
            strategies[memory.strategy_used] += 1
            
            # Update scores
            new_avg = (avg_score * sample_count + memory.score_achieved) / (sample_count + 1)
            new_best = max(best_score, memory.score_achieved)
            new_count = sample_count + 1
            
            cursor.execute('''
                UPDATE category_strategies 
                SET successful_strategies=?, avg_score=?, best_score=?, sample_count=?, last_updated=?
                WHERE category=?
            ''', (json.dumps(strategies), new_avg, new_best, new_count, time.time(), memory.prompt_category))
        else:
            # Create new category entry
            strategies = {memory.strategy_used: 1}
            cursor.execute('''
                INSERT INTO category_strategies 
                (category, successful_strategies, avg_score, best_score, sample_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (memory.prompt_category, json.dumps(strategies), memory.score_achieved, 
                  memory.score_achieved, 1, time.time()))
    
    def get_category_knowledge(self, category: str) -> Dict[str, Any]:
        """Get learned knowledge for a specific category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get category strategies
        cursor.execute(
            'SELECT successful_strategies, avg_score, best_score, sample_count FROM category_strategies WHERE category = ?',
            (category,)
        )
        strategy_result = cursor.fetchone()
        
        # Get recent successful examples
        cursor.execute('''
            SELECT original_prompt, optimized_prompt, strategy_used, score_achieved, key_improvements
            FROM optimization_memories 
            WHERE prompt_category = ? AND score_achieved >= 0.8
            ORDER BY timestamp DESC LIMIT 5
        ''', (category,))
        examples = cursor.fetchall()
        
        conn.close()
        
        knowledge = {
            'category': category,
            'has_experience': strategy_result is not None,
            'strategies': json.loads(strategy_result[0]) if strategy_result else {},
            'avg_score': strategy_result[1] if strategy_result else 0.0,
            'best_score': strategy_result[2] if strategy_result else 0.0,
            'sample_count': strategy_result[3] if strategy_result else 0,
            'recent_examples': [
                {
                    'original': ex[0],
                    'optimized': ex[1],
                    'strategy': ex[2],
                    'score': ex[3],
                    'improvements': json.loads(ex[4])
                }
                for ex in examples
            ]
        }
        
        return knowledge

class IntelligentLLaMAOptimizer:
    """Intelligent LLaMA 3.2-powered prompt optimizer with learning"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2:3b"
        self.categorizer = PromptCategorizer()
        self.memory_db = LearningMemoryDB()
        
        # Performance tracking
        self.stats = {
            'total_optimizations': 0,
            'custom_optimizations': 0,
            'memory_guided_optimizations': 0,
            'avg_optimization_time': 0.0,
            'category_coverage': {}
        }
        
        self._test_connection()
        print(f"🧠 INTELLIGENT LLaMA 3.2 OPTIMIZER INITIALIZED")
        print(f"   🔗 Ollama: {ollama_url}")
        print(f"   🗄️ Learning Memory: Enabled")
        print(f"   🎯 Model: {self.model}")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ LLaMA 3.2 Connected")
                return True
        except Exception as e:
            raise ConnectionError(f"❌ LLaMA 3.2 unavailable: {e}")
    
    def _query_llama(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Query LLaMA 3.2 with intelligent prompting"""
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
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(f"{self.ollama_url}/api/chat", json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()["message"]["content"].strip()
        else:
            raise Exception(f"LLaMA query failed: {response.status_code}")
    
    def optimize_prompt(self, prompt: str, feedback_score: Optional[float] = None) -> OptimizationResult:
        """Intelligently optimize prompt using LLaMA 3.2 with learning"""
        
        start_time = time.time()
        self.stats['total_optimizations'] += 1
        
        # Step 1: Categorize the prompt
        category = self.categorizer.categorize_prompt(prompt)
        
        # Step 2: Get learned knowledge for this category
        knowledge = self.memory_db.get_category_knowledge(category)
        
        # Step 3: Build intelligent system prompt
        system_prompt = self._build_intelligent_system_prompt(category, knowledge)
        
        # Step 4: Create context-aware user prompt
        user_prompt = self._build_user_prompt(prompt, category, knowledge)
        
        # Step 5: Get LLaMA's intelligent response
        llama_response = self._query_llama(system_prompt, user_prompt, temperature=0.8)
        
        # Step 6: Parse the response
        optimization_result = self._parse_llama_response(llama_response, prompt, category)
        optimization_result.optimization_time = time.time() - start_time
        
        # Step 7: Update statistics
        self.stats['category_coverage'][category] = self.stats['category_coverage'].get(category, 0) + 1
        if knowledge['has_experience']:
            self.stats['memory_guided_optimizations'] += 1
        else:
            self.stats['custom_optimizations'] += 1
        
        return optimization_result
    
    def _build_intelligent_system_prompt(self, category: str, knowledge: Dict) -> str:
        """Build intelligent system prompt based on category and learned knowledge"""
        
        base_prompt = """You are an expert 3D prompt optimization AI with deep learning capabilities.

MISSION: Transform prompts to achieve 0.9+ validation scores for 3D model generation.

KEY PRINCIPLES:
1. ANALYZE the prompt's unique characteristics and requirements
2. APPLY category-specific optimizations based on material, shape, and function
3. LEARN from previous successful optimizations 
4. CREATE custom solutions, not rigid templates
5. BALANCE technical precision with creative specificity

OPTIMIZATION STRATEGIES:
- Material Enhancement: Add material-specific technical descriptors
- Precision Engineering: Include manufacturing/engineering terms when appropriate  
- Quality Amplification: Boost perceived quality and craftsmanship
- Technical Specification: Add relevant technical details
- Context Optimization: Ensure proper 3D rendering context

RESPONSE FORMAT:
ANALYSIS: [Brief analysis of prompt characteristics and optimization opportunities]
STRATEGY: [Chosen optimization approach and reasoning]
OPTIMIZED: [The optimized prompt - must start with "wbgmsst," and end with ", white background"]
REASONING: [Explanation of key changes and expected impact]
CONFIDENCE: [0.1-1.0 confidence in optimization success]
CHANGES: [List of specific modifications made]"""

        # Add category-specific knowledge
        if knowledge['has_experience']:
            knowledge_section = f"""
LEARNED KNOWLEDGE FOR {category.upper().replace('_', ' ')}:
- Previous optimizations: {knowledge['sample_count']} successful examples
- Best score achieved: {knowledge['best_score']:.3f}
- Average score: {knowledge['avg_score']:.3f}
- Successful strategies: {', '.join(knowledge['strategies'].keys())}

RECENT SUCCESSFUL EXAMPLES:"""
            
            for example in knowledge['recent_examples'][:3]:
                knowledge_section += f"""
- Original: "{example['original']}"
- Optimized: "{example['optimized']}"
- Strategy: {example['strategy']} (Score: {example['score']:.3f})"""
            
            base_prompt += knowledge_section
        else:
            base_prompt += f"""
NEW CATEGORY: {category.upper().replace('_', ' ')}
- No previous experience with this category
- Use general optimization principles
- Focus on material/shape-specific enhancements
- Be creative and experimental"""
        
        return base_prompt
    
    def _build_user_prompt(self, prompt: str, category: str, knowledge: Dict) -> str:
        """Build context-aware user prompt"""
        
        user_prompt = f"""OPTIMIZE THIS PROMPT:

Original: "{prompt}"
Category: {category.replace('_', ' ')}
Current Issues: [Analyze what might cause low scores]

Requirements:
1. Must start with "wbgmsst,"
2. Must end with ", white background"  
3. Keep the core object: extract and preserve the main subject
4. Add strategic enhancements for this specific prompt type
5. Aim for 0.9+ validation score

"""
        
        if knowledge['has_experience']:
            user_prompt += f"Apply learned knowledge from {knowledge['sample_count']} previous {category} optimizations.\n"
        else:
            user_prompt += f"This is a new {category} prompt - be innovative with optimization approach.\n"
        
        user_prompt += "PROVIDE YOUR ANALYSIS AND OPTIMIZED PROMPT:"
        
        return user_prompt
    
    def _parse_llama_response(self, response: str, original_prompt: str, category: str) -> OptimizationResult:
        """Parse LLaMA's intelligent response"""
        
        # Extract sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(('ANALYSIS:', 'STRATEGY:', 'OPTIMIZED:', 'REASONING:', 'CONFIDENCE:', 'CHANGES:')):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.split(':')[0]
                current_content = [':'.join(line.split(':')[1:]).strip()]
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract optimized prompt
        optimized_prompt = sections.get('OPTIMIZED', original_prompt)
        
        # Clean up the optimized prompt
        optimized_prompt = optimized_prompt.replace('"', '').strip()
        if not optimized_prompt.startswith('wbgmsst'):
            optimized_prompt = f"wbgmsst, {optimized_prompt}"
        if not optimized_prompt.endswith('white background'):
            if optimized_prompt.endswith(','):
                optimized_prompt += " white background"
            else:
                optimized_prompt += ", white background"
        
        # Extract confidence
        confidence_str = sections.get('CONFIDENCE', '0.7')
        try:
            confidence = float(confidence_str.split()[0])
        except:
            confidence = 0.7
        
        # Extract changes
        changes_text = sections.get('CHANGES', '')
        changes = [change.strip('- ').strip() for change in changes_text.split('\n') if change.strip()]
        
        return OptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            strategy_used=sections.get('STRATEGY', 'intelligent_optimization'),
            reasoning=sections.get('REASONING', 'LLaMA 3.2 intelligent optimization'),
            confidence=confidence,
            predicted_score=min(0.95, 0.6 + confidence * 0.35),  # Conservative prediction
            optimization_time=0.0,  # Will be set by caller
            category_detected=category,
            key_changes=changes,
            memory_used=[f"Category: {category}"]
        )
    
    def learn_from_feedback(self, result: OptimizationResult, actual_score: float):
        """Learn from validation feedback"""
        
        if actual_score >= 0.8:  # Only learn from successful optimizations
            memory = OptimizationMemory(
                original_prompt=result.original_prompt,
                optimized_prompt=result.optimized_prompt,
                strategy_used=result.strategy_used,
                score_achieved=actual_score,
                prompt_category=result.category_detected,
                key_improvements=result.key_changes,
                timestamp=time.time()
            )
            
            self.memory_db.save_memory(memory)
            print(f"🧠 Learned from success: {result.category_detected} → {actual_score:.3f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total = max(1, self.stats['total_optimizations'])
        return {
            'total_optimizations': self.stats['total_optimizations'],
            'custom_optimization_rate': self.stats['custom_optimizations'] / total,
            'memory_guided_rate': self.stats['memory_guided_optimizations'] / total,
            'categories_encountered': len(self.stats['category_coverage']),
            'category_breakdown': self.stats['category_coverage'],
            'avg_optimization_time': self.stats['avg_optimization_time']
        }

def main():
    """Demo the intelligent LLaMA optimizer"""
    print("🧠 INTELLIGENT LLaMA 3.2 OPTIMIZER DEMO")
    print("="*60)
    
    try:
        optimizer = IntelligentLLaMAOptimizer()
        
        # Test diverse prompts to show intelligent adaptation
        test_prompts = [
            "hexagonal prism steel structure",
            "elegant silk fabric draping", 
            "transparent glass sphere with reflections",
            "wooden sculpture with intricate details",
            "chrome motorcycle engine parts",
            "soft cotton blanket texture",
            "crystalline ice formation",
            "leather bound vintage book"
        ]
        
        print(f"\n🎯 Testing intelligent optimization on {len(test_prompts)} diverse prompts:")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*40} PROMPT {i} {'='*40}")
            
            result = optimizer.optimize_prompt(prompt)
            
            print(f"📝 Original: {result.original_prompt}")
            print(f"🎯 Category: {result.category_detected}")
            print(f"🔧 Strategy: {result.strategy_used}")
            print(f"✨ Optimized: {result.optimized_prompt}")
            print(f"🧠 Reasoning: {result.reasoning}")
            print(f"📊 Confidence: {result.confidence:.3f}")
            print(f"⏱️ Time: {result.optimization_time:.3f}s")
            if result.key_changes:
                print(f"🔄 Changes: {', '.join(result.key_changes[:3])}...")
            
            # Simulate learning from feedback
            simulated_score = 0.85 + (result.confidence * 0.1)
            optimizer.learn_from_feedback(result, simulated_score)
        
        # Show statistics
        stats = optimizer.get_stats()
        print(f"\n📊 INTELLIGENT OPTIMIZATION STATISTICS:")
        print(f"   Total Optimizations: {stats['total_optimizations']}")
        print(f"   Custom Rate: {stats['custom_optimization_rate']:.1%}")
        print(f"   Memory-Guided Rate: {stats['memory_guided_rate']:.1%}")
        print(f"   Categories Discovered: {stats['categories_encountered']}")
        print(f"   Category Breakdown: {stats['category_breakdown']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Ensure Ollama is running with llama3.2:3b model")

if __name__ == "__main__":
    main() 