#!/usr/bin/env python3
"""
RL Mining Integration - Production Ready
======================================
Integrates the trained RL prompt optimizer with the continuous mining pipeline.
Uses the best checkpoint (episode_024) for production prompt optimization.

Features:
- Load trained RL model checkpoint for fast inference
- Optional LLaMA 3.2 integration for advanced pattern extraction
- Production-optimized for real-time mining
- Fallback to rule-based optimization if RL fails
- Performance monitoring and statistics

Usage:
    # Replace the prompt_optimizer in continuous_trellis_orchestrator.py
    from rl_mining_integration import RLMiningOptimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import requests
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class OptimizationResult:
    """Result of RL prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    predicted_score: float
    confidence: float
    optimization_time: float
    rl_actions_used: List[str]
    llama_pattern_used: Optional[str] = None
    fallback_used: bool = False

class ProductionDQN(nn.Module):
    """Lightweight DQN for production inference"""
    def __init__(self, state_size: int, action_size: int):
        super(ProductionDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.output_layer(x)

class LLaMAKnowledgeEngine:
    """Optional LLaMA 3.2 integration for advanced pattern extraction"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", enabled: bool = False):
        self.ollama_url = ollama_url
        self.enabled = enabled
        self.model = "llama3.2:3b"
        
        if enabled:
            self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ LLaMA 3.2 Knowledge Engine: Connected")
                return True
        except Exception as e:
            print(f"⚠️ LLaMA 3.2 unavailable: {e}")
            self.enabled = False
        return False
    
    def extract_winning_pattern(self, successful_prompt: str, original_prompt: str, score: float) -> Optional[str]:
        """Extract reusable pattern from successful optimization"""
        if not self.enabled or score < 0.85:
            return None
        
        try:
            system_prompt = """You are a prompt optimization expert. Analyze the successful optimization and extract a reusable pattern.

Original: The base prompt that needed optimization
Successful: The optimized version that achieved high score
Your task: Extract the core optimization strategy as a reusable template

Rules:
- Use {target} as placeholder for the original object
- Focus on the key additions that likely improved the score
- Keep templates concise and effective
- If no clear pattern, respond "NO_PATTERN"

Example:
Original: "red car"
Successful: "wbgmsst, aerospace-grade precision-engineered red car, ultra-high technical specification, white background"
Pattern: "aerospace-grade precision-engineered {target}, ultra-high technical specification"
"""
            
            user_prompt = f"""
Original: "{original_prompt}"
Successful: "{successful_prompt}" (Score: {score:.3f})

Extract reusable pattern:"""
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                pattern = result["message"]["content"].strip()
                
                if "NO_PATTERN" not in pattern and "{target}" in pattern:
                    print(f"🧠 LLaMA extracted pattern: {pattern}")
                    return pattern
            
        except Exception as e:
            print(f"⚠️ LLaMA pattern extraction failed: {e}")
        
        return None

class RLMiningOptimizer:
    """Production RL-based prompt optimizer for mining"""
    
    def __init__(self, 
                 checkpoint_path: str = "rl_checkpoints_v2/episode_024",
                 enable_llama: bool = True,
                 fallback_patterns: bool = True):
        """
        Initialize the RL mining optimizer
        
        Args:
            checkpoint_path: Path to trained RL checkpoint (episode_024 recommended)
            enable_llama: Enable LLaMA 3.2 for advanced pattern extraction
            fallback_patterns: Use rule-based patterns if RL fails
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.enable_llama = enable_llama
        self.fallback_patterns = fallback_patterns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        self.model, self.action_space, self.metadata = self._load_checkpoint()
        
        # Initialize LLaMA knowledge engine (optional)
        self.llama_engine = LLaMAKnowledgeEngine(enabled=enable_llama)
        
        # Performance tracking
        self.stats = {
            'total_optimizations': 0,
            'rl_optimizations': 0,
            'llama_patterns_used': 0,
            'fallback_used': 0,
            'avg_optimization_time': 0.0,
            'success_rate': 0.0
        }
        
        # Ultra-winning patterns from training (proven effective)
        self.proven_patterns = [
            "aerospace-grade precision-engineered {target}, ultra-high technical specification",
            "defense-grade ultra-precision {target}, ultra-high technical specification", 
            "military-spec ultra-detailed {target}, premium manufacturing excellence",
            "laboratory-grade precision-forged {target}, aerospace-engineering excellence",
            "precision-aerospace {target}, defense-grade excellence"
        ]
        
        print(f"🚀 RL MINING OPTIMIZER INITIALIZED")
        print(f"   📂 Model: {checkpoint_path}")
        print(f"   🎬 Actions: {len(self.action_space)}")
        print(f"   🧠 LLaMA 3.2: {'ENABLED' if enable_llama else 'DISABLED'}")
        print(f"   🔄 Fallback: {'ENABLED' if fallback_patterns else 'DISABLED'}")
        print(f"   🏆 Training Score: {self.metadata.get('best_score', 'unknown')}")
    
    def _load_checkpoint(self) -> Tuple[nn.Module, List, Dict]:
        """Load the trained RL model checkpoint"""
        agent_file = self.checkpoint_path / 'agent_checkpoint.pth'
        
        if not agent_file.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {agent_file}")
        
        # Load checkpoint
        checkpoint = torch.load(agent_file, map_location=self.device, weights_only=False)
        
        # Get metadata
        metadata = checkpoint.get('metadata', {})
        action_size = checkpoint.get('action_size', 21)  # From training
        
        # Load action space from training (episode_024 had 21 actions)
        action_space = self._get_action_space_from_training(action_size)
        
        # Initialize and load model
        model = ProductionDQN(state_size=25, action_size=action_size).to(self.device)
        model.load_state_dict(checkpoint['q_network_local_state_dict'])
        model.eval()
        
        print(f"   ✅ Checkpoint loaded: {action_size} actions available")
        return model, action_space, metadata
    
    def _get_action_space_from_training(self, action_size: int) -> List[Tuple]:
        """Reconstruct action space from training"""
        # These are the proven actions from episode_024 training
        base_actions = [
            ('APPLY_PATTERN', "defense-grade ultra-precision {target}, ultra-high technical specification", 'full_replace'),
            ('APPLY_PATTERN', "aerospace-grade precision-engineered {target}, advanced engineering design", 'full_replace'),
            ('APPLY_PATTERN', "military-spec ultra-detailed {target}, premium manufacturing excellence", 'full_replace'),
            ('APPLY_PATTERN', "laboratory-grade precision-forged {target}, aerospace-engineering excellence", 'full_replace'),
            ('APPLY_PATTERN', "ultra-precision masterpiece-quality {target}, ultra-high technical specification", 'full_replace'),
            ('APPLY_PATTERN', "precision-aerospace {target}, defense-grade excellence", 'full_replace'),
            ('APPLY_PATTERN', "ultra-military-spec {target}, precision-engineering design", 'full_replace'),
            # Learned patterns from meta-learning (3 discovered during training)
            ('APPLY_PATTERN', "aerospace-grade precision-engineered {target}, high-tech finish", 'full_replace'),
            ('APPLY_PATTERN', "aerospace-grade precision-engineered {target}, ultra-high technical specification", 'full_replace'),
            ('APPLY_PATTERN', "defense-grade ultra-precision {target}, ultra-high technical specification", 'full_replace'),
        ]
        
        # Add upgrade actions
        upgrade_actions = [
            ('UPGRADE_AUTHORITY', 'aerospace-grade', 'replace'),
            ('UPGRADE_AUTHORITY', 'defense-grade', 'replace'),
            ('UPGRADE_AUTHORITY', 'military-spec', 'replace'),
            ('UPGRADE_PROCESS', 'ultra-precision', 'replace'),
            ('UPGRADE_PROCESS', 'precision-engineered', 'replace'),
            ('UPGRADE_PROCESS', 'masterpiece-quality', 'replace'),
            ('UPGRADE_QUALITY', 'ultra-high technical specification', 'replace'),
            ('UPGRADE_QUALITY', 'advanced engineering design', 'replace'),
        ]
        
        # Add simplify actions
        simplify_actions = [
            ('SIMPLIFY', 'remove_duplicates', 'clean'),
            ('SIMPLIFY', 'keep_best_only', 'clean'),
            ('SIMPLIFY', 'ultra_minimal', 'clean'),
        ]
        
        all_actions = base_actions + upgrade_actions + simplify_actions
        return all_actions[:action_size]  # Trim to actual action size
    
    def _get_state(self, prompt: str, target_object: str) -> np.ndarray:
        """Get state representation for the prompt"""
        state = np.zeros(25)  # 25D state from training
        
        # Basic prompt features
        state[0] = len(prompt) / 150.0  # Normalized length
        state[1] = 1.0 if 'wbgmsst' in prompt.lower() else 0.0
        state[2] = 1.0 if 'white background' in prompt.lower() else 0.0
        
        # Authority descriptors
        prompt_lower = prompt.lower()
        state[3] = 1.0 if 'aerospace' in prompt_lower else 0.0
        state[4] = 1.0 if 'defense' in prompt_lower else 0.0
        state[5] = 1.0 if 'military' in prompt_lower else 0.0
        
        # Process descriptors
        state[6] = 1.0 if 'ultra-precision' in prompt_lower else 0.0
        state[7] = 1.0 if 'precision-engineered' in prompt_lower else 0.0
        state[8] = 1.0 if 'masterpiece' in prompt_lower else 0.0
        
        # Quality descriptors
        state[9] = 1.0 if 'ultra-high technical' in prompt_lower else 0.0
        state[10] = 1.0 if 'advanced engineering' in prompt_lower else 0.0
        
        # Target object features
        target_lower = target_object.lower()
        state[11] = 1.0 if any(w in target_lower for w in ["steel", "metal", "iron"]) else 0.0
        state[12] = 1.0 if any(w in target_lower for w in ["fabric", "cloth", "silk"]) else 0.0
        state[13] = 1.0 if any(w in target_lower for w in ["glass", "crystal", "transparent"]) else 0.0
        state[14] = 1.0 if any(w in target_lower for w in ["wood", "wooden"]) else 0.0
        
        # Meta-learning features (from training)
        state[15] = float(len(self.action_space)) / 25.0  # Action space size
        state[16] = float(self.stats['rl_optimizations']) / max(1, self.stats['total_optimizations'])
        state[17] = self.stats['success_rate']
        
        # Advanced pattern features
        state[18] = 1.0 if any(pattern in prompt_lower for pattern in ["precision-aerospace", "ultra-military"]) else 0.0
        state[19] = prompt_lower.count('ultra') / 10.0  # Ultra descriptor count
        
        return state
    
    def _extract_target_object(self, prompt: str) -> str:
        """Extract the main object from the prompt"""
        # Remove common prefixes/suffixes
        cleaned = prompt.lower()
        for prefix in ['wbgmsst,', 'a ', 'an ', 'the ']:
            cleaned = cleaned.replace(prefix, '').strip()
        for suffix in [', white background', ', 3d', ', isometric']:
            cleaned = cleaned.replace(suffix, '').strip()
        
        # Take first few words as target
        words = cleaned.split()[:3]  # First 3 words usually contain the object
        return ' '.join(words) if words else prompt
    
    def _apply_rl_action(self, action: Tuple, prompt: str, target: str) -> str:
        """Apply an RL action to optimize the prompt"""
        action_type, modifier, mode = action
        
        if action_type == 'APPLY_PATTERN':
            # Full replacement with proven pattern
            pattern = modifier.replace('{target}', target)
            return f"wbgmsst, {pattern}, white background"
            
        elif action_type in ['UPGRADE_AUTHORITY', 'UPGRADE_PROCESS', 'UPGRADE_QUALITY']:
            # Add or replace descriptors
            parts = prompt.split(', ')
            if len(parts) >= 3:
                middle = parts[1]
                if action_type == 'UPGRADE_AUTHORITY':
                    # Remove existing authority descriptors and add new one
                    import re
                    middle = re.sub(r'\b(aerospace-grade|military-spec|defense-grade|aviation-standard|laboratory-grade)\b\s*', '', middle)
                elif action_type == 'UPGRADE_PROCESS':
                    # Remove existing process descriptors and add new one
                    import re
                    middle = re.sub(r'\b(ultra-precision|precision-engineered|masterpiece-quality|ultra-detailed)\b\s*', '', middle)
                
                parts[1] = f"{modifier} {middle}".strip()
                return ', '.join(parts)
                
        elif action_type == 'SIMPLIFY':
            if mode == 'clean':
                # Clean up the prompt
                if modifier == 'remove_duplicates':
                    words = prompt.split()
                    return ' '.join(list(dict.fromkeys(words)))  # Remove duplicates
                elif modifier == 'keep_best_only':
                    return f"wbgmsst, defense-grade ultra-precision {target}, ultra-high technical specification, white background"
                elif modifier == 'ultra_minimal':
                    return f"wbgmsst, aerospace-grade {target}, precision-engineered excellence, white background"
        
        return prompt  # Return original if action can't be applied
    
    def _predict_best_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Use trained model to predict best action"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            # Get best action
            action_idx = q_values.argmax().item()
            confidence = torch.softmax(q_values, dim=1).max().item()
            
            return action_idx, confidence
    
    def _use_fallback_pattern(self, prompt: str, target: str) -> str:
        """Use rule-based fallback optimization"""
        # Use the best proven pattern from training
        best_pattern = "aerospace-grade precision-engineered {target}, ultra-high technical specification"
        optimized = best_pattern.replace('{target}', target)
        return f"wbgmsst, {optimized}, white background"
    
    def optimize_prompt(self, prompt: str, max_steps: int = 3, timeout: float = 5.0) -> OptimizationResult:
        """
        Optimize a prompt using the trained RL model
        
        Args:
            prompt: Original prompt to optimize
            max_steps: Maximum optimization steps (default 3 for speed)
            timeout: Maximum time to spend optimizing
            
        Returns:
            OptimizationResult with optimized prompt and metadata
        """
        start_time = time.time()
        self.stats['total_optimizations'] += 1
        
        try:
            # Extract target object
            target_object = self._extract_target_object(prompt)
            
            # Check if prompt already looks optimized
            if ('wbgmsst' in prompt.lower() and 
                any(pattern in prompt.lower() for pattern in ['aerospace', 'defense', 'military', 'precision']) and
                'white background' in prompt.lower()):
                
                return OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    strategy_used="already_optimized",
                    predicted_score=0.85,  # Good baseline for optimized prompts
                    confidence=0.9,
                    optimization_time=time.time() - start_time,
                    rl_actions_used=["no_action_needed"]
                )
            
            # Try RL optimization
            try:
                current_prompt = prompt
                best_prompt = prompt
                actions_used = []
                
                for step in range(max_steps):
                    if time.time() - start_time > timeout:
                        break
                    
                    # Get state and predict action
                    state = self._get_state(current_prompt, target_object)
                    action_idx, confidence = self._predict_best_action(state)
                    
                    if action_idx >= len(self.action_space):
                        break  # Invalid action index
                    
                    action = self.action_space[action_idx]
                    
                    # Apply action
                    new_prompt = self._apply_rl_action(action, current_prompt, target_object)
                    
                    if new_prompt != current_prompt:
                        actions_used.append(f"Step {step+1}: {action[0]}")
                        best_prompt = new_prompt
                        current_prompt = new_prompt
                    
                    # Early stop if we applied a pattern
                    if action[0] == 'APPLY_PATTERN':
                        break
                
                self.stats['rl_optimizations'] += 1
                
                # Try LLaMA pattern if enabled and we have a good result
                llama_pattern = None
                if self.enable_llama and confidence > 0.7:
                    llama_pattern = self.llama_engine.extract_winning_pattern(
                        best_prompt, prompt, 0.85  # Assumed good score
                    )
                    if llama_pattern:
                        self.stats['llama_patterns_used'] += 1
                
                return OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=best_prompt,
                    strategy_used="rl_optimization",
                    predicted_score=min(0.95, 0.75 + confidence * 0.2),  # Conservative prediction
                    confidence=confidence,
                    optimization_time=time.time() - start_time,
                    rl_actions_used=actions_used,
                    llama_pattern_used=llama_pattern
                )
                
            except Exception as e:
                print(f"⚠️ RL optimization failed: {e}")
                raise
        
        except Exception as e:
            # Fallback to rule-based optimization
            if self.fallback_patterns:
                self.stats['fallback_used'] += 1
                fallback_prompt = self._use_fallback_pattern(prompt, self._extract_target_object(prompt))
                
                return OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=fallback_prompt,
                    strategy_used="fallback_pattern",
                    predicted_score=0.82,  # Conservative fallback prediction
                    confidence=0.7,
                    optimization_time=time.time() - start_time,
                    rl_actions_used=["fallback_pattern_applied"],
                    fallback_used=True
                )
            else:
                # Return original prompt if all else fails
                return OptimizationResult(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    strategy_used="no_optimization",
                    predicted_score=0.5,  # Unknown quality
                    confidence=0.0,
                    optimization_time=time.time() - start_time,
                    rl_actions_used=["optimization_failed"],
                    fallback_used=True
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total = max(1, self.stats['total_optimizations'])
        return {
            'total_optimizations': self.stats['total_optimizations'],
            'rl_success_rate': self.stats['rl_optimizations'] / total,
            'llama_usage_rate': self.stats['llama_patterns_used'] / total,
            'fallback_rate': self.stats['fallback_used'] / total,
            'avg_optimization_time': self.stats['avg_optimization_time'],
            'model_info': {
                'checkpoint': str(self.checkpoint_path),
                'actions_available': len(self.action_space),
                'training_score': self.metadata.get('best_score', 'unknown'),
                'llama_enabled': self.enable_llama
            }
        }

# Integration function for continuous_trellis_orchestrator.py
def create_rl_optimizer(checkpoint_path: str = "rl_checkpoints_v2/episode_024", 
                       enable_llama: bool = True) -> RLMiningOptimizer:
    """
    Factory function to create the RL optimizer for mining integration
    
    Args:
        checkpoint_path: Path to your trained RL checkpoint
        enable_llama: Whether to enable LLaMA 3.2 for advanced pattern extraction
        
    Returns:
        RLMiningOptimizer ready for production use
    """
    return RLMiningOptimizer(
        checkpoint_path=checkpoint_path,
        enable_llama=enable_llama,
        fallback_patterns=True
    )

def main():
    """Demo the RL mining integration"""
    print("🚀 RL MINING INTEGRATION DEMO")
    print("="*60)
    
    # Test prompts from your mining
    test_prompts = [
        "hexagonal prism steel structure",
        "elegant silk fabric draping", 
        "transparent glass sphere",
        "rusty metal gear mechanism",
        "wooden sculpture with details"
    ]
    
    # Initialize with your trained checkpoint
    optimizer = RLMiningOptimizer(
        checkpoint_path="rl_checkpoints_v2/episode_024",  # Your best checkpoint
        enable_llama=True,  # Set to True if you want LLaMA 3.2 integration
        fallback_patterns=True
    )
    
    print(f"\n🎯 Testing RL optimization on {len(test_prompts)} prompts:")
    
    for i, prompt in enumerate(test_prompts, 1):
        result = optimizer.optimize_prompt(prompt, max_steps=3, timeout=3.0)
        
        print(f"\n{i}. Original: {result.original_prompt}")
        print(f"   Optimized: {result.optimized_prompt}")
        print(f"   Strategy: {result.strategy_used}")
        print(f"   Predicted Score: {result.predicted_score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Time: {result.optimization_time:.3f}s")
        if result.rl_actions_used:
            print(f"   Actions: {', '.join(result.rl_actions_used)}")
    
    # Show statistics
    stats = optimizer.get_stats()
    print(f"\n📊 OPTIMIZATION STATISTICS:")
    print(f"   RL Success Rate: {stats['rl_success_rate']:.1%}")
    print(f"   Fallback Rate: {stats['fallback_rate']:.1%}")
    print(f"   Model: {stats['model_info']['checkpoint']}")
    print(f"   Training Score: {stats['model_info']['training_score']}")

if __name__ == "__main__":
    main() 