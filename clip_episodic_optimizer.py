#!/usr/bin/env python3
"""
Episodic CLIP Score Optimizer with Persistent Memory and Intelligent LoRA Routing
===============================================================================
🧠 Persistent memory across sessions
🔄 Episodic learning with cross-session knowledge
🎯 Force improvement over historical best scores
📊 Rich context injection to LLM with past attempts
🚫 Early termination on stuck optimization
🎯 INTELLIGENT LORA ROUTING: Pre-optimization generator selection
📈 MULTI-GENERATOR HISTORY: Track generator choices across episodes
🏆 TIE-BREAKING: Test multiple generators when router is uncertain
"""

import json
import time
import random
import logging
import os
import statistics
import requests
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from get_max_clip_score import CLIPScoreMaximizer, RLLoopAgent, OptimizationAttempt, RLOptimizationSession

# Import the hybrid router
from hybrid_ultimate_router import HybridUltimateRouter, RouterResult

@dataclass
class GeneratorHistory:
    """History of generator choices for a prompt across episodes"""
    generator_name: str
    episode_count: int
    total_clip_score: float
    best_clip_score: float
    last_used_episode: int
    success_rate: float  # Percentage of times this generator achieved target

@dataclass
class EpisodicMemory:
    """Persistent memory entry for a prompt across episodes"""
    original_prompt: str
    best_score: float
    best_prompt: str
    best_negative_prompt: str
    best_generator: str  # NEW: Track best performing generator
    attempt_history: List[Dict[str, Any]]
    episodes_run: int
    total_attempts: int
    first_seen: str
    last_optimized: str
    score_progression: List[float]
    strategy_performance: Dict[str, Dict[str, Any]]
    successful_patterns: List[str]
    failed_patterns: List[str]
    # NEW: Generator routing history
    generator_history: Dict[str, GeneratorHistory]
    router_decisions: List[Dict[str, Any]]  # Track router decisions per episode

class MultiGeneratorCLIPOptimizer:
    """
    Multi-generator CLIP optimizer with intelligent LoRA routing
    """
    
    def __init__(self, 
                 num_episodes: int = 50,
                 target_score: float = 0.85,
                 max_rounds_per_episode: int = 15,
                 memory_file: str = "episodic_clip_memory.json",
                 log_dir: str = "episodic_clip_logs",
                 improvement_threshold: float = 0.05,
                 enable_router: bool = True,
                 tie_break_threshold: float = 0.1):  # Minimum score difference for tie-breaking
        
        self.num_episodes = num_episodes
        self.target_score = target_score
        self.max_rounds_per_episode = max_rounds_per_episode
        self.memory_file = Path(memory_file)
        self.log_dir = Path(log_dir)
        self.improvement_threshold = improvement_threshold
        self.enable_router = enable_router
        self.tie_break_threshold = tie_break_threshold
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.log_dir / f"multi_generator_clip_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize LoRA router
        if self.enable_router:
            self.lora_router = HybridUltimateRouter()
            self.logger.info("🎯 LoRA Router initialized")
        
        # Generator endpoints mapping
        self.generator_endpoints = {
            "Patched Realism": "http://localhost:8096/generate_image/patched_realism/",
            "Team Fortress 2 Style": "http://localhost:8096/generate_image/tf2_style/",
            "Cartoon 3D Render": "http://localhost:8096/generate_image/cartoon_3d/",
            "3D Game Assets": "http://localhost:8096/generate_image/game_assets/",
            "Game Icon Institute": "http://localhost:8096/generate_image/sd15_game_icon/",
            "Cinema Style": "http://localhost:8096/generate_image/cinema/",
            "Flux Isometric 3D": "http://localhost:8096/generate_image/isometric_3d/",
            "Baolei Style": "http://localhost:8096/generate_image/baolei/"
        }
        
        # Initialize CLIP maximizer and RL agent
        self.clip_maximizer = CLIPScoreMaximizer(target_score=target_score)
        self.rl_agent = RLLoopAgent(
            clip_maximizer=self.clip_maximizer,
            memory_file=str(self.log_dir / "rl_memory.json")
        )
        
        # Override RL parameters for episodic settings
        self.rl_agent.max_optimization_rounds = max_rounds_per_episode
        self.rl_agent.min_score_threshold = target_score
        self.rl_agent.convergence_threshold = 0.01  # Stricter convergence
        
        # Load episodic memory
        self.episodic_memory: Dict[str, EpisodicMemory] = {}
        self._load_episodic_memory()
        
        # Episode tracking
        self.episode_results = []
        self.global_insights = []
        
        self.logger.info(f"🧠 Multi-Generator CLIP Optimizer initialized")
        self.logger.info(f"   Target score: {target_score}")
        self.logger.info(f"   Episodes: {num_episodes}")
        self.logger.info(f"   Max rounds per episode: {max_rounds_per_episode}")
        self.logger.info(f"   Router enabled: {enable_router}")
        self.logger.info(f"   Memory entries: {len(self.episodic_memory)}")
    
    def _load_episodic_memory(self):
        """Load episodic memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                for prompt, memory_data in data.items():
                    # Handle legacy memory format
                    if 'generator_history' not in memory_data:
                        memory_data['generator_history'] = {}
                        memory_data['router_decisions'] = []
                        memory_data['best_generator'] = "Cinema Style"  # Default
                    
                    # Convert generator_history to proper format
                    if memory_data['generator_history']:
                        converted_history = {}
                        for gen_name, gen_data in memory_data['generator_history'].items():
                            if isinstance(gen_data, dict):
                                converted_history[gen_name] = GeneratorHistory(**gen_data)
                            else:
                                # Handle legacy format
                                converted_history[gen_name] = GeneratorHistory(
                                    generator_name=gen_name,
                                    episode_count=gen_data.get('episode_count', 1),
                                    total_clip_score=gen_data.get('total_clip_score', 0.0),
                                    best_clip_score=gen_data.get('best_clip_score', 0.0),
                                    last_used_episode=gen_data.get('last_used_episode', 0),
                                    success_rate=gen_data.get('success_rate', 0.0)
                                )
                        memory_data['generator_history'] = converted_history
                    
                    self.episodic_memory[prompt] = EpisodicMemory(**memory_data)
                
                self.logger.info(f"📚 Loaded episodic memory: {len(self.episodic_memory)} prompts")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load episodic memory: {e}")
                self.episodic_memory = {}
        else:
            self.logger.info("📄 Starting fresh episodic memory")
    
    def _save_episodic_memory(self):
        """Save episodic memory to file"""
        try:
            data = {}
            for prompt, memory in self.episodic_memory.items():
                memory_dict = asdict(memory)
                # Convert GeneratorHistory objects to dicts
                memory_dict['generator_history'] = {
                    gen_name: asdict(gen_data) for gen_name, gen_data in memory.generator_history.items()
                }
                data[prompt] = memory_dict
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Failed to save episodic memory: {e}")
    
    def _select_generator_for_prompt(self, prompt: str, episode_num: int) -> Tuple[str, str, Dict[str, Any]]:
        """Select the best generator for a prompt using router + history"""
        
        if not self.enable_router:
            return "Cinema Style", self.generator_endpoints["Cinema Style"], {"method": "default"}
        
        # Check if we have memory for this prompt
        has_memory = prompt in self.episodic_memory
        memory = self.episodic_memory.get(prompt, None)
        
        if has_memory and memory.generator_history:
            # Analyze generator history
            generator_stats = {}
            for gen_name, gen_history in memory.generator_history.items():
                generator_stats[gen_name] = {
                    'episode_count': gen_history.episode_count,
                    'best_score': gen_history.best_clip_score,
                    'avg_score': gen_history.total_clip_score / gen_history.episode_count if gen_history.episode_count > 0 else 0.0,
                    'success_rate': gen_history.success_rate,
                    'last_used': gen_history.last_used_episode
                }
            
            # Find generators with highest episode count (majority)
            max_episodes = max(stats['episode_count'] for stats in generator_stats.values())
            majority_generators = [
                gen_name for gen_name, stats in generator_stats.items() 
                if stats['episode_count'] == max_episodes
            ]
            
            self.logger.info(f"   📊 Generator history: {len(memory.generator_history)} generators")
            self.logger.info(f"   📊 Majority generators: {majority_generators} (used {max_episodes} times)")
            
            if len(majority_generators) == 1:
                # Clear majority - use historical best
                selected_generator = majority_generators[0]
                self.logger.info(f"   🎯 HISTORICAL MAJORITY: {selected_generator}")
                return selected_generator, self.generator_endpoints[selected_generator], {
                    "method": "historical_majority",
                    "episode_count": max_episodes,
                    "generator_stats": generator_stats
                }
            
            elif len(majority_generators) > 1:
                # Tie - need to test multiple generators
                self.logger.info(f"   🤝 TIE DETECTED: {majority_generators}")
                return self._resolve_generator_tie(prompt, majority_generators, episode_num)
        
        # No history or no clear majority - use router
        self.logger.info(f"   🧠 NO HISTORY - USING ROUTER")
        router_result = self.lora_router.route_hybrid(prompt)
        selected_generator = router_result.recommended_lora
        
        self.logger.info(f"   🎯 ROUTER SELECTION: {selected_generator}")
        return selected_generator, self.generator_endpoints[selected_generator], {
            "method": "router_selection",
            "router_confidence": router_result.confidence,
            "router_reasoning": router_result.reasoning,
            "alternatives": router_result.alternatives
        }
    
    def _resolve_generator_tie(self, prompt: str, tied_generators: List[str], episode_num: int) -> Tuple[str, str, Dict[str, Any]]:
        """Resolve tie by testing multiple generators and selecting the best"""
        
        self.logger.info(f"   🔬 TIE-BREAKING: Testing {len(tied_generators)} generators")
        
        # Test each tied generator with a quick optimization
        generator_scores = {}
        
        for generator in tied_generators:
            self.logger.info(f"   🔬 Testing generator: {generator}")
            
            try:
                # Quick optimization test (fewer rounds for tie-breaking)
                test_result = self._quick_optimization_test(prompt, generator, max_rounds=5)
                generator_scores[generator] = test_result.get('final_score', 0.0)
                
                self.logger.info(f"   🔬 {generator} score: {generator_scores[generator]:.4f}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to test {generator}: {e}")
                generator_scores[generator] = 0.0
        
        # Find the best generator
        best_generator = max(generator_scores.keys(), key=lambda g: generator_scores[g])
        best_score = generator_scores[best_generator]
        
        # Check if scores are close (within threshold)
        close_generators = [
            gen for gen, score in generator_scores.items()
            if abs(score - best_score) <= self.tie_break_threshold
        ]
        
        if len(close_generators) > 1:
            self.logger.info(f"   🤝 Scores too close - using historical success rate")
            # Use historical success rate as tie-breaker
            memory = self.episodic_memory.get(prompt)
            if memory and memory.generator_history:
                best_success_rate = 0.0
                for gen in close_generators:
                    if gen in memory.generator_history:
                        success_rate = memory.generator_history[gen].success_rate
                        if success_rate > best_success_rate:
                            best_success_rate = success_rate
                            best_generator = gen
        
        self.logger.info(f"   🏆 TIE RESOLVED: {best_generator} (score: {best_score:.4f})")
        
        return best_generator, self.generator_endpoints[best_generator], {
            "method": "tie_break_testing",
            "tied_generators": tied_generators,
            "generator_scores": generator_scores,
            "best_score": best_score,
            "close_generators": close_generators
        }
    
    def _quick_optimization_test(self, prompt: str, generator: str, max_rounds: int = 5) -> Dict[str, Any]:
        """Quick optimization test for tie-breaking"""
        
        # Temporarily modify the CLIP maximizer endpoint
        original_endpoint = self.clip_maximizer.dit_server_url
        self.clip_maximizer.dit_server_url = self.generator_endpoints[generator]
        
        try:
            # Run quick optimization
            result = self.rl_agent.optimize_with_rl_loop(
                prompt, 
                seed=42, 
                max_rounds=max_rounds
            )
            return result
        finally:
            # Restore original endpoint
            self.clip_maximizer.dit_server_url = original_endpoint
    
    def _build_historical_context(self, prompt: str) -> str:
        """Build rich historical context for the LLM"""
        if prompt not in self.episodic_memory:
            return "HISTORICAL CONTEXT: This is the first time optimizing this prompt."
        
        memory = self.episodic_memory[prompt]
        
        context = f"""HISTORICAL CONTEXT FOR PROMPT: "{prompt}"
        
🎯 CURRENT BEST TO BEAT:
   Score: {memory.best_score:.4f}
   Prompt: "{memory.best_prompt}"
   Negative: "{memory.best_negative_prompt}"
   Generator: {memory.best_generator}
   
📊 PERFORMANCE HISTORY:
   Episodes run: {memory.episodes_run}
   Total attempts: {memory.total_attempts}
   Score progression: {[f'{s:.3f}' for s in memory.score_progression[-10:]]}
   Last optimized: {memory.last_optimized}
   
🎮 GENERATOR HISTORY:"""
        
        for gen_name, gen_history in memory.generator_history.items():
            context += f"\n   {gen_name}: {gen_history.episode_count} episodes, best={gen_history.best_clip_score:.3f}, success_rate={gen_history.success_rate:.1%}"
        
        context += f"""
   
🧠 LEARNED PATTERNS:
   Successful patterns: {memory.successful_patterns}
   Failed patterns: {memory.failed_patterns}
   
🎛️ STRATEGY PERFORMANCE:"""
        
        for strategy, perf in memory.strategy_performance.items():
            context += f"\n   {strategy}: avg={perf.get('avg_score', 0):.3f}, attempts={perf.get('attempts', 0)}"
        
        context += f"""
        
🚨 IMPROVEMENT REQUIREMENT:
   You MUST achieve a score > {memory.best_score:.4f} (current best)
   Minimum improvement needed: {self.improvement_threshold:.3f}
   Target score: {self.target_score:.3f}
   
💡 RECENT INSIGHTS:
"""
        
        # Add recent attempt insights
        if memory.attempt_history:
            recent_attempts = memory.attempt_history[-5:]
            for i, attempt in enumerate(recent_attempts, 1):
                context += f"   {i}. Score: {attempt.get('score', 0):.3f}, Generator: {attempt.get('generator', 'unknown')}, Strategy: {attempt.get('strategy', 'unknown')}\n"
                context += f"      Prompt: {attempt.get('prompt', '')[:100]}...\n"
        
        return context
    
    def optimize_prompt_episodically(self, prompt: str, episode_num: int) -> Dict[str, Any]:
        """Optimize a single prompt with episodic memory context and generator selection"""
        episode_start_time = time.time()
        
        self.logger.info(f"\n🔄 Episode {episode_num}: Optimizing '{prompt}'")
        
        # Check if we already have memory for this prompt
        has_memory = prompt in self.episodic_memory
        memory = self.episodic_memory.get(prompt, None)
        
        if has_memory:
            self.logger.info(f"   📚 Found historical data: best score {memory.best_score:.4f}")
            self.logger.info(f"   📚 Episodes run: {memory.episodes_run}, Total attempts: {memory.total_attempts}")
            
            # Check if already achieved target
            if memory.best_score >= self.target_score:
                self.logger.info(f"   ✅ Target already achieved! Skipping optimization.")
                return {
                    'episode': episode_num,
                    'original_prompt': prompt,
                    'result': 'target_already_achieved',
                    'best_score': memory.best_score,
                    'best_prompt': memory.best_prompt,
                    'best_generator': memory.best_generator,
                    'duration': 0.0,
                    'attempts': 0
                }
        
        # Select generator for this episode
        selected_generator, generator_endpoint, selection_info = self._select_generator_for_prompt(prompt, episode_num)
        
        self.logger.info(f"   🎮 Selected generator: {selected_generator}")
        self.logger.info(f"   🎮 Selection method: {selection_info['method']}")
        
        # Temporarily set the generator endpoint
        original_endpoint = self.clip_maximizer.dit_server_url
        self.clip_maximizer.dit_server_url = generator_endpoint
        
        try:
            # Build historical context
            historical_context = self._build_historical_context(prompt)
            
            # Run RL optimization with historical context
            result = self.rl_agent.optimize_with_rl_loop(prompt, seed=42)
            
            final_score = result.get('final_score', 0.0)
            final_prompt = result.get('final_optimized_prompt', prompt)
            attempts = result.get('total_rounds', 0)
            session_duration = result.get('processing_time', 0.0)
            
            self.logger.info(f"   📊 Final score: {final_score:.4f}")
            self.logger.info(f"   📊 Rounds: {attempts}")
            self.logger.info(f"   📊 Duration: {session_duration:.1f}s")
            
            # Check for improvement requirement
            improvement_achieved = True
            improvement_amount = 0.0
            
            if has_memory:
                improvement_amount = final_score - memory.best_score
                improvement_achieved = improvement_amount >= self.improvement_threshold
                
                if not improvement_achieved:
                    self.logger.warning(f"   ⚠️ Insufficient improvement: {improvement_amount:.4f} < {self.improvement_threshold:.3f}")
                else:
                    self.logger.info(f"   🎉 Improvement achieved: +{improvement_amount:.4f}")
            
            # Update episodic memory
            self._update_episodic_memory(prompt, result, historical_context, episode_num, selected_generator, selection_info)
            
            episode_result = {
                'episode': episode_num,
                'original_prompt': prompt,
                'selected_generator': selected_generator,
                'selection_method': selection_info['method'],
                'final_score': final_score,
                'final_prompt': final_prompt,
                'attempts': attempts,
                'duration': session_duration,
                'had_previous_memory': has_memory,
                'improvement_achieved': improvement_achieved,
                'improvement_amount': improvement_amount,
                'target_achieved': final_score >= self.target_score,
                'convergence_achieved': result.get('convergence_achieved', False),
                'convergence_reason': result.get('convergence_reason', 'Not converged'),
                'exploration_ratio': result.get('exploration_ratio', 0.0),
                'learned_insights': result.get('learned_insights', [])
            }
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"   ❌ Episode failed: {e}")
            return {
                'episode': episode_num,
                'original_prompt': prompt,
                'selected_generator': selected_generator,
                'result': 'failed',
                'error': str(e),
                'duration': time.time() - episode_start_time
            }
        finally:
            # Restore original endpoint
            self.clip_maximizer.dit_server_url = original_endpoint
    
    def _update_episodic_memory(self, prompt: str, rl_result: Dict[str, Any], 
                              historical_context: str, episode_num: int,
                              selected_generator: str, selection_info: Dict[str, Any]):
        """Update episodic memory with new optimization results"""
        
        final_score = rl_result.get('final_score', 0.0)
        final_prompt = rl_result.get('final_optimized_prompt', prompt)
        attempts = rl_result.get('total_rounds', 0)
        score_progression = rl_result.get('score_progression', [])
        strategy_sequence = rl_result.get('strategy_sequence', [])
        learned_insights = rl_result.get('learned_insights', [])
        
        # Extract patterns from successful/failed attempts
        successful_patterns = []
        failed_patterns = []
        
        if score_progression and len(score_progression) > 1:
            best_score_idx = score_progression.index(max(score_progression))
            if best_score_idx < len(strategy_sequence):
                successful_patterns.append(strategy_sequence[best_score_idx])
            
            # Find consistently poor strategies
            for i, (score, strategy) in enumerate(zip(score_progression, strategy_sequence)):
                if score < statistics.mean(score_progression) * 0.8:  # 20% below average
                    failed_patterns.append(strategy)
        
        if prompt in self.episodic_memory:
            # Update existing memory
            memory = self.episodic_memory[prompt]
            
            # Update if new score is better
            if final_score > memory.best_score:
                memory.best_score = final_score
                memory.best_prompt = final_prompt
                memory.best_generator = selected_generator
                memory.best_negative_prompt = "blurry, shadows, artistic, grainy, low-resolution"  # Default
            
            memory.episodes_run += 1
            memory.total_attempts += attempts
            memory.last_optimized = datetime.now().isoformat()
            memory.score_progression.extend(score_progression)
            
            # Keep only last 50 scores
            if len(memory.score_progression) > 50:
                memory.score_progression = memory.score_progression[-50:]
            
            # Update strategy performance
            for strategy in strategy_sequence:
                if strategy not in memory.strategy_performance:
                    memory.strategy_performance[strategy] = {'attempts': 0, 'total_score': 0.0}
                
                memory.strategy_performance[strategy]['attempts'] += 1
                memory.strategy_performance[strategy]['total_score'] += final_score
                memory.strategy_performance[strategy]['avg_score'] = (
                    memory.strategy_performance[strategy]['total_score'] / 
                    memory.strategy_performance[strategy]['attempts']
                )
            
            # Update patterns
            memory.successful_patterns.extend(successful_patterns)
            memory.failed_patterns.extend(failed_patterns)
            
            # Keep unique and recent patterns
            memory.successful_patterns = list(set(memory.successful_patterns))[-10:]
            memory.failed_patterns = list(set(memory.failed_patterns))[-10:]
            
        else:
            # Create new memory entry
            strategy_performance = {}
            for strategy in strategy_sequence:
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'attempts': 0, 'total_score': 0.0}
                
                strategy_performance[strategy]['attempts'] += 1
                strategy_performance[strategy]['total_score'] += final_score
                strategy_performance[strategy]['avg_score'] = (
                    strategy_performance[strategy]['total_score'] / 
                    strategy_performance[strategy]['attempts']
                )
            
            self.episodic_memory[prompt] = EpisodicMemory(
                original_prompt=prompt,
                best_score=final_score,
                best_prompt=final_prompt,
                best_generator=selected_generator,
                best_negative_prompt="blurry, shadows, artistic, grainy, low-resolution",
                attempt_history=[],
                episodes_run=1,
                total_attempts=attempts,
                first_seen=datetime.now().isoformat(),
                last_optimized=datetime.now().isoformat(),
                score_progression=score_progression,
                strategy_performance=strategy_performance,
                successful_patterns=successful_patterns,
                failed_patterns=failed_patterns,
                generator_history={},
                router_decisions=[]
            )
        
        # Update generator history
        if selected_generator not in self.episodic_memory[prompt].generator_history:
            self.episodic_memory[prompt].generator_history[selected_generator] = GeneratorHistory(
                generator_name=selected_generator,
                episode_count=0,
                total_clip_score=0.0,
                best_clip_score=0.0,
                last_used_episode=0,
                success_rate=0.0
            )
        
        gen_history = self.episodic_memory[prompt].generator_history[selected_generator]
        gen_history.episode_count += 1
        gen_history.total_clip_score += final_score
        gen_history.last_used_episode = episode_num
        
        if final_score > gen_history.best_clip_score:
            gen_history.best_clip_score = final_score
        
        # Update success rate
        target_achieved = final_score >= self.target_score
        total_episodes = gen_history.episode_count
        successful_episodes = sum(1 for attempt in self.episodic_memory[prompt].attempt_history 
                                if attempt.get('generator') == selected_generator and 
                                attempt.get('score', 0) >= self.target_score)
        if target_achieved:
            successful_episodes += 1
        
        gen_history.success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        # Add router decision to history
        router_decision = {
            'episode': episode_num,
            'selected_generator': selected_generator,
            'selection_method': selection_info['method'],
            'selection_info': selection_info,
            'final_score': final_score,
            'target_achieved': target_achieved,
            'timestamp': datetime.now().isoformat()
        }
        self.episodic_memory[prompt].router_decisions.append(router_decision)
        
        # Keep only last 20 router decisions
        if len(self.episodic_memory[prompt].router_decisions) > 20:
            self.episodic_memory[prompt].router_decisions = self.episodic_memory[prompt].router_decisions[-20:]
        
        # Add this attempt to history
        attempt_record = {
            'episode': episode_num,
            'score': final_score,
            'prompt': final_prompt,
            'generator': selected_generator,
            'strategy': strategy_sequence[-1] if strategy_sequence else 'unknown',
            'attempts': attempts,
            'timestamp': datetime.now().isoformat(),
            'insights': learned_insights
        }
        
        self.episodic_memory[prompt].attempt_history.append(attempt_record)
        
        # Keep only last 20 attempts
        if len(self.episodic_memory[prompt].attempt_history) > 20:
            self.episodic_memory[prompt].attempt_history = self.episodic_memory[prompt].attempt_history[-20:]
    
    def run_all_episodes(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Run episodic optimization on all test prompts with multi-generator support"""
        
        overall_start_time = time.time()
        
        self.logger.info(f"\n🚀 Starting Multi-Generator CLIP optimization")
        self.logger.info(f"   Test prompts: {len(test_prompts)}")
        self.logger.info(f"   Episodes per prompt: {self.num_episodes}")
        self.logger.info(f"   Target score: {self.target_score}")
        self.logger.info(f"   Router enabled: {self.enable_router}")
        
        all_results = []
        
        for episode in range(1, self.num_episodes + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPISODE {episode}/{self.num_episodes}")
            self.logger.info(f"{'='*60}")
            
            episode_results = []
            
            # Randomly shuffle prompts each episode for variety
            shuffled_prompts = test_prompts.copy()
            random.shuffle(shuffled_prompts)
            
            for prompt_idx, prompt in enumerate(shuffled_prompts, 1):
                self.logger.info(f"\n--- Prompt {prompt_idx}/{len(shuffled_prompts)} ---")
                
                result = self.optimize_prompt_episodically(prompt, episode)
                episode_results.append(result)
                all_results.append(result)
                
                # Save memory after each prompt
                self._save_episodic_memory()
            
            # Episode summary
            successful_episodes = [r for r in episode_results if r.get('target_achieved', False)]
            improved_episodes = [r for r in episode_results if r.get('improvement_achieved', False)]
            
            # Generator usage statistics
            generator_usage = {}
            for result in episode_results:
                if 'selected_generator' in result:
                    gen = result['selected_generator']
                    generator_usage[gen] = generator_usage.get(gen, 0) + 1
            
            self.logger.info(f"\n📊 Episode {episode} Summary:")
            self.logger.info(f"   Targets achieved: {len(successful_episodes)}/{len(episode_results)}")
            self.logger.info(f"   Improvements made: {len(improved_episodes)}/{len(episode_results)}")
            self.logger.info(f"   Generator usage: {generator_usage}")
            
            avg_score = statistics.mean([r.get('final_score', 0) for r in episode_results if 'final_score' in r])
            self.logger.info(f"   Average score: {avg_score:.4f}")
        
        # Final analysis
        overall_duration = time.time() - overall_start_time
        
        analysis = self._analyze_episodic_learning(all_results)
        
        final_summary = {
            'total_episodes': self.num_episodes,
            'total_prompts': len(test_prompts),
            'total_duration': overall_duration,
            'all_results': all_results,
            'analysis': analysis,
            'memory_entries': len(self.episodic_memory),
            'router_enabled': self.enable_router,
            'completed_at': datetime.now().isoformat()
        }
        
        self._save_final_results(final_summary)
        self._log_final_summary(analysis, overall_duration)
        
        return final_summary
    
    def _analyze_episodic_learning(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Analyze learning progress across episodes with generator insights"""
        
        analysis = {
            'total_optimizations': len(all_results),
            'successful_optimizations': 0,
            'improved_optimizations': 0,
            'average_score': 0.0,
            'score_distribution': {},
            'strategy_effectiveness': {},
            'learning_trends': {},
            'memory_insights': {},
            'generator_insights': {}  # NEW: Generator-specific insights
        }
        
        # Basic stats
        valid_results = [r for r in all_results if 'final_score' in r]
        if valid_results:
            scores = [r['final_score'] for r in valid_results]
            analysis['average_score'] = statistics.mean(scores)
            analysis['successful_optimizations'] = len([r for r in valid_results if r.get('target_achieved', False)])
            analysis['improved_optimizations'] = len([r for r in valid_results if r.get('improvement_achieved', False)])
            
            # Score distribution
            analysis['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        
        # Generator insights
        generator_stats = {}
        for result in valid_results:
            if 'selected_generator' in result:
                gen = result['selected_generator']
                if gen not in generator_stats:
                    generator_stats[gen] = {
                        'usage_count': 0,
                        'total_score': 0.0,
                        'best_score': 0.0,
                        'target_achievements': 0,
                        'selection_methods': {}
                    }
                
                stats = generator_stats[gen]
                stats['usage_count'] += 1
                stats['total_score'] += result['final_score']
                stats['best_score'] = max(stats['best_score'], result['final_score'])
                
                if result.get('target_achieved', False):
                    stats['target_achievements'] += 1
                
                method = result.get('selection_method', 'unknown')
                stats['selection_methods'][method] = stats['selection_methods'].get(method, 0) + 1
        
        # Calculate averages and success rates
        for gen, stats in generator_stats.items():
            stats['avg_score'] = stats['total_score'] / stats['usage_count']
            stats['success_rate'] = stats['target_achievements'] / stats['usage_count']
        
        analysis['generator_insights'] = generator_stats
        
        # Memory insights
        total_attempts = sum(memory.total_attempts for memory in self.episodic_memory.values())
        total_episodes_run = sum(memory.episodes_run for memory in self.episodic_memory.values())
        
        analysis['memory_insights'] = {
            'unique_prompts_optimized': len(self.episodic_memory),
            'total_optimization_attempts': total_attempts,
            'total_episodes_across_prompts': total_episodes_run,
            'average_attempts_per_prompt': total_attempts / len(self.episodic_memory) if self.episodic_memory else 0,
            'prompts_achieving_target': len([m for m in self.episodic_memory.values() if m.best_score >= self.target_score])
        }
        
        return analysis
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results to file"""
        results_file = self.log_dir / f"multi_generator_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
    
    def _log_final_summary(self, analysis: Dict[str, Any], duration: float):
        """Log final summary with generator insights"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"MULTI-GENERATOR CLIP OPTIMIZATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"📊 Total Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        self.logger.info(f"📊 Total Optimizations: {analysis['total_optimizations']}")
        self.logger.info(f"📊 Successful (target achieved): {analysis['successful_optimizations']}")
        self.logger.info(f"📊 Improved over previous: {analysis['improved_optimizations']}")
        self.logger.info(f"📊 Average Score: {analysis['average_score']:.4f}")
        
        # Generator insights
        generator_insights = analysis['generator_insights']
        self.logger.info(f"\n🎮 GENERATOR INSIGHTS:")
        for gen_name, stats in generator_insights.items():
            self.logger.info(f"   {gen_name}:")
            self.logger.info(f"     Usage: {stats['usage_count']} times")
            self.logger.info(f"     Avg Score: {stats['avg_score']:.4f}")
            self.logger.info(f"     Best Score: {stats['best_score']:.4f}")
            self.logger.info(f"     Success Rate: {stats['success_rate']:.1%}")
            self.logger.info(f"     Selection Methods: {stats['selection_methods']}")
        
        memory_insights = analysis['memory_insights']
        self.logger.info(f"\n🧠 MEMORY INSIGHTS:")
        self.logger.info(f"   Unique prompts optimized: {memory_insights['unique_prompts_optimized']}")
        self.logger.info(f"   Total attempts: {memory_insights['total_optimization_attempts']}")
        self.logger.info(f"   Prompts achieving target: {memory_insights['prompts_achieving_target']}")
        
        self.logger.info(f"📚 Episodic memory entries: {len(self.episodic_memory)}")
        self.logger.info(f"💾 Memory saved to: {self.memory_file}")

def main():
    """Main function for running multi-generator episodic CLIP optimization"""
    
    # Example test prompts
    # test_prompts = [
    #     "luxurious cream sedan elegant",
    #     "ornate heart-shaped pendant",
    #     "small wooden hammer with screws",
    #     "robot in sitting down position",
    #     "mystical orb pulsating with arcane energy"
    # ]

    test_prompts = [
         "plastic straw of drink", # start test 
        "small yellow triangular wooden kitchen knife",
        "ukulele sporting vibrant sunflower yellow",
        "white elf holding ancient golden staff",
        "purple troll with long hair and green",
        "long narrow black drill",
        "modern futuristic assault rifle",
        "velvet green sofa with gold piping",
        "sharp red bayonet on rifle",
        "statue of animal flying on pedestal",
        "banana split with toppings",
        "large shiny silver screwdriver",
        "glass vase with intricate blue patterns",
        "huge pink dinosaur with blue spots",
        "deep green emerald sleek cut",
        "charming pink minivan sporting floral decals",
        "sleek blue cricket bat poised",
        "lime green golf club",
        "eggshell green metal screwdriver with silver handle",
        "robot with red and blue stripes in crate",
        "sturdy plastic bucket bright yellow color useful for painting",
        "whole wheat pizza with extra cheese",
        "green alien with large head and big eyes",
        "green hammer with wood",
        "matte-black octahedral robot with sharp edges",
        "robot with orange star shape",
        "small white square-shaped quartz crystal with gold chain",
        "amethyst teardrop-shaped earrings set in gold",
        "blue polycarbonate drill bit",
        "greek amphora scene detail",
        "necklace with opals in it",
        "harmonica in classic matte black",
        "rose gold locket necklace with floral",
        "robot with olive green circle shape",
        "green sports car sleek aerodynamic",
        "glossy blue saxophone with sleek lines",
        "small and oval shaped black gemstone with rough surface and dark color",
        "green leafy staff with heart-shaped handle",
        "bright orange baseball bat with intricate wood grain",
        "limestone figure of dancing faun",
        "heavy-duty wrench black and yellow stripes",
        "gold-trimmed sword elegant curve",
        "lightweight white chisel with pointed end",
        "radiant heart-shaped opal necklace",
        "large golden trowel",
        "green squash racket streamlined",
        "dull irregular quartz crystal",
        "sturdy metal screwdriver tip pointed blue handle",
        "black clarinet classic and timeless",
        "green rubber gloves pack",
        "sleek green marimba",
        "glowing tower with spiraling silver staircases",
        "matte black bass guitar with sleek modern",
        "blue putty knife flat surface",
        "white ice cube tray",
        "antique silver candlestick two-pronged",
        "old-fashioned hammer with black head",
        "matte blue cricket bat stands ready",
        "drill bit yellow slender pointed tip",
        "onyx helmet crowned by sapphire crest",
        "textured jasper mineral chunk",
        "polished silver clarinet with elegant",
        "golden intricate long crossbow",
        "dark green van reversing now",
        "silver serpent coiled around glowing crystal staff",
        "saxophone mouthpiece in elegant silver",
        "white wooden golf club head",
        "enormous black robot with round body",
        "bow with green handle and yellow bow",
        "wooden flute adorned with colorful beads",
        "glowing staff topped with radiant sapphire stone",# --- end test 

    ]

    optimizer = MultiGeneratorCLIPOptimizer(
        num_episodes=3,  # Start with fewer episodes for testing
        target_score=0.8,  # More achievable target
        max_rounds_per_episode=20,
        enable_router=True,
        tie_break_threshold=0.05
    )
    
    try:
        results = optimizer.run_all_episodes(test_prompts)
        print(f"\n✅ Multi-generator episodic optimization complete!")
        print(f"Results saved to: {optimizer.log_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        optimizer._save_episodic_memory()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 