# Phase 2: Parallel RL Execution - Design & Implementation

## 🎯 Overview

Phase 2 implements the core parallel RL execution engine, where each GPU independently runs episodic RL optimization loops with cross-GPU learning and real-time strategy sharing. This phase builds directly on Phase 1's job distribution foundation.

## 🔄 Parallel RL Execution Flow

Each GPU Agent follows this autonomous optimization loop while participating in distributed learning:

### **Per-GPU Independent Processing:**
1. **Episodic Memory Loading**: Load prompt-specific historical data and cross-GPU insights
2. **RL Strategy Selection**: Choose exploration vs exploitation strategies based on global knowledge
3. **Optimization Loops**: Generate and refine prompts iteratively with local TRELLIS validation
4. **Local Memory Updates**: Update episodic memory with results and strategy effectiveness
5. **Cross-GPU Sharing**: Share successful strategies and insights with coordinator

### **Coordination Points:**
- **Strategy Insights**: Real-time sharing of successful strategies (score > 0.8)
- **Batch Synchronization**: Memory sync and strategy aggregation at batch completion
- **Failure Recovery**: Automatic work redistribution on GPU failures

## 🏗️ Core Implementation Components

### 1. Enhanced RL Agent (`src/gpu_agent/distributed_rl_agent.py`)

```python
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import existing RL components
from smart_prompt_optimizer_v5_rl_loop import RLLoopAgent, OptimizationAttempt
from episodic_trellis_optimizer import EpisodicTrellisOptimizer
from subnet_accurate_validator_a6000ada_slow import validate_with_production_logic

@dataclass
class DistributedRLContext:
    """Context for distributed RL processing on a single GPU"""
    gpu_id: int
    job_id: str
    batch_id: str
    prompts: List[str]
    target_score: float
    max_episodes: int
    max_rounds_per_episode: int
    
    # Memory context
    episodic_memories: Dict[str, Any]
    cross_gpu_insights: List[Dict[str, Any]]
    strategy_recommendations: Dict[str, float]
    
    # Processing state
    current_prompt_idx: int = 0
    current_episode: int = 0
    current_round: int = 0
    best_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.best_scores is None:
            self.best_scores = {prompt: 0.0 for prompt in self.prompts}

@dataclass
class StrategyInsight:
    """Cross-GPU strategy insight for sharing"""
    gpu_id: int
    prompt: str
    strategy: str
    score_achieved: float
    improvement_delta: float
    prompt_complexity: float
    episode_number: int
    timestamp: datetime
    confidence: float = 1.0

class DistributedRLAgent(RLLoopAgent):
    """Enhanced RL Agent with distributed processing capabilities"""
    
    def __init__(self, 
                 gpu_id: int,
                 coordinator_url: str,
                 trellis_server_port: int,
                 **kwargs):
        
        # Initialize parent RLLoopAgent with existing capabilities
        super().__init__(**kwargs)
        
        # Distributed-specific configuration
        self.gpu_id = gpu_id
        self.coordinator_url = coordinator_url
        self.trellis_server_port = trellis_server_port
        
        # Override TRELLIS server to use local GPU instance
        self.trellis_server_url = f"http://localhost:{trellis_server_port}"
        
        # Local processing state
        self.current_context: Optional[DistributedRLContext] = None
        self.local_memory_cache: Dict[str, Any] = {}
        self.strategy_insights_buffer: List[StrategyInsight] = []
        
        # Communication client
        self.coordinator_client = self._setup_coordinator_client()
        
        # Performance tracking
        self.processing_stats = {
            'prompts_processed': 0,
            'total_episodes': 0,
            'successful_optimizations': 0,
            'strategies_shared': 0,
            'insights_received': 0
        }
        
        # Local TRELLIS validator (integrated from existing system)
        self.local_validator = self._setup_local_validator()
        
        # Enhanced episodic optimizer (extends existing)
        self.episodic_optimizer = EpisodicTrellisOptimizer(
            gpu_id=gpu_id,
            distributed_mode=True
        )
        
        logger.info(f"DistributedRLAgent initialized for GPU {gpu_id}")
    
    async def process_batch(self, context: DistributedRLContext) -> Dict[str, Any]:
        """Main entry point: Process a batch of prompts with distributed RL"""
        
        logger.info(f"GPU {self.gpu_id}: Starting batch processing")
        logger.info(f"  Job: {context.job_id}, Batch: {context.batch_id}")
        logger.info(f"  Prompts: {len(context.prompts)}, Target: {context.target_score}")
        
        self.current_context = context
        batch_start_time = time.time()
        
        # Initialize local memory cache
        await self._load_batch_memory(context)
        
        # Process each prompt with episodic RL
        batch_results = []
        for i, prompt in enumerate(context.prompts):
            context.current_prompt_idx = i
            
            logger.info(f"GPU {self.gpu_id}: Processing prompt {i+1}/{len(context.prompts)}")
            
            # Run episodic optimization for this prompt
            prompt_result = await self._process_single_prompt_episodic(prompt, context)
            batch_results.append(prompt_result)
            
            # Send progress update to coordinator
            await self._send_progress_update(i + 1, len(context.prompts))
            
            # Share insights if score is good
            if prompt_result['final_score'] > 0.8:
                await self._share_strategy_insight(prompt, prompt_result, context)
        
        # Calculate batch completion time
        processing_time = time.time() - batch_start_time
        
        # Prepare batch results
        final_results = {
            'job_id': context.job_id,
            'batch_id': context.batch_id,
            'gpu_id': self.gpu_id,
            'prompts_processed': len(batch_results),
            'processing_time_minutes': processing_time / 60.0,
            'average_score': sum(r['final_score'] for r in batch_results) / len(batch_results),
            'best_score': max(r['final_score'] for r in batch_results),
            'total_episodes': sum(r['episodes_run'] for r in batch_results),
            'successful_optimizations': sum(1 for r in batch_results if r['final_score'] > context.target_score),
            'results': batch_results,
            'processing_stats': self.processing_stats.copy()
        }
        
        # Sync with coordinator at batch completion
        await self._sync_batch_completion(final_results)
        
        # Update local statistics
        self.processing_stats['prompts_processed'] += len(batch_results)
        
        logger.info(f"GPU {self.gpu_id}: Batch completed in {processing_time/60:.1f} minutes")
        logger.info(f"  Average score: {final_results['average_score']:.4f}")
        logger.info(f"  Success rate: {final_results['successful_optimizations']}/{len(batch_results)}")
        
        return final_results
    
    async def _process_single_prompt_episodic(self, prompt: str, context: DistributedRLContext) -> Dict[str, Any]:
        """Process single prompt with episodic RL optimization"""
        
        prompt_start_time = time.time()
        
        # Load episodic memory for this prompt
        prompt_memory = self.local_memory_cache.get(prompt, {})
        
        # Initialize tracking
        best_overall_score = prompt_memory.get('best_score', 0.0)
        best_overall_prompt = prompt_memory.get('best_prompt', prompt)
        episode_results = []
        
        # Run multiple episodes for this prompt
        for episode in range(context.max_episodes):
            context.current_episode = episode
            
            logger.debug(f"GPU {self.gpu_id}: Episode {episode+1}/{context.max_episodes} for prompt")
            
            # Build enhanced context with cross-GPU insights
            enhanced_context = await self._build_enhanced_context(prompt, context, episode)
            
            # Select strategy based on episode and insights
            strategy = self._select_episode_strategy(prompt, episode, context)
            
            # Run RL optimization for this episode
            episode_result = await self._run_episode_optimization(
                prompt=best_overall_prompt,  # Start from best so far
                strategy=strategy,
                context=enhanced_context,
                max_rounds=context.max_rounds_per_episode
            )
            
            episode_results.append(episode_result)
            
            # Update best if improved
            if episode_result['final_score'] > best_overall_score:
                best_overall_score = episode_result['final_score']
                best_overall_prompt = episode_result['final_prompt']
                
                logger.info(f"GPU {self.gpu_id}: New best score {best_overall_score:.4f}")
            
            # Check if target achieved
            if best_overall_score >= context.target_score:
                logger.info(f"GPU {self.gpu_id}: Target achieved after {episode+1} episodes")
                break
            
            # Early stopping check
            if episode >= 2:
                recent_scores = [er['final_score'] for er in episode_results[-3:]]
                if len(recent_scores) >= 3 and max(recent_scores) - min(recent_scores) < 0.02:
                    logger.info(f"GPU {self.gpu_id}: Early stopping due to score plateau")
                    break
        
        # Calculate final results
        processing_time = time.time() - prompt_start_time
        
        final_result = {
            'original_prompt': prompt,
            'final_prompt': best_overall_prompt,
            'final_score': best_overall_score,
            'episodes_run': len(episode_results),
            'total_rounds': sum(er.get('rounds_run', 0) for er in episode_results),
            'processing_time_minutes': processing_time / 60.0,
            'score_progression': [er['final_score'] for er in episode_results],
            'strategy_sequence': [er.get('strategy', 'unknown') for er in episode_results],
            'improvement_delta': best_overall_score - prompt_memory.get('best_score', 0.0),
            'target_achieved': best_overall_score >= context.target_score,
            'gpu_id': self.gpu_id
        }
        
        # Update local memory cache
        await self._update_local_memory(prompt, final_result)
        
        return final_result
    
    async def _run_episode_optimization(self, 
                                       prompt: str,
                                       strategy: str, 
                                       context: str,
                                       max_rounds: int) -> Dict[str, Any]:
        """Run RL optimization for a single episode"""
        
        # Use existing RLLoopAgent optimization with enhancements
        episode_start_time = time.time()
        
        try:
            # Call parent class optimization method with local TRELLIS
            optimization_result = self.optimize_with_rl_loop(
                prompt=prompt,
                use_validation=True,
                prompt_with_context=context,
                max_rounds=max_rounds,
                target_score=0.8,  # Episode-level target
                trellis_server_url_w_port=self.trellis_server_url
            )
            
            # Enhance with distributed-specific data
            optimization_result.update({
                'strategy': strategy,
                'processing_time': time.time() - episode_start_time,
                'gpu_id': self.gpu_id,
                'local_validation': True
            })
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Episode optimization failed: {e}")
            return {
                'final_score': 0.0,
                'final_prompt': prompt,
                'strategy': strategy,
                'error': str(e),
                'processing_time': time.time() - episode_start_time,
                'gpu_id': self.gpu_id
            }
    
    async def _build_enhanced_context(self, prompt: str, context: DistributedRLContext, episode: int) -> str:
        """Build enhanced context with episodic memory and cross-GPU insights"""
        
        context_parts = []
        
        # Add episodic memory context
        prompt_memory = self.local_memory_cache.get(prompt, {})
        if prompt_memory:
            context_parts.append(f"""
EPISODIC MEMORY:
- Best score achieved: {prompt_memory.get('best_score', 0):.4f}
- Best prompt: "{prompt_memory.get('best_prompt', prompt)}"
- Episodes completed: {prompt_memory.get('episodes_run', 0)}
- Successful strategies: {prompt_memory.get('successful_strategies', [])}
""")
        
        # Add cross-GPU insights
        relevant_insights = self._get_relevant_insights(prompt)
        if relevant_insights:
            context_parts.append("CROSS-GPU INSIGHTS:")
            for insight in relevant_insights[:3]:  # Top 3 insights
                context_parts.append(f"- GPU {insight['gpu_id']}: {insight['strategy']} → {insight['score']:.4f}")
        
        # Add strategy recommendations
        if context.strategy_recommendations:
            best_strategies = sorted(context.strategy_recommendations.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            context_parts.append("RECOMMENDED STRATEGIES:")
            for strategy, confidence in best_strategies:
                context_parts.append(f"- {strategy}: {confidence:.2f} confidence")
        
        # Add episode context
        context_parts.append(f"""
CURRENT CONTEXT:
- GPU: {self.gpu_id}
- Episode: {episode + 1}/{context.max_episodes}
- Target Score: {context.target_score}
- Job: {context.job_id}
""")
        
        return "\n".join(context_parts)
    
    def _select_episode_strategy(self, prompt: str, episode: int, context: DistributedRLContext) -> str:
        """Select optimization strategy for current episode"""
        
        # Strategy selection logic based on episode and insights
        strategies = ['creative_expansion', 'detail_enhancement', 'style_optimization', 'technical_precision']
        
        # Episode 0: Use recommended strategy or default
        if episode == 0:
            if context.strategy_recommendations:
                return max(context.strategy_recommendations.keys(), 
                          key=lambda k: context.strategy_recommendations[k])
            return 'creative_expansion'
        
        # Episode 1+: Explore based on previous results and cross-GPU insights
        relevant_insights = self._get_relevant_insights(prompt)
        if relevant_insights:
            # Use strategy from best performing insight
            best_insight = max(relevant_insights, key=lambda x: x['score'])
            return best_insight['strategy']
        
        # Fallback: Cycle through strategies
        return strategies[episode % len(strategies)]
    
    def _get_relevant_insights(self, prompt: str) -> List[Dict[str, Any]]:
        """Get relevant cross-GPU insights for current prompt"""
        
        # Simple relevance matching (can be enhanced with ML)
        prompt_words = set(prompt.lower().split())
        relevant = []
        
        for insight in self.strategy_insights_buffer[-20:]:  # Recent insights
            insight_prompt = insight.get('prompt', '').lower()
            insight_words = set(insight_prompt.split())
            
            # Check word overlap
            overlap = len(prompt_words & insight_words) / len(prompt_words | insight_words)
            if overlap > 0.3:  # 30% overlap threshold
                relevant.append(insight)
        
        # Sort by score and recency
        return sorted(relevant, key=lambda x: (x['score'], x['timestamp']), reverse=True)
    
    async def _share_strategy_insight(self, prompt: str, result: Dict[str, Any], context: DistributedRLContext):
        """Share successful strategy insight with coordinator"""
        
        insight = StrategyInsight(
            gpu_id=self.gpu_id,
            prompt=prompt,
            strategy=result.get('strategy', 'unknown'),
            score_achieved=result['final_score'],
            improvement_delta=result.get('improvement_delta', 0),
            prompt_complexity=self._estimate_prompt_complexity(prompt),
            episode_number=context.current_episode,
            timestamp=datetime.now(),
            confidence=min(1.0, result['final_score'])
        )
        
        try:
            # Send to coordinator for distribution
            await self.coordinator_client.post(
                "/gpu_insight",
                json={
                    'gpu_id': self.gpu_id,
                    'insight': insight.__dict__
                }
            )
            
            self.processing_stats['strategies_shared'] += 1
            logger.debug(f"GPU {self.gpu_id}: Shared strategy insight - {insight.strategy} → {insight.score_achieved:.4f}")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to share insight: {e}")
    
    async def receive_cross_gpu_insight(self, insight: Dict[str, Any]):
        """Receive insight from another GPU via coordinator"""
        
        # Add to local insights buffer
        self.strategy_insights_buffer.append(insight)
        
        # Keep buffer manageable
        if len(self.strategy_insights_buffer) > 100:
            self.strategy_insights_buffer = self.strategy_insights_buffer[-50:]
        
        self.processing_stats['insights_received'] += 1
        
        logger.debug(f"GPU {self.gpu_id}: Received insight from GPU {insight.get('gpu_id')}")
    
    async def _load_batch_memory(self, context: DistributedRLContext):
        """Load episodic memories and cross-GPU insights for batch"""
        
        # Load from coordinator's memory system
        for prompt in context.prompts:
            if prompt in context.episodic_memories:
                self.local_memory_cache[prompt] = context.episodic_memories[prompt]
        
        # Load cross-GPU insights
        self.strategy_insights_buffer.extend(context.cross_gpu_insights)
        
        logger.info(f"GPU {self.gpu_id}: Loaded {len(context.episodic_memories)} memories, "
                   f"{len(context.cross_gpu_insights)} insights")
    
    async def _update_local_memory(self, prompt: str, result: Dict[str, Any]):
        """Update local memory cache with new results"""
        
        if prompt not in self.local_memory_cache:
            self.local_memory_cache[prompt] = {}
        
        memory = self.local_memory_cache[prompt]
        
        # Update best results
        if result['final_score'] > memory.get('best_score', 0):
            memory['best_score'] = result['final_score']
            memory['best_prompt'] = result['final_prompt']
        
        # Update statistics
        memory['episodes_run'] = memory.get('episodes_run', 0) + result['episodes_run']
        memory['total_attempts'] = memory.get('total_attempts', 0) + result['total_rounds']
        
        # Track successful strategies
        if result['final_score'] > 0.8:
            successful_strategies = memory.get('successful_strategies', [])
            strategy = result.get('strategy', 'unknown')
            if strategy not in successful_strategies:
                successful_strategies.append(strategy)
            memory['successful_strategies'] = successful_strategies
    
    def _estimate_prompt_complexity(self, prompt: str) -> float:
        """Simple prompt complexity estimation"""
        
        # Word count factor
        word_count = len(prompt.split())
        length_factor = min(1.0, word_count / 20.0)
        
        # Content complexity indicators
        complex_words = ['detailed', 'intricate', 'photorealistic', 'hyperrealistic', 'complex']
        complexity_factor = sum(1 for word in complex_words if word in prompt.lower()) / len(complex_words)
        
        return (length_factor + complexity_factor) / 2.0
    
    def _setup_coordinator_client(self):
        """Setup HTTP client for coordinator communication"""
        import aiohttp
        return aiohttp.ClientSession(
            base_url=self.coordinator_url,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    def _setup_local_validator(self):
        """Setup local TRELLIS validation (integrated from existing system)"""
        # This would integrate with your existing validation logic
        return None
    
    async def _send_progress_update(self, completed: int, total: int):
        """Send progress update to coordinator"""
        
        update = {
            'gpu_id': self.gpu_id,
            'job_id': self.current_context.job_id,
            'completed': completed,
            'total': total,
            'percentage': (completed / total) * 100
        }
        
        try:
            await self.coordinator_client.post("/gpu_progress", json=update)
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to send progress update: {e}")
    
    async def _sync_batch_completion(self, results: Dict[str, Any]):
        """Sync with coordinator on batch completion"""
        
        try:
            # Send final results
            await self.coordinator_client.post("/batch_complete", json=results)
            
            # Request memory sync
            await self.coordinator_client.post("/request_memory_sync", json={
                'gpu_id': self.gpu_id,
                'local_cache': self.local_memory_cache
            })
            
            logger.info(f"GPU {self.gpu_id}: Batch completion sync successful")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to sync batch completion: {e}")
```

### 2. Local TRELLIS Integration (`src/gpu_agent/local_trellis_validator.py`)

```python
"""
Local TRELLIS Validation Integration
Integrates existing TRELLIS validation with distributed RL processing
"""

import asyncio
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Import existing validation components
from subnet_accurate_validator_a6000ada_slow import (
    generate_and_get_ply_data,
    validate_with_production_logic
)

class LocalTrellisValidator:
    """Local TRELLIS validation integrated with GPU RL agent"""
    
    def __init__(self, gpu_id: int, trellis_server_port: int):
        self.gpu_id = gpu_id
        self.trellis_server_port = trellis_server_port
        self.trellis_server_url = f"http://localhost:{trellis_server_port}"
        
        # Validation configuration
        self.config = {
            'use_production_validation': True,
            'local_validation_threshold': 0.7,
            'generation_timeout': 120,  # 2 minutes
            'validation_timeout': 60    # 1 minute
        }
        
        logger.info(f"LocalTrellisValidator initialized for GPU {gpu_id}")
    
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Generate and validate 3D model for prompt"""
        
        validation_start = time.time()
        
        try:
            # Step 1: Generate PLY data using local TRELLIS server
            ply_result = await self._generate_ply_local(prompt)
            
            if not ply_result['success']:
                return {
                    'success': False,
                    'score': 0.0,
                    'error': ply_result.get('error', 'Generation failed'),
                    'processing_time': time.time() - validation_start
                }
            
            # Step 2: Validate with production logic
            validation_score = await self._validate_ply_production(
                prompt, 
                ply_result['ply_data']
            )
            
            processing_time = time.time() - validation_start
            
            return {
                'success': True,
                'score': validation_score,
                'ply_data_size': len(ply_result['ply_data']),
                'processing_time': processing_time,
                'gpu_id': self.gpu_id
            }
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Validation failed for prompt: {e}")
            return {
                'success': False,
                'score': 0.0,
                'error': str(e),
                'processing_time': time.time() - validation_start
            }
    
    async def _generate_ply_local(self, prompt: str) -> Dict[str, Any]:
        """Generate PLY data using local TRELLIS server"""
        
        try:
            # Use existing generation logic with local server
            ply_data = await asyncio.wait_for(
                generate_and_get_ply_data(prompt, self.trellis_server_url),
                timeout=self.config['generation_timeout']
            )
            
            if ply_data and len(ply_data) > 1000:  # Minimum size check
                return {
                    'success': True,
                    'ply_data': ply_data,
                    'size_bytes': len(ply_data)
                }
            else:
                return {
                    'success': False,
                    'error': 'Generated PLY data too small or empty'
                }
                
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f'Generation timeout after {self.config["generation_timeout"]}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Generation error: {str(e)}'
            }
    
    async def _validate_ply_production(self, prompt: str, ply_data: bytes) -> float:
        """Validate PLY data using production validation logic"""
        
        try:
            # Use existing production validation
            validation_score = await asyncio.wait_for(
                validate_with_production_logic(prompt, ply_data),
                timeout=self.config['validation_timeout']
            )
            
            return max(0.0, min(1.0, validation_score))  # Clamp to [0,1]
            
        except asyncio.TimeoutError:
            logger.warning(f"GPU {self.gpu_id}: Validation timeout for prompt")
            return 0.0
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Validation error: {e}")
            return 0.0
```

### 3. GPU Server Management (`src/gpu_agent/gpu_server_manager.py`)

```python
"""
GPU Server Management for Distributed RL
Manages local TRELLIS server instances per GPU
"""

import asyncio
import subprocess
import time
from typing import Dict, Any, Optional
from pathlib import Path

class GPUServerManager:
    """Manages TRELLIS server for a specific GPU"""
    
    def __init__(self, gpu_id: int, base_port: int = 8096):
        self.gpu_id = gpu_id
        self.port = base_port + gpu_id
        self.server_process: Optional[subprocess.Popen] = None
        self.server_ready = False
        
        # Server configuration
        self.config = {
            'startup_timeout': 120,  # 2 minutes to start
            'health_check_interval': 30,  # 30 seconds
            'restart_attempts': 3
        }
        
        logger.info(f"GPUServerManager initialized for GPU {gpu_id} on port {self.port}")
    
    async def start_server(self) -> bool:
        """Start TRELLIS server for this GPU"""
        
        if self.server_process and self.server_process.poll() is None:
            logger.info(f"GPU {self.gpu_id}: Server already running")
            return True
        
        logger.info(f"GPU {self.gpu_id}: Starting TRELLIS server on port {self.port}")
        
        try:
            # Use existing server startup logic
            cmd = [
                "python", "-m", "trellis_server",
                "--gpu", str(self.gpu_id),
                "--port", str(self.port),
                "--batch-size", "1",  # Process one at a time for RL
                "--timeout", "120"
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"CUDA_VISIBLE_DEVICES": str(self.gpu_id)}
            )
            
            # Wait for server to be ready
            server_ready = await self._wait_for_server_ready()
            
            if server_ready:
                self.server_ready = True
                logger.info(f"GPU {self.gpu_id}: Server started successfully")
                
                # Start health monitoring
                asyncio.create_task(self._health_monitor_loop())
                
                return True
            else:
                logger.error(f"GPU {self.gpu_id}: Server failed to start")
                await self.stop_server()
                return False
                
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Error starting server: {e}")
            return False
    
    async def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready to accept requests"""
        
        import aiohttp
        
        start_time = time.time()
        
        while time.time() - start_time < self.config['startup_timeout']:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/health") as response:
                        if response.status == 200:
                            return True
            except:
                pass
            
            await asyncio.sleep(2)
        
        return False
    
    async def _health_monitor_loop(self):
        """Monitor server health and restart if needed"""
        
        import aiohttp
        
        while self.server_ready:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                # Check if process is still running
                if self.server_process and self.server_process.poll() is not None:
                    logger.warning(f"GPU {self.gpu_id}: Server process died, restarting...")
                    await self._restart_server()
                    continue
                
                # Check server responsiveness
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/health", timeout=10) as response:
                        if response.status != 200:
                            logger.warning(f"GPU {self.gpu_id}: Server health check failed")
                            await self._restart_server()
                
            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Health check error: {e}")
                await self._restart_server()
    
    async def _restart_server(self):
        """Restart the TRELLIS server"""
        
        logger.info(f"GPU {self.gpu_id}: Restarting server...")
        
        await self.stop_server()
        await asyncio.sleep(5)  # Brief pause
        
        success = await self.start_server()
        if not success:
            logger.error(f"GPU {self.gpu_id}: Server restart failed")
            self.server_ready = False
    
    async def stop_server(self):
        """Stop the TRELLIS server"""
        
        if self.server_process:
            logger.info(f"GPU {self.gpu_id}: Stopping server...")
            
            self.server_process.terminate()
            
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.server_process.wait),
                    timeout=10
                )
            except asyncio.TimeoutError:
                logger.warning(f"GPU {self.gpu_id}: Force killing server")
                self.server_process.kill()
                await asyncio.to_thread(self.server_process.wait)
            
            self.server_process = None
        
        self.server_ready = False
        logger.info(f"GPU {self.gpu_id}: Server stopped")
```

## 🚀 Phase 2 Integration Points

### **1. Coordinator Enhancements**

The existing Phase 1 coordinator needs these additions:

```python
# In distributed_rl_coordinator.py

async def distribute_batch_with_rl_context(self, job: JobRequest, assignments: Dict[int, List[str]]):
    """Enhanced batch distribution with RL context"""
    
    for gpu_id, prompts in assignments.items():
        # Load memory context for this batch
        memory_context = await self.global_memory.load_batch_memory(gpu_id, f"batch_{job.job_id}", prompts)
        
        # Create distributed RL context
        rl_context = DistributedRLContext(
            gpu_id=gpu_id,
            job_id=job.job_id,
            batch_id=f"batch_{gpu_id}_{job.job_id}",
            prompts=prompts,
            target_score=job.target_score,
            max_episodes=job.max_episodes,
            max_rounds_per_episode=job.max_rounds_per_episode,
            episodic_memories=memory_context.prompt_memories,
            cross_gpu_insights=memory_context.relevant_insights,
            strategy_recommendations=memory_context.recommended_strategies
        )
        
        # Send to GPU agent
        await self._send_rl_batch_to_gpu(gpu_id, rl_context)

async def handle_cross_gpu_insights(self, gpu_id: int, insight: Dict[str, Any]):
    """Handle cross-GPU insight sharing"""
    
    # Store insight globally
    await self.global_memory.add_cross_gpu_insight(insight)
    
    # Broadcast to other active GPUs
    for other_gpu_id, state in self.gpu_states.items():
        if other_gpu_id != gpu_id and state.status == "busy":
            await self._send_insight_to_gpu(other_gpu_id, insight)
```

### **2. Memory System Enhancements**

The existing episodic memory system gains real-time capabilities:

```python
# In episodic_loader.py

async def stream_cross_gpu_insights(self, gpu_id: int) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream real-time insights to GPU"""
    
    last_seen = datetime.now()
    
    while True:
        # Get new insights since last check
        new_insights = await self.get_insights_since(last_seen)
        
        for insight in new_insights:
            if insight.gpu_id != gpu_id:  # Don't send back to sender
                yield insight.to_dict()
        
        last_seen = datetime.now()
        await asyncio.sleep(5)  # Check every 5 seconds
```

## 🎯 Expected Performance Improvements

### **Phase 2 Performance Targets:**

1. **Parallel Processing**: 6-8x speedup vs sequential
2. **Cross-GPU Learning**: 5-10% score improvement through strategy sharing
3. **Adaptive Optimization**: 15% better strategy selection via episodic memory
4. **Real-time Coordination**: <2 second latency for cross-GPU insights

### **Integration Benefits:**

- **Seamless RL Processing**: Each GPU runs independent RL loops with global learning
- **Real-time Strategy Sharing**: Successful strategies propagate immediately
- **Memory-Driven Optimization**: Historical performance guides strategy selection
- **Failure Resilience**: Failed GPUs don't impact other GPU processing

Phase 2 transforms the system from intelligent job distribution to a true **distributed reinforcement learning engine** where each GPU contributes to collective intelligence while processing independently. The integration with existing TRELLIS validation ensures production-quality results at scale.

**Ready for implementation and integration with your existing RL loop systems!** 🚀




