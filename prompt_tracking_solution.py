#!/usr/bin/env python3
"""
Prompt Tracking Solution for Meta-Learning Systems
==================================================
This solution ensures that both original prompts and optimized prompts are properly 
tracked when meta-learning provides successful alternatives, preventing the loss of 
original prompt data for analysis.
"""

import sqlite3
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class EnhancedTaskRecord:
    """Enhanced task record that tracks both original and optimized prompts"""
    task_id: str
    original_prompt: str  # Always preserved - the prompt from validator
    optimized_prompt: Optional[str] = None  # The prompt actually used for generation
    optimization_applied: bool = False
    optimization_strategy: Optional[str] = None
    optimization_confidence: Optional[float] = None
    prompt_hash: str = ""
    validator_uid: int = 0
    validator_hotkey: str = ""
    validator_stake: float = 0.0
    validation_threshold: float = 0.0
    pulled_at: float = 0.0
    processed_at: Optional[float] = None
    submitted_at: Optional[float] = None
    generation_time: Optional[float] = None
    validation_time: Optional[float] = None
    local_validation_score: Optional[float] = None
    submission_success: bool = False
    feedback_received: bool = False
    task_fidelity_score: Optional[float] = None
    average_fidelity_score: Optional[float] = None
    current_miner_reward: Optional[float] = None
    validation_failed: Optional[bool] = None
    generations_in_window: Optional[int] = None
    ply_file_path: Optional[str] = None
    compressed_file_path: Optional[str] = None

class EnhancedTaskDatabase:
    """Enhanced database that properly tracks original vs optimized prompts"""
    
    def __init__(self, db_path: str = "enhanced_trellis_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced tasks table with separate original and optimized prompt columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_tasks (
                task_id TEXT PRIMARY KEY,
                original_prompt TEXT NOT NULL,
                optimized_prompt TEXT,
                optimization_applied BOOLEAN DEFAULT FALSE,
                optimization_strategy TEXT,
                optimization_confidence REAL,
                prompt_hash TEXT NOT NULL,
                validator_uid INTEGER NOT NULL,
                validator_hotkey TEXT NOT NULL,
                validator_stake REAL NOT NULL,
                validation_threshold REAL NOT NULL,
                pulled_at REAL NOT NULL,
                processed_at REAL,
                submitted_at REAL,
                generation_time REAL,
                validation_time REAL,
                local_validation_score REAL,
                submission_success BOOLEAN DEFAULT FALSE,
                feedback_received BOOLEAN DEFAULT FALSE,
                task_fidelity_score REAL,
                average_fidelity_score REAL,
                current_miner_reward REAL,
                validation_failed BOOLEAN,
                generations_in_window INTEGER,
                ply_file_path TEXT,
                compressed_file_path TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        # Optimization tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                original_prompt TEXT NOT NULL,
                optimized_prompt TEXT NOT NULL,
                strategy_used TEXT NOT NULL,
                confidence_score REAL,
                risk_factors TEXT,  -- JSON array of risk factors
                applied_optimizations TEXT,  -- JSON array of applied strategies
                optimization_time REAL NOT NULL,
                validation_score REAL,
                task_fidelity_score REAL,
                success BOOLEAN DEFAULT FALSE,
                timestamp REAL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (task_id) REFERENCES enhanced_tasks (task_id)
            )
        ''')
        
        # Meta-learning insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_pattern TEXT NOT NULL,
                successful_optimization TEXT NOT NULL,
                pattern_category TEXT,
                success_rate REAL,
                avg_improvement REAL,
                usage_count INTEGER DEFAULT 1,
                last_used REAL,
                confidence REAL,
                timestamp REAL DEFAULT (strftime('%s', 'now'))
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_enhanced_task(self, task: EnhancedTaskRecord):
        """Save enhanced task record with both original and optimized prompts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO enhanced_tasks 
            (task_id, original_prompt, optimized_prompt, optimization_applied, 
             optimization_strategy, optimization_confidence, prompt_hash, validator_uid, 
             validator_hotkey, validator_stake, validation_threshold, pulled_at, 
             processed_at, submitted_at, generation_time, validation_time, 
             local_validation_score, submission_success, feedback_received,
             task_fidelity_score, average_fidelity_score, current_miner_reward,
             validation_failed, generations_in_window, ply_file_path, compressed_file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id, task.original_prompt, task.optimized_prompt, task.optimization_applied,
            task.optimization_strategy, task.optimization_confidence, task.prompt_hash, 
            task.validator_uid, task.validator_hotkey, task.validator_stake, 
            task.validation_threshold, task.pulled_at, task.processed_at, task.submitted_at,
            task.generation_time, task.validation_time, task.local_validation_score,
            task.submission_success, task.feedback_received, task.task_fidelity_score,
            task.average_fidelity_score, task.current_miner_reward, task.validation_failed,
            task.generations_in_window, task.ply_file_path, task.compressed_file_path
        ))
        
        conn.commit()
        conn.close()
    
    def record_optimization(self, task_id: str, original_prompt: str, optimized_prompt: str,
                          strategy: str, confidence: float, risk_factors: list,
                          applied_optimizations: list, optimization_time: float):
        """Record the optimization details separately"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO optimization_history 
            (task_id, original_prompt, optimized_prompt, strategy_used, confidence_score,
             risk_factors, applied_optimizations, optimization_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id, original_prompt, optimized_prompt, strategy, confidence,
            json.dumps(risk_factors), json.dumps(applied_optimizations), optimization_time
        ))
        
        conn.commit()
        conn.close()
    
    def update_optimization_success(self, task_id: str, validation_score: float, 
                                  task_fidelity_score: float, success: bool):
        """Update optimization record with success metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE optimization_history 
            SET validation_score = ?, task_fidelity_score = ?, success = ?
            WHERE task_id = ?
        ''', (validation_score, task_fidelity_score, success, task_id))
        
        conn.commit()
        conn.close()
    
    def get_zero_fidelity_original_prompts(self) -> list:
        """Get all original prompts that received 0 task fidelity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT original_prompt 
            FROM enhanced_tasks 
            WHERE task_fidelity_score = 0.0 
            AND feedback_received = 1
            ORDER BY original_prompt
        ''')
        
        prompts = [row[0] for row in cursor.fetchall()]
        conn.close()
        return prompts
    
    def get_optimization_analysis(self) -> Dict[str, Any]:
        """Analyze optimization effectiveness"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get optimization statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_optimizations,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(task_fidelity_score) as avg_fidelity,
                AVG(validation_score) as avg_validation,
                COUNT(CASE WHEN task_fidelity_score > 0 THEN 1 END) as non_zero_fidelity
            FROM optimization_history 
            WHERE validation_score IS NOT NULL
        ''')
        
        stats = cursor.fetchone()
        
        # Get strategy effectiveness
        cursor.execute('''
            SELECT 
                strategy_used,
                COUNT(*) as usage_count,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(task_fidelity_score) as avg_fidelity
            FROM optimization_history 
            WHERE validation_score IS NOT NULL
            GROUP BY strategy_used
            ORDER BY success_rate DESC
        ''')
        
        strategies = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_optimizations': stats[0],
            'optimization_success_rate': stats[1],
            'avg_fidelity_optimized': stats[2],
            'avg_validation_optimized': stats[3],
            'non_zero_fidelity_count': stats[4],
            'strategy_effectiveness': [
                {
                    'strategy': row[0],
                    'usage_count': row[1],
                    'success_rate': row[2],
                    'avg_fidelity': row[3]
                }
                for row in strategies
            ]
        }

class MetaLearningPromptTracker:
    """Tracks both original and optimized prompts for meta-learning analysis"""
    
    def __init__(self, db: EnhancedTaskDatabase):
        self.db = db
        self.optimization_cache = {}
    
    def track_optimization(self, task_id: str, original_prompt: str, 
                         optimization_result: Dict[str, Any]) -> str:
        """
        Track prompt optimization while preserving original prompt.
        
        Args:
            task_id: Unique task identifier
            original_prompt: The original prompt from validator (NEVER LOST)
            optimization_result: Result from meta-learning optimization
            
        Returns:
            The prompt to use for generation (optimized or original)
        """
        
        # Always preserve the original prompt
        optimized_prompt = optimization_result.get('optimized_prompt', original_prompt)
        strategy = optimization_result.get('strategy_used', 'none')
        confidence = optimization_result.get('confidence', 0.0)
        risk_factors = optimization_result.get('risk_factors', [])
        applied_optimizations = optimization_result.get('applied_strategies', [])
        
        # Determine if optimization should be applied
        should_optimize = (
            optimization_result.get('improvement_expected', False) and
            confidence >= 0.5 and  # Minimum confidence threshold
            optimized_prompt != original_prompt
        )
        
        if should_optimize:
            # Record the optimization
            self.db.record_optimization(
                task_id=task_id,
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt, 
                strategy=strategy,
                confidence=confidence,
                risk_factors=risk_factors,
                applied_optimizations=applied_optimizations,
                optimization_time=time.time()
            )
            
            # Store in cache for later success tracking
            self.optimization_cache[task_id] = {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'optimization_applied': True,
                'strategy': strategy,
                'confidence': confidence
            }
            
            print(f"🔧 Optimization Applied:")
            print(f"   📝 Original: {original_prompt}")
            print(f"   ✨ Optimized: {optimized_prompt}")
            print(f"   🎯 Strategy: {strategy}")
            print(f"   📊 Confidence: {confidence:.3f}")
            
            return optimized_prompt
        else:
            # No optimization applied - use original
            self.optimization_cache[task_id] = {
                'original_prompt': original_prompt,
                'optimized_prompt': original_prompt,
                'optimization_applied': False,
                'strategy': 'none',
                'confidence': 1.0
            }
            
            print(f"✅ Using original prompt (no optimization): {original_prompt}")
            return original_prompt
    
    def track_task_result(self, task_id: str, task_fidelity_score: float, 
                         validation_score: Optional[float] = None):
        """Track the final result and update optimization success metrics"""
        
        if task_id not in self.optimization_cache:
            return
        
        cache_entry = self.optimization_cache[task_id]
        
        # Update optimization success in database
        if cache_entry['optimization_applied']:
            success = task_fidelity_score > 0.0  # Consider non-zero as success
            self.db.update_optimization_success(
                task_id=task_id,
                validation_score=validation_score or 0.0,
                task_fidelity_score=task_fidelity_score,
                success=success
            )
            
            if success:
                print(f"🎉 Optimization SUCCESS! Task {task_id} scored {task_fidelity_score:.3f}")
            else:
                print(f"⚠️ Optimization didn't prevent zero fidelity for task {task_id}")
        
        # Clean up cache
        del self.optimization_cache[task_id]
    
    def analyze_original_vs_optimized_performance(self) -> Dict[str, Any]:
        """Analyze how optimization affects the original prompt failure patterns"""
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get original prompts that failed vs succeeded with optimization
        cursor.execute('''
            SELECT 
                e.original_prompt,
                e.optimization_applied,
                e.task_fidelity_score,
                o.strategy_used,
                o.confidence_score
            FROM enhanced_tasks e
            LEFT JOIN optimization_history o ON e.task_id = o.task_id
            WHERE e.feedback_received = 1
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # Analyze patterns
        original_failures = {}
        optimization_rescues = {}
        
        for row in results:
            original_prompt, optimization_applied, fidelity_score, strategy, confidence = row
            
            # Track original prompt performance
            if original_prompt not in original_failures:
                original_failures[original_prompt] = {'total': 0, 'failures': 0, 'optimized_saves': 0}
            
            original_failures[original_prompt]['total'] += 1
            
            if fidelity_score == 0.0:
                original_failures[original_prompt]['failures'] += 1
            elif optimization_applied and fidelity_score > 0.0:
                original_failures[original_prompt]['optimized_saves'] += 1
                optimization_rescues[original_prompt] = {
                    'strategy': strategy,
                    'confidence': confidence,
                    'fidelity_score': fidelity_score
                }
        
        return {
            'original_failure_patterns': original_failures,
            'optimization_rescues': optimization_rescues,
            'rescue_rate': len(optimization_rescues) / len(original_failures) if original_failures else 0
        }

def integrate_with_continuous_orchestrator():
    """
    Integration guide for the continuous trellis orchestrator.
    
    STEP 1: Replace the TaskRecord with EnhancedTaskRecord
    STEP 2: Replace the database with EnhancedTaskDatabase  
    STEP 3: Update the optimize_prompt_for_generation method
    """
    
    integration_code = '''
# In continuous_trellis_orchestrator.py

# STEP 1: Import the enhanced components
from prompt_tracking_solution import EnhancedTaskRecord, EnhancedTaskDatabase, MetaLearningPromptTracker

class ContinuousTrellisOrchestrator:
    def __init__(self, config_path: str = "config.json"):
        # ... existing initialization ...
        
        # STEP 2: Replace database with enhanced version
        self.db = EnhancedTaskDatabase()
        self.prompt_tracker = MetaLearningPromptTracker(self.db)
    
    def optimize_prompt_for_generation(self, task: EnhancedTaskRecord) -> str:
        """Enhanced version that preserves original prompt"""
        try:
            if not self.config.get('enable_prompt_optimization', True):
                return task.original_prompt
            
            # Get optimization recommendation
            optimization_result = self.prompt_optimizer.optimize_prompt(
                task.original_prompt, 
                aggressive=self.config.get('optimization_aggressive_mode', False)
            )
            
            # Track optimization while preserving original
            final_prompt = self.prompt_tracker.track_optimization(
                task_id=task.task_id,
                original_prompt=task.original_prompt,  # ALWAYS PRESERVED
                optimization_result=optimization_result
            )
            
            # Update task record
            task.optimized_prompt = final_prompt
            task.optimization_applied = (final_prompt != task.original_prompt)
            task.optimization_strategy = optimization_result.get('strategy_used', 'none')
            task.optimization_confidence = optimization_result.get('confidence', 0.0)
            
            return final_prompt
            
        except Exception as e:
            self.logger.error(f"❌ Prompt optimization failed: {e}")
            return task.original_prompt  # FALLBACK TO ORIGINAL
    
    async def submit_result(self, task: EnhancedTaskRecord, generation_result: Dict[str, Any]) -> bool:
        """Enhanced submission that tracks optimization success"""
        # ... existing submission logic ...
        
        if response and hasattr(response, 'feedback') and response.feedback:
            # ... existing feedback processing ...
            
            # STEP 3: Track optimization results
            self.prompt_tracker.track_task_result(
                task_id=task.task_id,
                task_fidelity_score=task.task_fidelity_score,
                validation_score=task.local_validation_score
            )
        
        # Save enhanced task record
        self.db.save_enhanced_task(task)
        return success
'''
    
    return integration_code

def main():
    """Demonstration of the enhanced tracking system"""
    
    print("🎯 ENHANCED PROMPT TRACKING SYSTEM")
    print("=" * 60)
    
    # Initialize enhanced database
    db = EnhancedTaskDatabase()
    tracker = MetaLearningPromptTracker(db)
    
    # Example: Track optimization for a failing prompt
    task_id = "test-task-123"
    original_prompt = "glass jug filled juice"  # Known to cause zero fidelity
    
    # Simulate meta-learning optimization
    optimization_result = {
        'optimized_prompt': "wbgmsst, ceramic container filled with orange granules, 3D object, white background",
        'strategy_used': 'material_replacement',
        'confidence': 0.85,
        'improvement_expected': True,
        'risk_factors': ['transparent_material', 'liquid_content'],
        'applied_strategies': ['material_replacement', 'liquid_to_solid']
    }
    
    # Track the optimization
    final_prompt = tracker.track_optimization(task_id, original_prompt, optimization_result)
    
    print(f"\n📊 TRACKING RESULT:")
    print(f"   Original (PRESERVED): {original_prompt}")
    print(f"   Final (USED): {final_prompt}")
    
    # Simulate successful result
    tracker.track_task_result(task_id, task_fidelity_score=0.75, validation_score=0.8)
    
    # Analyze results
    analysis = tracker.analyze_original_vs_optimized_performance()
    print(f"\n📈 ANALYSIS:")
    print(f"   Rescue rate: {analysis['rescue_rate']:.2%}")
    
    print(f"\n✅ Integration code available via integrate_with_continuous_orchestrator()")

if __name__ == "__main__":
    main() 