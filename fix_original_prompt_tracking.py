#!/usr/bin/env python3
"""
Fix Original Prompt Tracking in Meta-Learning System
===================================================
This script provides a practical fix for the existing continuous_trellis_orchestrator.py
to ensure original prompts are never lost when meta-learning provides successful alternatives.

PROBLEM: Currently, when meta-learning optimizes a prompt and it succeeds, only the optimized
prompt is tracked, losing the original prompt data which is crucial for analysis.

SOLUTION: Track both original and optimized prompts separately.
"""

import sqlite3
import time
from typing import Dict, Any, Optional

class OriginalPromptFix:
    """Fix to ensure original prompts are always preserved"""
    
    def __init__(self):
        self.optimization_log = {}
    
    def create_migration_script(self):
        """Create SQL migration to add original prompt tracking"""
        migration_sql = """
        -- Add columns to existing tasks table to track original vs optimized prompts
        ALTER TABLE tasks ADD COLUMN original_prompt TEXT;
        ALTER TABLE tasks ADD COLUMN optimized_prompt TEXT; 
        ALTER TABLE tasks ADD COLUMN optimization_applied BOOLEAN DEFAULT FALSE;
        ALTER TABLE tasks ADD COLUMN optimization_strategy TEXT;
        ALTER TABLE tasks ADD COLUMN optimization_confidence REAL;
        
        -- Update existing records where original_prompt is null
        UPDATE tasks SET original_prompt = prompt WHERE original_prompt IS NULL;
        
        -- Create optimization tracking table
        CREATE TABLE IF NOT EXISTS prompt_optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            original_prompt TEXT NOT NULL,
            optimized_prompt TEXT NOT NULL,
            strategy_used TEXT,
            confidence_score REAL,
            risk_level TEXT,
            optimization_time REAL,
            task_fidelity_score REAL,
            success BOOLEAN,
            timestamp REAL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        );
        """
        
        with open('migration_add_original_prompt_tracking.sql', 'w') as f:
            f.write(migration_sql)
        
        print("✅ Created migration_add_original_prompt_tracking.sql")
        print("   Run this SQL script on your database to add original prompt tracking")

def patch_continuous_orchestrator():
    """Generate the code patches needed for continuous_trellis_orchestrator.py"""
    
    patches = {
        "1_import_additions": '''
# ADD THESE IMPORTS at the top of continuous_trellis_orchestrator.py
import json
from typing import Tuple
''',
        
        "2_database_enhancement": '''
# REPLACE the existing save_task method in TaskDatabase class
def save_task(self, task: TaskRecord):
    """Enhanced save_task that preserves original prompt information"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Get optimization info if available  
    optimization_info = getattr(task, '_optimization_info', {})
    original_prompt = optimization_info.get('original_prompt', task.prompt)
    optimized_prompt = optimization_info.get('optimized_prompt', task.prompt) 
    optimization_applied = optimization_info.get('applied', False)
    optimization_strategy = optimization_info.get('strategy', None)
    optimization_confidence = optimization_info.get('confidence', None)
    
    cursor.execute(\'\'\'
        INSERT OR REPLACE INTO tasks 
        (task_id, prompt, original_prompt, optimized_prompt, optimization_applied,
         optimization_strategy, optimization_confidence, prompt_hash, validator_uid, 
         validator_hotkey, validator_stake, validation_threshold, pulled_at, 
         processed_at, submitted_at, generation_time, validation_time, 
         local_validation_score, submission_success, feedback_received,
         task_fidelity_score, average_fidelity_score, current_miner_reward,
         validation_failed, generations_in_window, ply_file_path, compressed_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    \'\'\', (
        task.task_id, task.prompt, original_prompt, optimized_prompt, optimization_applied,
        optimization_strategy, optimization_confidence, task.prompt_hash, task.validator_uid,
        task.validator_hotkey, task.validator_stake, task.validation_threshold,
        task.pulled_at, task.processed_at, task.submitted_at, task.generation_time,
        task.validation_time, task.local_validation_score, task.submission_success,
        task.feedback_received, task.task_fidelity_score, task.average_fidelity_score,
        task.current_miner_reward, task.validation_failed, task.generations_in_window,
        task.ply_file_path, task.compressed_file_path
    ))
    
    conn.commit()
    conn.close()
''',
        
        "3_optimize_prompt_method_fix": '''
# REPLACE the optimize_prompt_for_generation method in ContinuousTrellisOrchestrator class
def optimize_prompt_for_generation(self, task: TaskRecord) -> str:
    """FIXED: Optimize prompt while preserving original prompt information"""
    try:
        # CRITICAL: Store the original prompt from validator (NEVER LOSE THIS)
        original_validator_prompt = task.prompt
        
        # Check if optimization is enabled
        if not self.config.get('enable_prompt_optimization', True):
            # Even when disabled, preserve the original
            task._optimization_info = {
                'original_prompt': original_validator_prompt,
                'optimized_prompt': original_validator_prompt,
                'applied': False,
                'strategy': 'none',
                'confidence': 1.0
            }
            return original_validator_prompt
        
        # Analyze and optimize the prompt
        optimization_result = self.prompt_optimizer.optimize_prompt(
            original_validator_prompt,  # Always optimize based on original
            aggressive=self.config.get('optimization_aggressive_mode', False)
        )
        analysis = optimization_result['analysis']
        
        # Log the analysis if enabled
        if self.config.get('log_optimization_details', True):
            self.logger.info(f"🔍 Prompt Analysis for '{original_validator_prompt[:50]}...':")
            self.logger.info(f"   Risk Level: {analysis['risk_level']}")
            
            if analysis['risk_factors']:
                self.logger.info(f"   Risk Factors:")
                for factor in analysis['risk_factors']:
                    self.logger.info(f"     • {factor}")
        
        # Determine if optimization should be applied
        should_apply_optimization = optimization_result.get('improvement_expected', False)
        optimized_prompt = optimization_result.get('optimized_prompt', original_validator_prompt)
        
        if should_apply_optimization and optimized_prompt != original_validator_prompt:
            # Apply optimization but preserve original
            applied_strategies = optimization_result.get('applied_strategies', [])
            
            if self.config.get('log_optimization_details', True):
                self.logger.info(f"🔧 Prompt Optimization Applied:")
                self.logger.info(f"   Original (PRESERVED): {original_validator_prompt}")
                self.logger.info(f"   Optimized (USED): {optimized_prompt}")
                self.logger.info(f"   Strategies: {', '.join(applied_strategies)}")
            else:
                self.logger.info(f"🔧 Optimized prompt (risk: {analysis['risk_level']})")
            
            # Store optimization information for database tracking
            task._optimization_info = {
                'original_prompt': original_validator_prompt,
                'optimized_prompt': optimized_prompt,
                'applied': True,
                'strategy': ', '.join(applied_strategies),
                'confidence': optimization_result.get('confidence', 0.5),
                'risk_level': analysis['risk_level']
            }
            
            # Update statistics
            self.stats['prompts_optimized'] += 1
            self.stats['optimization_improvements'] += 1
            
            return optimized_prompt
        else:
            # No optimization applied - use original
            if self.config.get('log_optimization_details', True):
                self.logger.info(f"✅ Using original prompt (risk: {analysis['risk_level']})")
            
            # Store that no optimization was applied
            task._optimization_info = {
                'original_prompt': original_validator_prompt,
                'optimized_prompt': original_validator_prompt,
                'applied': False,
                'strategy': 'none',
                'confidence': 1.0,
                'risk_level': analysis['risk_level']
            }
            
            self.stats['prompts_optimized'] += 1
            return original_validator_prompt
            
    except Exception as e:
        self.logger.error(f"❌ Prompt optimization failed: {e}")
        # CRITICAL: Always fallback to original prompt
        task._optimization_info = {
            'original_prompt': original_validator_prompt,
            'optimized_prompt': original_validator_prompt,
            'applied': False,
            'strategy': 'error',
            'confidence': 0.0,
            'error': str(e)
        }
        return original_validator_prompt
''',
        
        "4_analysis_methods": '''
# ADD THESE NEW METHODS to ContinuousTrellisOrchestrator class for analysis

def log_optimization_results(self, task: TaskRecord):
    """Log optimization success/failure for analysis"""
    if not hasattr(task, '_optimization_info'):
        return
    
    opt_info = task._optimization_info
    
    # Log to optimization tracking table if it exists
    try:
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute(\'\'\'
            INSERT OR IGNORE INTO prompt_optimizations 
            (task_id, original_prompt, optimized_prompt, strategy_used, 
             confidence_score, risk_level, optimization_time, task_fidelity_score, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        \'\'\', (
            task.task_id,
            opt_info['original_prompt'],
            opt_info['optimized_prompt'], 
            opt_info['strategy'],
            opt_info['confidence'],
            opt_info.get('risk_level', 'unknown'),
            time.time(),
            task.task_fidelity_score or 0.0,
            (task.task_fidelity_score or 0.0) > 0.0
        ))
        
        conn.commit()
        conn.close()
        
    except sqlite3.Error:
        pass  # Table might not exist yet

def analyze_original_prompt_patterns(self):
    """Analyze which original prompts consistently fail despite optimization"""
    try:
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get original prompts that still result in zero fidelity
        cursor.execute(\'\'\'
            SELECT 
                original_prompt,
                COUNT(*) as total_attempts,
                SUM(CASE WHEN task_fidelity_score = 0.0 THEN 1 ELSE 0 END) as zero_fidelity_count,
                AVG(task_fidelity_score) as avg_fidelity,
                optimization_applied
            FROM tasks 
            WHERE feedback_received = 1 AND original_prompt IS NOT NULL
            GROUP BY original_prompt, optimization_applied
            HAVING zero_fidelity_count > 0
            ORDER BY zero_fidelity_count DESC
        \'\'\')
        
        results = cursor.fetchall()
        conn.close()
        
        self.logger.info(f"📊 Original Prompt Analysis:")
        self.logger.info(f"   Found {len(results)} original prompts with zero fidelity issues")
        
        for row in results[:10]:  # Top 10 problematic originals
            original, total, zeros, avg_fidelity, optimized = row
            self.logger.info(f"   '{original[:50]}...' - {zeros}/{total} failures (avg: {avg_fidelity:.3f}) [optimized: {optimized}]")
        
        return results
        
    except Exception as e:
        self.logger.error(f"❌ Analysis failed: {e}")
        return []
''',
        
        "5_submit_result_enhancement": '''
# MODIFY the submit_result method to call log_optimization_results

# ADD THIS LINE after setting task.task_fidelity_score:
if response and hasattr(response, 'feedback') and response.feedback:
    # ... existing feedback processing ...
    task.task_fidelity_score = feedback.task_fidelity_score
    # ... other assignments ...
    
    # ADD THIS LINE:
    self.log_optimization_results(task)  # Track optimization effectiveness
'''
    }
    
    return patches

def create_extraction_script():
    """Create script to extract original prompts from enhanced database"""
    script_content = '''#!/usr/bin/env python3
"""
Extract Original Zero Fidelity Prompts (Enhanced)
=================================================
This script extracts original prompts that received zero fidelity,
distinguishing between prompts that failed with/without optimization.
"""

import sqlite3
import json

def extract_original_zero_fidelity_prompts(db_path="continuous_trellis_tasks.db"):
    """Extract original prompts with zero fidelity, preserving optimization info"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if enhanced columns exist
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [row[1] for row in cursor.fetchall()]
    has_original_column = 'original_prompt' in columns
    
    if has_original_column:
        # Enhanced query with original prompt tracking
        query = """
        SELECT DISTINCT 
            original_prompt,
            prompt as final_prompt,
            optimization_applied,
            optimization_strategy,
            COUNT(*) as failure_count
        FROM tasks 
        WHERE task_fidelity_score = 0.0 
        AND feedback_received = 1
        AND original_prompt IS NOT NULL
        GROUP BY original_prompt, optimization_applied
        ORDER BY original_prompt
        """
    else:
        # Fallback to standard query
        query = """
        SELECT DISTINCT prompt, COUNT(*) as failure_count
        FROM tasks 
        WHERE task_fidelity_score = 0.0 
        AND feedback_received = 1
        GROUP BY prompt
        ORDER BY prompt
        """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    if has_original_column:
        # Enhanced output
        original_prompts = []
        failed_despite_optimization = []
        failed_without_optimization = []
        
        for row in results:
            original, final, optimized, strategy, count = row
            original_prompts.append(original)
            
            if optimized:
                failed_despite_optimization.append({
                    'original_prompt': original,
                    'optimized_prompt': final,
                    'strategy': strategy,
                    'failure_count': count
                })
            else:
                failed_without_optimization.append({
                    'original_prompt': original,
                    'failure_count': count
                })
        
        # Save enhanced results
        with open('original_zero_fidelity_analysis.json', 'w') as f:
            json.dump({
                'unique_original_prompts': list(set(original_prompts)),
                'failed_despite_optimization': failed_despite_optimization,
                'failed_without_optimization': failed_without_optimization,
                'total_unique_originals': len(set(original_prompts)),
                'optimization_failure_count': len(failed_despite_optimization),
                'no_optimization_failure_count': len(failed_without_optimization)
            }, f, indent=2)
        
        print(f"📊 Enhanced Analysis Complete:")
        print(f"   Unique original prompts: {len(set(original_prompts))}")
        print(f"   Failed despite optimization: {len(failed_despite_optimization)}")
        print(f"   Failed without optimization: {len(failed_without_optimization)}")
        
    else:
        # Standard output
        prompts = [row[0] for row in results]
        
        with open('zero_fidelity_prompts_original.txt', 'w') as f:
            f.write("# Original Zero Fidelity Prompts\\n")
            f.write(f"# Total: {len(prompts)}\\n\\n")
            f.write("original_zero_fidelity_prompts = [\\n")
            for i, prompt in enumerate(prompts):
                if i == len(prompts) - 1:
                    f.write(f'    "{prompt}"\\n')
                else:
                    f.write(f'    "{prompt}",\\n')
            f.write("]\\n")
        
        print(f"📊 Standard extraction: {len(prompts)} prompts")
    
    return results

if __name__ == "__main__":
    extract_original_zero_fidelity_prompts()
'''
    
    with open('extract_original_prompts_enhanced.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created extract_original_prompts_enhanced.py")

def main():
    """Main function to generate all the fixes"""
    
    print("🛠️ ORIGINAL PROMPT TRACKING FIX GENERATOR")
    print("=" * 60)
    
    # Create migration script
    fix = OriginalPromptFix()
    fix.create_migration_script()
    
    # Generate code patches
    patches = patch_continuous_orchestrator()
    
    print("\\n📝 CODE PATCHES NEEDED:")
    print("=" * 40)
    
    for patch_name, patch_code in patches.items():
        filename = f"patch_{patch_name}.py"
        with open(filename, 'w') as f:
            f.write(patch_code)
        print(f"✅ Created {filename}")
    
    # Create extraction script
    create_extraction_script()
    
    print("\\n🎯 IMPLEMENTATION STEPS:")
    print("=" * 40)
    print("1. Run migration_add_original_prompt_tracking.sql on your database")
    print("2. Apply patches 1-5 to continuous_trellis_orchestrator.py")
    print("3. Restart your mining system") 
    print("4. Use extract_original_prompts_enhanced.py to analyze results")
    
    print("\\n✅ BENEFITS OF THIS FIX:")
    print("   • Original validator prompts are NEVER lost")
    print("   • Can analyze which originals fail despite optimization")
    print("   • Can track optimization success/failure rates")
    print("   • Enables better meta-learning by preserving ground truth")

if __name__ == "__main__":
    main() 