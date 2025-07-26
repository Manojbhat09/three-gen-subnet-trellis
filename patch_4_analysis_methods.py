
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
        
        cursor.execute('''
            INSERT OR IGNORE INTO prompt_optimizations 
            (task_id, original_prompt, optimized_prompt, strategy_used, 
             confidence_score, risk_level, optimization_time, task_fidelity_score, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
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
        cursor.execute('''
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
        ''')
        
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
