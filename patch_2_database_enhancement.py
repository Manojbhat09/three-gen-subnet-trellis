
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
    
    cursor.execute('''
        INSERT OR REPLACE INTO tasks 
        (task_id, prompt, original_prompt, optimized_prompt, optimization_applied,
         optimization_strategy, optimization_confidence, prompt_hash, validator_uid, 
         validator_hotkey, validator_stake, validation_threshold, pulled_at, 
         processed_at, submitted_at, generation_time, validation_time, 
         local_validation_score, submission_success, feedback_received,
         task_fidelity_score, average_fidelity_score, current_miner_reward,
         validation_failed, generations_in_window, ply_file_path, compressed_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
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
