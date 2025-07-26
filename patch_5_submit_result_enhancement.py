
# MODIFY the submit_result method to call log_optimization_results

# ADD THIS LINE after setting task.task_fidelity_score:
if response and hasattr(response, 'feedback') and response.feedback:
    # ... existing feedback processing ...
    task.task_fidelity_score = feedback.task_fidelity_score
    # ... other assignments ...
    
    # ADD THIS LINE:
    self.log_optimization_results(task)  # Track optimization effectiveness
