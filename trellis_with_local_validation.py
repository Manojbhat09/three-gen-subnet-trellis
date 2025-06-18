#!/usr/bin/env python3
"""
Enhanced TRELLIS Orchestrator with Local Validation

This script integrates local task fidelity validation into the TRELLIS
mining pipeline to filter out low-quality submissions before sending
them to validators.
"""

import asyncio
import time
import traceback
from typing import Dict, Any, Optional, Tuple
from local_task_fidelity_validator import LocalTaskFidelityValidator
from continuous_trellis_orchestrator import ContinuousTrellisOrchestrator

class TrellisWithLocalValidation(ContinuousTrellisOrchestrator):
    """Enhanced TRELLIS orchestrator with local validation filtering"""
    
    def __init__(self, config_file: str = "trellis_config.json"):
        super().__init__(config_file)
        
        # Initialize local validator
        self.local_validator = LocalTaskFidelityValidator()
        
        # Local validation settings
        self.local_validation_enabled = self.config.get('local_validation', {}).get('enabled', True)
        self.local_validation_threshold = self.config.get('local_validation', {}).get('threshold', 0.5)
        self.local_validation_min_alignment = self.config.get('local_validation', {}).get('min_alignment', 0.35)
        self.local_validation_timeout = self.config.get('local_validation', {}).get('timeout', 60)
        
        # Statistics
        self.local_validation_stats = {
            'total_validated': 0,
            'passed_local': 0,
            'failed_local': 0,
            'skipped_submissions': 0,
            'validation_time_total': 0.0,
            'score_distribution': {'excellent': 0, 'good': 0, 'low': 0, 'failed': 0}
        }
        
        self.logger.info(f"🔍 Local validation enabled: {self.local_validation_enabled}")
        if self.local_validation_enabled:
            self.logger.info(f"   Threshold: {self.local_validation_threshold:.3f}")
            self.logger.info(f"   Min alignment: {self.local_validation_min_alignment:.3f}")
    
    async def validate_locally_before_submission(self, task: Any, ply_data: bytes) -> Tuple[bool, Optional[Dict]]:
        """Validate PLY data locally before submission to validator"""
        if not self.local_validation_enabled:
            return True, None
        
        try:
            self.logger.info(f"🔍 Running local validation for task {task.task_id}...")
            start_time = time.time()
            
            # Save PLY data temporarily for validation
            temp_ply_path = f"/tmp/temp_validation_{task.task_id}.ply"
            with open(temp_ply_path, 'wb') as f:
                f.write(ply_data)
            
            # Run local validation
            result = self.local_validator.validate_ply_file(
                temp_ply_path, 
                task.prompt, 
                self.local_validation_threshold
            )
            
            validation_time = time.time() - start_time
            
            # Update statistics
            self.local_validation_stats['total_validated'] += 1
            self.local_validation_stats['validation_time_total'] += validation_time
            
            # Grade distribution
            grade_key = result.quality_grade.lower()
            if grade_key in self.local_validation_stats['score_distribution']:
                self.local_validation_stats['score_distribution'][grade_key] += 1
            
            # Decision logic
            should_submit = self._should_submit_based_on_local_validation(result)
            
            if should_submit:
                self.local_validation_stats['passed_local'] += 1
                self.logger.info(f"✅ Local validation passed ({validation_time:.2f}s)")
                self.logger.info(f"   Score: {result.final_score:.4f}, Grade: {result.quality_grade}")
            else:
                self.local_validation_stats['failed_local'] += 1
                self.local_validation_stats['skipped_submissions'] += 1
                self.logger.warning(f"❌ Local validation failed ({validation_time:.2f}s)")
                self.logger.warning(f"   Score: {result.final_score:.4f}, Reason: {result.recommendation}")
            
            # Clean up temp file
            try:
                import os
                os.remove(temp_ply_path)
            except:
                pass
            
            return should_submit, {
                'local_validation_result': result,
                'validation_time': validation_time,
                'should_submit': should_submit
            }
            
        except Exception as e:
            self.logger.error(f"❌ Local validation error: {e}")
            # On error, allow submission (fail open)
            return True, {'error': str(e)}
    
    def _should_submit_based_on_local_validation(self, result) -> bool:
        """Determine if model should be submitted based on local validation"""
        # Hard requirements
        if result.final_score < self.local_validation_threshold:
            return False
        
        if result.alignment_score < self.local_validation_min_alignment:
            return False
        
        # Additional quality checks
        if result.gaussian_count and result.gaussian_count < 7000:
            return False
        
        # Soft requirements (warnings but still submit)
        if result.quality_score < 0.6:
            self.logger.warning(f"⚠️ Quality score low ({result.quality_score:.3f}) but submitting")
        
        return True
    
    async def process_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced task processing with local validation"""
        task_start = time.time()
        
        # Create task record
        task_record = self._create_task_record(task_info)
        
        try:
            # Step 1: Generate 3D model
            self.logger.info(f"🎨 Generating 3D model for: {task_info['prompt'][:50]}...")
            generation_result = await self.generate_3d_model(task_info)
            
            if not generation_result.get('success', False):
                self.logger.error(f"❌ Generation failed: {generation_result.get('error', 'Unknown error')}")
                task_record.processed_at = time.time()
                self.db.save_task(task_record)
                return {'status': 'generation_failed', 'error': generation_result.get('error')}
            
            task_record.processed_at = time.time()
            task_record.generation_time = generation_result.get('generation_time', 0.0)
            
            # Step 2: Local validation (if enabled)
            ply_data = generation_result.get('ply_data')
            if ply_data and self.local_validation_enabled:
                should_submit, local_validation_info = await self.validate_locally_before_submission(
                    task_record, ply_data
                )
                
                # Update task record with local validation info
                if local_validation_info and 'local_validation_result' in local_validation_info:
                    local_result = local_validation_info['local_validation_result']
                    task_record.local_validation_score = local_result.final_score
                    task_record.validation_time = local_validation_info.get('validation_time', 0.0)
                
                if not should_submit:
                    self.logger.warning(f"🚫 Skipping submission due to low local validation score")
                    task_record.submission_success = False
                    self.db.save_task(task_record)
                    
                    return {
                        'status': 'skipped_low_quality',
                        'local_validation': local_validation_info,
                        'generation_result': generation_result
                    }
            
            # Step 3: Submit to validator (if local validation passed or disabled)
            if self.config['submit_results'] and not task_info.get('is_default', False):
                submission_success, feedback = await self.submit_result(task_record, generation_result, None)
                task_record.submitted_at = time.time()
                task_record.submission_success = submission_success
                
                if feedback:
                    task_record.feedback_received = True
                    task_record.task_fidelity_score = feedback.task_fidelity_score
                    task_record.average_fidelity_score = feedback.average_fidelity_score
                    task_record.current_miner_reward = feedback.current_miner_reward
                    task_record.validation_failed = feedback.validation_failed
                    task_record.generations_in_window = feedback.generations_within_the_window
                    
                    # Compare local vs validator scores
                    if hasattr(task_record, 'local_validation_score') and task_record.local_validation_score:
                        score_diff = abs(task_record.local_validation_score - feedback.task_fidelity_score)
                        self.logger.info(f"📊 Score comparison - Local: {task_record.local_validation_score:.4f}, "
                                       f"Validator: {feedback.task_fidelity_score:.4f} (diff: {score_diff:.4f})")
            
            # Save to database
            self.db.save_task(task_record)
            
            task_time = time.time() - task_start
            
            # Compile task result
            task_result = {
                'task_info': task_info,
                'generation_result': generation_result,
                'local_validation': local_validation_info if self.local_validation_enabled else None,
                'submission_success': task_record.submission_success,
                'total_time': task_time,
                'timestamp': time.time(),
                'status': 'completed'
            }
            
            self.completed_tasks.append(task_result)
            
            self.logger.info(f"✅ Task completed in {task_time:.2f}s: {task_record.task_id}")
            if task_record.local_validation_score:
                self.logger.info(f"   Local score: {task_record.local_validation_score:.4f}")
            if task_record.task_fidelity_score:
                self.logger.info(f"   Validator score: {task_record.task_fidelity_score:.4f}")
            
            return task_result
            
        except Exception as e:
            self.logger.error(f"❌ Task processing error: {e}")
            traceback.print_exc()
            
            task_record.processed_at = time.time()
            self.db.save_task(task_record)
            
            return {
                'status': 'error',
                'error': str(e),
                'task_info': task_info
            }
    
    def print_enhanced_stats(self):
        """Print enhanced statistics including local validation"""
        super().print_stats()
        
        if self.local_validation_enabled and self.local_validation_stats['total_validated'] > 0:
            stats = self.local_validation_stats
            
            print(f"\n🔍 LOCAL VALIDATION STATISTICS:")
            print(f"   Total validated:    {stats['total_validated']}")
            print(f"   Passed locally:     {stats['passed_local']} ({stats['passed_local']/stats['total_validated']*100:.1f}%)")
            print(f"   Failed locally:     {stats['failed_local']} ({stats['failed_local']/stats['total_validated']*100:.1f}%)")
            print(f"   Skipped submissions: {stats['skipped_submissions']}")
            print(f"   Avg validation time: {stats['validation_time_total']/stats['total_validated']:.2f}s")
            
            print(f"\n   Grade Distribution:")
            for grade, count in stats['score_distribution'].items():
                percentage = count / stats['total_validated'] * 100 if stats['total_validated'] > 0 else 0
                print(f"     {grade.title()}: {count} ({percentage:.1f}%)")
            
            # Calculate efficiency improvement
            if stats['skipped_submissions'] > 0:
                print(f"\n   💡 Efficiency: Avoided {stats['skipped_submissions']} low-quality submissions")
                print(f"      This helps maintain higher average fidelity score!")

def main():
    """Main function to run enhanced TRELLIS orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TRELLIS Orchestrator with Local Validation")
    parser.add_argument("--config", type=str, default="trellis_config.json", help="Config file path")
    parser.add_argument("--disable-local-validation", action="store_true", help="Disable local validation")
    parser.add_argument("--local-threshold", type=float, help="Local validation threshold override")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = TrellisWithLocalValidation(args.config)
    
    # Override settings if provided
    if args.disable_local_validation:
        orchestrator.local_validation_enabled = False
    
    if args.local_threshold:
        orchestrator.local_validation_threshold = args.local_threshold
    
    # Run orchestrator
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\n🛑 Stopping orchestrator...")
        orchestrator.print_enhanced_stats()

if __name__ == "__main__":
    main() 