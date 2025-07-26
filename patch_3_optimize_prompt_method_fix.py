
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
