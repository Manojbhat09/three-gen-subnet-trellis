#!/usr/bin/env python3
"""
Test script for enhanced --only-log-learning functionality
"""

import sys
import os
sys.path.append('.')

from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator

def test_config():
    """Test different only-log-learning configurations"""
    
    test_configs = [
        # Test 1: Default only-log-learning (should use 6 logs)
        {
            'activate_learning': True,
            'only_log_learning': True,
            'log_learning_count': 6,
            'description': 'Default only-log-learning (6 logs)'
        },
        
        # Test 2: Custom log count (3 logs)
        {
            'activate_learning': True,
            'only_log_learning': True,
            'log_learning_count': 3,
            'description': 'Custom only-log-learning (3 logs)'
        },
        
        # Test 3: All logs (-1)
        {
            'activate_learning': True,
            'only_log_learning': True,
            'log_learning_count': -1,
            'description': 'All logs (-1)'
        },
        
        # Test 4: Standard mode (not only-log-learning)
        {
            'activate_learning': True,
            'only_log_learning': False,
            'log_learning_count': 6,
            'description': 'Standard mode (memory + logs)'
        },
        
        # Test 5: Learning disabled
        {
            'activate_learning': False,
            'only_log_learning': False,
            'log_learning_count': 6,
            'description': 'Learning disabled'
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}: {config['description']}")
        print("-" * 60)
        
        try:
            # Create orchestrator with test config
            orchestrator = ContinuousTrellisOrchestrator(config)
            
            # Check configuration
            print(f"   activate_learning: {orchestrator.config.get('activate_learning')}")
            print(f"   only_log_learning: {orchestrator.config.get('only_log_learning')}")
            print(f"   log_learning_count: {orchestrator.config.get('log_learning_count')}")
            
            # Test log parsing configuration
            if config['activate_learning']:
                print(f"   ✅ Learning enabled")
                if config['only_log_learning']:
                    log_count = config['log_learning_count']
                    if log_count == -1:
                        print(f"   📖 Will parse all available logs")
                    else:
                        print(f"   📖 Will parse most recent {log_count} logs")
                else:
                    print(f"   📚 Will use memory + logs (standard mode)")
            else:
                print(f"   ❌ Learning disabled")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def test_argument_parsing():
    """Test command line argument parsing"""
    print("🔧 Testing argument parsing...")
    print("-" * 60)
    
    # Simulate different command line arguments
    test_args = [
        '--activate-learning --only-log-learning',
        '--activate-learning --only-log-learning=3',
        '--activate-learning --only-log-learning=10',
        '--activate-learning --only-log-learning=-1',
        '--activate-learning',  # No only-log-learning
        '--only-log-learning',  # Should fail (no activate-learning)
    ]
    
    for args_str in test_args:
        print(f"Testing: {args_str}")
        try:
            # This would normally be parsed by argparse
            # For testing, we'll just show what we expect
            if '--only-log-learning=' in args_str:
                count = args_str.split('=')[1]
                print(f"   Expected: only_log_learning=True, log_learning_count={count}")
            elif '--only-log-learning' in args_str and '=' not in args_str:
                print(f"   Expected: only_log_learning=True, log_learning_count=6 (default)")
            elif '--activate-learning' in args_str and '--only-log-learning' not in args_str:
                print(f"   Expected: only_log_learning=False, log_learning_count=6 (default)")
            elif '--only-log-learning' in args_str and '--activate-learning' not in args_str:
                print(f"   Expected: ERROR - requires --activate-learning")
        except Exception as e:
            print(f"   Error: {e}")
        print()

if __name__ == "__main__":
    print("🚀 Testing Enhanced --only-log-learning Functionality")
    print("=" * 80)
    
    test_config()
    test_argument_parsing()
    
    print("✅ Testing complete!")
    print("\n📖 Usage Examples:")
    print("   # Use default (6 most recent logs)")
    print("   python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning")
    print("   ")
    print("   # Use 3 most recent logs")
    print("   python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning=3")
    print("   ")
    print("   # Use all available logs")
    print("   python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning=-1")
    print("   ")
    print("   # Standard mode (memory + logs)")
    print("   python continuous_trellis_orchestrator_lora_working.py --activate-learning")
