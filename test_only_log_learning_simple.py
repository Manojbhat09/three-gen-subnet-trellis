#!/usr/bin/env python3
"""
Simple test for enhanced --only-log-learning functionality
"""

def test_argument_parsing():
    """Test command line argument parsing logic"""
    print("🔧 Testing argument parsing logic...")
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
            # Simulate argument parsing
            activate_learning = '--activate-learning' in args_str
            only_log_learning = '--only-log-learning' in args_str
            
            if only_log_learning and not activate_learning:
                print(f"   ❌ ERROR - requires --activate-learning")
                continue
                
            if only_log_learning:
                if '=' in args_str:
                    count = args_str.split('=')[1]
                    print(f"   ✅ only_log_learning=True, log_learning_count={count}")
                else:
                    print(f"   ✅ only_log_learning=True, log_learning_count=6 (default)")
            elif activate_learning:
                print(f"   ✅ only_log_learning=False, log_learning_count=6 (default)")
            else:
                print(f"   ❌ Learning disabled")
                
        except Exception as e:
            print(f"   Error: {e}")
        print()

def test_config_logic():
    """Test configuration logic"""
    print("⚙️ Testing configuration logic...")
    print("-" * 60)
    
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
        print("-" * 40)
        
        # Simulate configuration logic
        activate_learning = config['activate_learning']
        only_log_learning = config['only_log_learning']
        log_learning_count = config['log_learning_count']
        
        print(f"   activate_learning: {activate_learning}")
        print(f"   only_log_learning: {only_log_learning}")
        print(f"   log_learning_count: {log_learning_count}")
        
        # Test log parsing configuration
        if activate_learning:
            print(f"   ✅ Learning enabled")
            if only_log_learning:
                if log_learning_count == -1:
                    print(f"   📖 Will parse all available logs")
                else:
                    print(f"   📖 Will parse most recent {log_learning_count} logs")
            else:
                print(f"   📚 Will use memory + logs (standard mode)")
        else:
            print(f"   ❌ Learning disabled")
        
        print()

def test_log_parsing_logic():
    """Test log parsing logic"""
    print("📖 Testing log parsing logic...")
    print("-" * 60)
    
    # Simulate available logs
    available_logs = 15
    print(f"Available logs: {available_logs}")
    
    test_scenarios = [
        {'mode': 'only-log-learning', 'count': 6, 'expected': 6},
        {'mode': 'only-log-learning', 'count': 3, 'expected': 3},
        {'mode': 'only-log-learning', 'count': -1, 'expected': 15},
        {'mode': 'only-log-learning', 'count': 20, 'expected': 15},  # More than available
        {'mode': 'standard', 'count': 10, 'expected': 10},
    ]
    
    for scenario in test_scenarios:
        mode = scenario['mode']
        count = scenario['count']
        expected = scenario['expected']
        
        if mode == 'only-log-learning':
            if count == -1:
                logs_to_parse = available_logs
                log_info = f"all {available_logs} available logs"
            else:
                logs_to_parse = min(count, available_logs)
                log_info = f"most recent {logs_to_parse} logs"
            
            print(f"   {mode} (count={count}): {log_info}")
        else:
            logs_to_parse = min(count, available_logs)
            print(f"   {mode} (count={count}): most recent {logs_to_parse} logs")
        
        print(f"      Expected: {expected}, Actual: {logs_to_parse}")
        print()

if __name__ == "__main__":
    print("🚀 Testing Enhanced --only-log-learning Functionality")
    print("=" * 80)
    
    test_argument_parsing()
    test_config_logic()
    test_log_parsing_logic()
    
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
    print("   ")
    print("   # Use 10 most recent logs")
    print("   python continuous_trellis_orchestrator_lora_working.py --activate-learning --only-log-learning=10")
