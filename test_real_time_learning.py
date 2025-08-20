#!/usr/bin/env python3
"""
Test script for the new real-time learning functions in the continuous orchestrator.
This script tests the log parsing and enhanced gold prompts functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_trellis_orchestrator_lora_working import ContinuousTrellisOrchestrator

def test_real_time_learning():
    """Test the real-time learning functions"""
    
    print("🧪 Testing Real-Time Learning Functions")
    print("=" * 50)
    
    # Create a test configuration with real-time learning enabled
    config = {
        'activate_learning': True,
        'generation_server_url': 'http://localhost:8097',
        'validation_server_url': 'http://localhost:10006',
        'output_dir': './test_outputs',
        'gold_prompts_reload_interval': 60,  # 1 minute for testing
    }
    
    try:
        # Create orchestrator instance
        print("📋 Creating orchestrator instance...")
        orchestrator = ContinuousTrellisOrchestrator(config)
        
        print("✅ Orchestrator created successfully")
        print(f"   Real-time learning enabled: {orchestrator.config.get('activate_learning', False)}")
        
        # Test log parsing
        print("\n📖 Testing log parsing...")
        try:
            log_prompts = orchestrator.parse_current_episode_logs()
            print(f"✅ Log parsing successful: {len(log_prompts)} prompts found")
            
            if log_prompts:
                print("   Sample prompts from logs:")
                for i, (prompt, data) in enumerate(list(log_prompts.items())[:3]):
                    print(f"     {i+1}. '{prompt[:50]}...' (score: {data.get('final_score', 'N/A')})")
            else:
                print("   No prompts found in logs (this is normal if no episodes are running)")
                
        except Exception as e:
            print(f"❌ Log parsing failed: {e}")
        
        # Test enhanced gold prompts
        print("\n🔄 Testing enhanced gold prompts...")
        try:
            fresh_prompts = orchestrator.get_fresh_gold_prompts()
            print(f"✅ Enhanced gold prompts successful: {len(fresh_prompts)} total prompts")
            
            memory_count = orchestrator.stats.get('memory_prompts', 0)
            log_count = orchestrator.stats.get('log_prompts', 0)
            print(f"   From episodic memory: {memory_count}")
            print(f"   From recent logs: {log_count}")
            print(f"   Combined total: {len(fresh_prompts)}")
            
        except Exception as e:
            print(f"❌ Enhanced gold prompts failed: {e}")
        
        # Test live monitoring setup
        print("\n📁 Testing live monitoring setup...")
        try:
            orchestrator.setup_live_episodic_memory_monitoring()
            print("✅ Live monitoring setup completed")
            
            # Check if monitoring is active
            if hasattr(orchestrator, 'episodic_memory_observer'):
                print("   Live monitoring: ACTIVE")
                # Stop monitoring for cleanup
                orchestrator.stop_live_monitoring()
                print("   Live monitoring: STOPPED")
            else:
                print("   Live monitoring: NOT AVAILABLE (watchdog not installed)")
                
        except Exception as e:
            print(f"❌ Live monitoring setup failed: {e}")
        
        print("\n🎯 Test Summary:")
        print(f"   Real-time learning: {'ENABLED' if config['activate_learning'] else 'DISABLED'}")
        print(f"   Log parsing: {'WORKING' if 'log_prompts' in locals() else 'FAILED'}")
        print(f"   Enhanced prompts: {'WORKING' if 'fresh_prompts' in locals() else 'FAILED'}")
        print(f"   Live monitoring: {'WORKING' if hasattr(orchestrator, 'episodic_memory_observer') else 'NOT AVAILABLE'}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    test_real_time_learning()
