#!/usr/bin/env python3
"""
Test script for dual-port continuous orchestrator
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from continuous_trellis_orchestrator_lora_working_multi import ContinuousTrellisOrchestrator

async def test_dual_port():
    """Test the dual-port functionality"""
    
    # Configuration for dual-port testing
    config = {
        'port1': 8097,  # Port for optimized prompt generation
        'port2': 8098,  # Port for original prompt generation
        'generation_server_url': 'http://localhost:8097',
        'validation_server_url': 'http://localhost:10006',
        'output_dir': './test_dual_port_outputs',
        'harvest_tasks': False,  # Disable task harvesting for testing
        'validate_generations': True,
        'submit_results': False,  # Disable submission for testing
        'enable_prompt_optimization': True,
        'enable_reproducibility_optimization': True,
        'enable_prompt_cleaning': True,
        'use_vllm': False,
        'ollama_url': 'http://localhost:11434'
    }
    
    print("🚀 Testing Dual-Port Continuous Orchestrator")
    print("=" * 60)
    print(f"Port 1 (optimized): {config['port1']}")
    print(f"Port 2 (original): {config['port2']}")
    print(f"Output directory: {config['output_dir']}")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = ContinuousTrellisOrchestrator(config)
    
    # Test dual-port generation with a sample task
    from dataclasses import dataclass
    
    @dataclass
    class TestTask:
        task_id: str = "test_001"
        prompt: str = "A cute 3D cat model with big eyes"
        prompt_hash: str = "test_hash"
        validator_uid: int = 1
        validator_hotkey: str = "test_hotkey"
        validator_stake: float = 1000.0
        validation_threshold: float = 0.6
        pulled_at: float = 0.0
    
    test_task = TestTask()
    
    print(f"\n🎯 Testing dual-port generation with task: {test_task.task_id}")
    print(f"Prompt: '{test_task.prompt}'")
    
    try:
        # Test dual-port generation
        result = await orchestrator.generate_3d_model_dual_port(test_task)
        
        if result:
            print("\n✅ Dual-port generation successful!")
            print(f"Port 1 PLY size: {len(result['port1_ply_data']):,} bytes")
            print(f"Port 2 PLY size: {len(result['port2_ply_data']):,} bytes")
            print(f"Optimized prompt: '{result['optimized_prompt'][:50]}...'")
            print(f"Original prompt: '{result['original_prompt'][:50]}...'")
            
            # Test validation
            print("\n🔍 Testing dual-port validation...")
            validation_result = await orchestrator.validate_model_dual_port(test_task, result)
            
            if validation_result:
                print("\n✅ Dual-port validation successful!")
                print(f"Port 1 score: {validation_result['port1_score']:.4f}")
                print(f"Port 2 score: {validation_result['port2_score']:.4f}")
                print(f"Winner: Port {validation_result['better_port']} with score {validation_result['better_score']:.4f}")
                print(f"Comparison: {validation_result['comparison_result']}")
            else:
                print("\n❌ Dual-port validation failed")
        else:
            print("\n❌ Dual-port generation failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Dual-Port Test Script")
    print("Make sure you have:")
    print("1. TRELLIS servers running on ports 8097 and 8098")
    print("2. Validation server running on port 10006")
    print("3. Ollama running on port 11434")
    print()
    
    response = input("Continue with test? (y/N): ")
    if response.lower() in ['y', 'yes']:
        asyncio.run(test_dual_port())
    else:
        print("Test cancelled")
