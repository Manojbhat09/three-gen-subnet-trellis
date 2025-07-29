#!/usr/bin/env python3
"""
Simple test script to verify the TRELLIS simulator works correctly
"""

import asyncio
import sys
from pathlib import Path

# Test prompts
TEST_PROMPTS = [
    "simple red cube",
    "blue sphere with white stripes",
    "green cylinder"
]

def create_test_prompt_file():
    """Create a test prompt file"""
    test_file = Path("test_prompts.py")
    
    content = f'''# Test prompts for simulator
EPISODIC_TEST_PROMPTS = {TEST_PROMPTS}
'''
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created test prompt file: {test_file}")
    return test_file

async def test_simulator():
    """Test the simulator with a few simple prompts"""
    try:
        from continuous_trellis_orchestrator_simulator import ContinuousTrellisSimulator
        
        # Create test prompt file
        test_file = create_test_prompt_file()
        
        # Configure simulator for testing
        config = {
            'promptfile': str(test_file),
            'validate_generations': True,  # Enable validation to see results
            'enable_prompt_optimization': False,  # Skip optimization for testing
            'save_intermediate_results': True,
            'output_dir': './test_simulation_outputs'
        }
        
        print("🚀 Starting test simulation...")
        simulator = ContinuousTrellisSimulator(config)
        await simulator.run_simulation()
        
        print("✅ Test simulation completed successfully!")
        
        # Clean up test file
        test_file.unlink()
        print(f"🧹 Cleaned up test file: {test_file}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure continuous_trellis_orchestrator_simulator.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_simulator())
    sys.exit(0 if success else 1) 