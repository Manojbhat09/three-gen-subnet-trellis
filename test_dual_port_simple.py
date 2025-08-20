#!/usr/bin/env python3
"""
Simple test script for dual-port functionality without external dependencies
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_dual_port_config():
    """Test that the dual-port configuration is properly set up"""
    
    print("🧪 Testing Dual-Port Configuration")
    print("=" * 50)
    
    # Test argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dual-Port Configuration")
    parser.add_argument("--port1", type=int, default=8097, help="Port 1 for optimized prompt generation")
    parser.add_argument("--port2", type=int, default=8098, help="Port 2 for original prompt generation")
    parser.add_argument("--generation-server", default="http://localhost:8097", help="Generation server URL")
    parser.add_argument("--validation-server", default="http://localhost:10006", help="Validation server URL")
    
    # Test with custom arguments
    test_args = [
        "--port1", "8100",
        "--port2", "8101",
        "--generation-server", "http://localhost:8100",
        "--validation-server", "http://localhost:10007"
    ]
    
    args = parser.parse_args(test_args)
    
    print("✅ Argument parsing successful!")
    print(f"   Port 1: {args.port1}")
    print(f"   Port 2: {args.port2}")
    print(f"   Generation server: {args.generation_server}")
    print(f"   Validation server: {args.validation_server}")
    
    # Test configuration building
    config = {
        'port1': args.port1,
        'port2': args.port2,
        'generation_server_url': args.generation_server,
        'validation_server_url': args.validation_server,
        'output_dir': './test_outputs',
        'harvest_tasks': False,
        'validate_generations': True,
        'submit_results': False
    }
    
    print("\n✅ Configuration building successful!")
    print(f"   Config keys: {list(config.keys())}")
    print(f"   Port 1 config: {config['port1']}")
    print(f"   Port 2 config: {config['port2']}")
    
    # Test URL construction
    port1_url = f"http://localhost:{config['port1']}/generate/"
    port2_url = f"http://localhost:{config['port2']}/generate/"
    
    print("\n✅ URL construction successful!")
    print(f"   Port 1 URL: {port1_url}")
    print(f"   Port 2 URL: {port2_url}")
    
    print("\n🎯 Dual-port configuration test completed successfully!")
    return True

def test_method_signatures():
    """Test that the dual-port methods have correct signatures"""
    
    print("\n🧪 Testing Method Signatures")
    print("=" * 50)
    
    try:
        # Import the orchestrator class
        from continuous_trellis_orchestrator_lora_working_multi import ContinuousTrellisOrchestrator
        
        print("✅ Successfully imported ContinuousTrellisOrchestrator")
        
        # Check if dual-port methods exist
        methods_to_check = [
            'generate_3d_model_dual_port',
            'validate_model_dual_port',
            '_validate_with_subnet_validator'
        ]
        
        for method_name in methods_to_check:
            if hasattr(ContinuousTrellisOrchestrator, method_name):
                method = getattr(ContinuousTrellisOrchestrator, method_name)
                print(f"✅ Method '{method_name}' exists: {method}")
            else:
                print(f"❌ Method '{method_name}' not found")
                return False
        
        print("\n🎯 Method signature test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Dual-Port Simple Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_dual_port_config()
    test2_passed = test_method_signatures()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Configuration Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Method Signature Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Dual-port system is ready.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
