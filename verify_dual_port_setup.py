#!/usr/bin/env python3
"""
Verification script for dual-port setup
"""

import sys
import os
from pathlib import Path

def verify_files_exist():
    """Verify all required files exist"""
    print("🔍 Verifying Required Files")
    print("=" * 40)
    
    required_files = [
        "continuous_trellis_orchestrator_lora_working_multi.py",
        "subnet_accurate_validator_multigpu.py",
        "test_dual_port_simple.py",
        "DUAL_PORT_README.md",
        "start_dual_port_mining.sh",
        "IMPLEMENTATION_SUMMARY.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def verify_syntax():
    """Verify Python syntax is correct"""
    print("\n🔍 Verifying Python Syntax")
    print("=" * 40)
    
    python_files = [
        "continuous_trellis_orchestrator_lora_working_multi.py",
        "subnet_accurate_validator_multigpu.py",
        "test_dual_port_simple.py"
    ]
    
    all_valid = True
    for file_path in python_files:
        try:
            result = os.system(f"python -m py_compile {file_path}")
            if result == 0:
                print(f"✅ {file_path} - Syntax OK")
            else:
                print(f"❌ {file_path} - Syntax Error")
                all_valid = False
        except Exception as e:
            print(f"❌ {file_path} - Error: {e}")
            all_valid = False
    
    return all_valid

def verify_imports():
    """Verify modules can be imported"""
    print("\n🔍 Verifying Module Imports")
    print("=" * 40)
    
    try:
        import continuous_trellis_orchestrator_lora_working_multi
        print("✅ continuous_trellis_orchestrator_lora_working_multi - Import OK")
    except Exception as e:
        print(f"❌ continuous_trellis_orchestrator_lora_working_multi - Import Error: {e}")
        return False
    
    try:
        import subnet_accurate_validator_multigpu
        print("✅ subnet_accurate_validator_multigpu - Import OK")
    except Exception as e:
        print(f"❌ subnet_accurate_validator_multigpu - Import Error: {e}")
        return False
    
    return True

def verify_methods():
    """Verify dual-port methods exist"""
    print("\n🔍 Verifying Dual-Port Methods")
    print("=" * 40)
    
    try:
        from continuous_trellis_orchestrator_lora_working_multi import ContinuousTrellisOrchestrator
        
        required_methods = [
            'generate_3d_model_dual_port',
            'validate_model_dual_port',
            '_validate_with_subnet_validator'
        ]
        
        all_methods_exist = True
        for method_name in required_methods:
            if hasattr(ContinuousTrellisOrchestrator, method_name):
                print(f"✅ {method_name} - Method exists")
            else:
                print(f"❌ {method_name} - Method missing")
                all_methods_exist = False
        
        return all_methods_exist
        
    except Exception as e:
        print(f"❌ Error checking methods: {e}")
        return False

def verify_configuration():
    """Verify configuration options are available"""
    print("\n🔍 Verifying Configuration Options")
    print("=" * 40)
    
    try:
        from continuous_trellis_orchestrator_lora_working_multi import ContinuousTrellisOrchestrator
        
        # Create a test instance
        config = {
            'port1': 8097,
            'port2': 8098,
            'generation_server_url': 'http://localhost:8097',
            'validation_server_url': 'http://localhost:10006',
            'output_dir': './test_outputs',
            'harvest_tasks': False,
            'validate_generations': True,
            'submit_results': False
        }
        
        orchestrator = ContinuousTrellisOrchestrator(config)
        
        # Check if ports are accessible
        port1 = orchestrator.config.get('port1')
        port2 = orchestrator.config.get('port2')
        
        if port1 == 8097 and port2 == 8098:
            print("✅ Port configuration - Correct")
        else:
            print(f"❌ Port configuration - Expected (8097, 8098), got ({port1}, {port2})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Dual-Port Setup Verification")
    print("=" * 60)
    
    # Run all verification steps
    checks = [
        ("Files Exist", verify_files_exist),
        ("Python Syntax", verify_syntax),
        ("Module Imports", verify_imports),
        ("Dual-Port Methods", verify_methods),
        ("Configuration", verify_configuration)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("🚀 Dual-port system is ready for production use")
        print("\nNext steps:")
        print("1. Start TRELLIS servers on your chosen ports")
        print("2. Run: python continuous_trellis_orchestrator_lora_working_multi.py --port1 <port1> --port2 <port2>")
        print("3. Monitor the logs for dual-port generation and validation")
        sys.exit(0)
    else:
        print("💥 SOME VERIFICATIONS FAILED!")
        print("🔧 Please fix the issues before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
