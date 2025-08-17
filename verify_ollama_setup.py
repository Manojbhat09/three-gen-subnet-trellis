#!/usr/bin/env python3
"""
Ollama Setup Verification
Check if Ollama integration is properly set up without running any servers
"""

import sys
import subprocess
import importlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_ollama_installation():
    """Check if Ollama is installed"""
    print("🦙 Checking Ollama Installation")
    print("-" * 30)
    
    try:
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama installed: {version}")
            return True
        else:
            print(f"❌ Ollama not working: {result.stderr}")
            print("📝 Install: curl -fsSL https://ollama.com/install.sh | sh")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not found")
        print("📝 Install: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_ollama_models():
    """Check if models are available"""
    print("\n📦 Checking Ollama Models")
    print("-" * 30)
    
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output and "NAME" in output:
                lines = output.split('\n')[1:]  # Skip header
                models = [line.split()[0] for line in lines if line.strip()]
                
                if models:
                    print(f"✅ Models available: {', '.join(models)}")
                    
                    # Check for recommended model
                    recommended = ["llama3.1:8b", "llama3.2:3b", "gemma2:2b"]
                    found_recommended = [m for m in models if m in recommended]
                    
                    if found_recommended:
                        print(f"✅ Recommended model found: {found_recommended[0]}")
                    else:
                        print("⚠️  No recommended models found")
                        print("📝 Consider: ollama pull llama3.1:8b")
                    
                    return True
                else:
                    print("❌ No models found")
                    print("📝 Pull a model: ollama pull llama3.1:8b")
                    return False
            else:
                print("❌ No models available")
                print("📝 Pull a model: ollama pull llama3.1:8b")
                return False
        else:
            print(f"❌ Failed to list models: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def check_python_dependencies():
    """Check if required Python packages are available"""
    print("\n🐍 Checking Python Dependencies")
    print("-" * 30)
    
    required_packages = [
        ("aiohttp", "Async HTTP client"),
        ("requests", "HTTP library"),
        ("fastapi", "Web framework"),
        ("uvicorn", "ASGI server")
    ]
    
    all_good = True
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            print(f"   📝 Install: pip install {package}")
            all_good = False
    
    return all_good

def check_project_structure():
    """Check if Ollama integration files exist"""
    print("\n📁 Checking Project Structure")
    print("-" * 30)
    
    required_files = [
        ("src/ollama_integration/__init__.py", "Ollama integration module"),
        ("src/ollama_integration/ollama_server_manager.py", "Ollama server manager"),
        ("config/settings.py", "Configuration file"),
        ("scripts/start_simple_system.py", "System startup script"),
        ("minimal_tests/test_ollama_integration.py", "Ollama tests")
    ]
    
    all_good = True
    
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} - {description}")
        else:
            print(f"❌ {file_path} - {description}")
            all_good = False
    
    return all_good

def check_configuration():
    """Check if Ollama configuration is set up"""
    print("\n⚙️  Checking Configuration")
    print("-" * 30)
    
    try:
        from config.settings import settings
        
        # Check Ollama settings
        ollama_attrs = [
            ("ollama_enabled", "Ollama enabled"),
            ("ollama_base_port", "Base port"),
            ("ollama_model_name", "Model name"),
            ("ollama_timeout", "Timeout"),
            ("ollama_startup_delay", "Startup delay")
        ]
        
        all_good = True
        
        for attr, description in ollama_attrs:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                print(f"✅ {attr}: {value} - {description}")
            else:
                print(f"❌ {attr} - {description} (missing)")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def check_gpu_agent_integration():
    """Check if GPU agent has Ollama integration"""
    print("\n🤖 Checking GPU Agent Integration")
    print("-" * 30)
    
    try:
        from src.gpu_agent.simple_gpu_agent import SimpleGPUAgent
        
        # Check if GPU agent has Ollama methods
        ollama_methods = [
            "_initialize_ollama_client",
            "_get_rl_strategy_from_ollama",
            "_get_fallback_strategy"
        ]
        
        all_good = True
        
        for method in ollama_methods:
            if hasattr(SimpleGPUAgent, method):
                print(f"✅ {method} - Method exists")
            else:
                print(f"❌ {method} - Method missing")
                all_good = False
        
        # Check constructor for ollama_port parameter
        import inspect
        sig = inspect.signature(SimpleGPUAgent.__init__)
        if 'ollama_port' in sig.parameters:
            print("✅ ollama_port parameter in constructor")
        else:
            print("❌ ollama_port parameter missing from constructor")
            all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking GPU agent integration: {e}")
        return False

def check_startup_script_integration():
    """Check if startup script includes Ollama"""
    print("\n🚀 Checking Startup Script Integration")
    print("-" * 30)
    
    try:
        startup_script = Path("scripts/start_simple_system.py")
        
        if not startup_script.exists():
            print("❌ Startup script not found")
            return False
        
        content = startup_script.read_text()
        
        checks = [
            ("OllamaServerManager", "Ollama manager import"),
            ("_start_ollama_servers", "Ollama startup method"),
            ("ollama_enabled", "Ollama enabled check"),
            ("Ollama Servers", "Status display integration")
        ]
        
        all_good = True
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✅ {check_str} - {description}")
            else:
                print(f"❌ {check_str} - {description}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking startup script: {e}")
        return False

def main():
    """Run all verification checks"""
    
    print("🔍 Ollama Integration Setup Verification")
    print("=" * 50)
    print("Checking if Ollama integration is properly set up...")
    print("(No servers will be started - completely safe)")
    print("")
    
    checks = [
        ("Ollama Installation", check_ollama_installation),
        ("Ollama Models", check_ollama_models),
        ("Python Dependencies", check_python_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("GPU Agent Integration", check_gpu_agent_integration),
        ("Startup Script Integration", check_startup_script_integration)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"💥 {check_name} check crashed: {e}")
            results[check_name] = False
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 Verification Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name:<25}: {status}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Ollama integration is properly set up")
        print("✅ Ready to run the safe test: python test_ollama_single_gpu_safe.py")
        print("✅ Ready for full system integration")
    elif passed >= 5:
        print("\n✅ MOSTLY READY!")
        print("✅ Core integration is set up")
        print("📝 Address the failed checks above")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("📝 Several components need attention")
        
        if not results.get("Ollama Installation", False):
            print("\n🔧 Quick Fix:")
            print("   1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            print("   2. Pull model: ollama pull llama3.1:8b")
            print("   3. Re-run this verification")

if __name__ == "__main__":
    main()

