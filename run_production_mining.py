#!/usr/bin/env python3
# Subnet 17 Production Mining Launcher
# Quick start script for production mining operations

import sys
import os
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch production mining with optimal settings"""
    print("🚀 Subnet 17 Production Mining Launcher")
    print("=" * 50)
    
    # Check if production pipeline exists
    pipeline_path = Path("final_submission_pipeline.py")
    if not pipeline_path.exists():
        print("❌ Production pipeline not found!")
        print("   Please ensure final_submission_pipeline.py is in the current directory")
        return 1
    
    print("✅ Production pipeline found")
    print("🏭 Launching with production settings...")
    print()
    
    # Production command
    import subprocess
    cmd = [
        sys.executable, 
        "final_submission_pipeline.py",
        "--max-tasks", "10",           # Process 10 mining tasks
        "--min-score", "0.7",          # High quality threshold
        "--variants", "5",             # 5 competitive variants per task
        "--concurrent", "3",           # 3 concurrent tasks
        # "--use-bpt",                 # Uncomment to enable BPT
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("🎯 Key Features Enabled:")
    print("   ✅ Validator blacklisting (UID 180)")
    print("   ✅ Async validator operations")
    print("   ✅ Mandatory SPZ compression")
    print("   ✅ Pre-submission validation")
    print("   ✅ Empty results for failed validation")
    print("   ✅ Competitive generation variants")
    print()
    print("Press Ctrl+C to stop mining")
    print("=" * 50)
    
    try:
        # Run the production pipeline
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Mining stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Mining failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 