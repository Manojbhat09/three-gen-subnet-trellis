#!/usr/bin/env python3
# Subnet 17 Real Mining Launcher
# Purpose: Quick start script for real Bittensor mining

import subprocess
import sys
import os

def main():
    print("🚀 Subnet 17 Real Mining Launcher")
    print("=" * 50)
    
    # Your registered credentials
    wallet_name = "test2m3b2"
    hotkey_name = "t2m3b21"
    
    print(f"💰 Wallet: {wallet_name}")
    print(f"🔑 Hotkey: {hotkey_name}")
    print(f"🆔 UID: 86")
    print()
    
    print("📋 Available Mining Scripts:")
    print("1. Real Bittensor Mining Pipeline (full production)")
    print("2. Subnet Protocol Integration (demo with real network)")
    print("3. Streamlined Production Pipeline (local generation only)")
    print()
    
    choice = input("Select script to run (1-3): ").strip()
    
    if choice == "1":
        print("🏭 Starting Real Bittensor Mining Pipeline...")
        cmd = [
            sys.executable, "real_bittensor_mining_pipeline.py",
            "--wallet", wallet_name,
            "--hotkey", hotkey_name,
            "--max-tasks", "3",
            "--concurrent", "2"
        ]
        
    elif choice == "2":
        print("🔗 Starting Subnet Protocol Integration Demo...")
        cmd = [
            sys.executable, "subnet_protocol_integration.py",
            "--wallet", wallet_name,
            "--hotkey", hotkey_name
        ]
        
    elif choice == "3":
        print("⚡ Starting Streamlined Production Pipeline...")
        cmd = [sys.executable, "streamlined_production_pipeline.py"]
        
    else:
        print("❌ Invalid choice")
        return 1
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the selected script
        result = subprocess.run(cmd, check=False)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Mining interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Error running script: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 