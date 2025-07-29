#!/usr/bin/env python3
"""
Test script for TRELLIS Output Cleanup
This script creates some test files and then tests the cleanup functionality
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path

def create_test_files():
    """Create some test files in the output directory"""
    output_dir = Path("./trellis_submit_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create some test files
    test_files = [
        "test_file_1.txt",
        "test_file_2.txt", 
        "test_file_3.txt"
    ]
    
    for filename in test_files:
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            f.write(f"This is test file {filename}\n" * 100)
    
    # Create a test directory with files
    test_dir = output_dir / "test_directory"
    test_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        test_file = test_dir / f"nested_file_{i}.txt"
        with open(test_file, 'w') as f:
            f.write(f"This is nested test file {i}\n" * 50)
    
    print(f"✅ Created test files in {output_dir}")
    print(f"   Files: {len(test_files)}")
    print(f"   Directories: 1")
    
    # Show directory size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"   Total size: {total_size / 1024:.1f} KB")

def check_output_directory():
    """Check what's in the output directory"""
    output_dir = Path("./trellis_submit_outputs")
    
    if not output_dir.exists():
        print("❌ Output directory does not exist")
        return
    
    files = list(output_dir.rglob('*'))
    file_count = len([f for f in files if f.is_file()])
    dir_count = len([f for f in files if f.is_dir()])
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    print(f"📁 Output directory contents:")
    print(f"   Files: {file_count}")
    print(f"   Directories: {dir_count}")
    print(f"   Total size: {total_size / 1024:.1f} KB")
    
    if files:
        print("   Contents:")
        for item in sorted(files):
            if item.is_file():
                size = item.stat().st_size
                print(f"     📄 {item.relative_to(output_dir)} ({size} bytes)")
            else:
                print(f"     📁 {item.relative_to(output_dir)}/")

def test_cleanup_dry_run():
    """Test cleanup in dry-run mode"""
    print("\n🧪 Testing cleanup in dry-run mode...")
    
    try:
        result = subprocess.run([
            "python3", "trellis_output_cleanup.py", 
            "--once", "--dry-run"
        ], capture_output=True, text=True, timeout=30)
        
        print("📋 Cleanup script output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Cleanup script timed out")
    except Exception as e:
        print(f"❌ Error running cleanup script: {e}")

def main():
    """Main test function"""
    print("🧪 TRELLIS Output Cleanup Test")
    print("=" * 50)
    
    # Step 1: Create test files
    print("\n1️⃣ Creating test files...")
    create_test_files()
    
    # Step 2: Check directory contents
    print("\n2️⃣ Checking directory contents...")
    check_output_directory()
    
    # Step 3: Test cleanup in dry-run mode
    print("\n3️⃣ Testing cleanup (dry-run)...")
    test_cleanup_dry_run()
    
    # Step 4: Check directory contents again
    print("\n4️⃣ Checking directory contents after dry-run...")
    check_output_directory()
    
    print("\n✅ Test completed!")
    print("\nNote: This test used dry-run mode, so no files were actually deleted.")
    print("To test actual cleanup, run: python3 trellis_output_cleanup.py --once")

if __name__ == "__main__":
    main() 