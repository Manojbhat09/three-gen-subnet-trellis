#!/usr/bin/env python3

"""
Script to clone all Hugging Face repositories used in the project to Manojb organization
Prerequisites:
- pip install huggingface_hub
- huggingface-cli login (or set HF_TOKEN environment variable)
"""

import os
import sys
import shutil
import subprocess
import time
import signal
import atexit
import glob
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import (
    HfApi, 
    snapshot_download, 
    create_repo, 
    upload_folder,
    whoami
)

# Configuration
ORG_NAME = "Manojb"
TEMP_DIR_PREFIX = "temp_hf_clone_"
MIN_FREE_SPACE_GB = 20  # Minimum free space required in GB

# ANSI color codes for colored output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_status(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def get_disk_space_gb():
    """Get available disk space in GB"""
    statvfs = os.statvfs('.')
    available_bytes = statvfs.f_frsize * statvfs.f_bavail
    return available_bytes / (1024**3)  # Convert to GB

def check_disk_space():
    """Check if there's enough disk space"""
    available_gb = get_disk_space_gb()
    print_status(f"Available disk space: {available_gb:.1f}GB")
    
    if available_gb < MIN_FREE_SPACE_GB:
        print_error(f"Insufficient disk space! Need at least {MIN_FREE_SPACE_GB}GB free, have {available_gb:.1f}GB")
        print_error("Please free up disk space before continuing.")
        return False
    return True

def show_disk_usage():
    """Show current disk usage"""
    available_gb = get_disk_space_gb()
    print_status(f"Current available disk space: {available_gb:.1f}GB")

def cleanup_all_temp():
    """Clean up all temporary directories"""
    print_status("Cleaning up any existing temporary directories...")
    temp_dirs = glob.glob(f"{TEMP_DIR_PREFIX}*")
    temp_dirs.extend(glob.glob("temp_*"))
    
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print_status(f"Removed {temp_dir}")
        except Exception as e:
            print_warning(f"Could not remove {temp_dir}: {e}")
    
    if temp_dirs:
        print_success("Cleanup completed")
    else:
        print_status("No temporary directories found")

def get_folder_size(folder_path):
    """Get the size of a folder in human readable format"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f}{unit}"
            total_size /= 1024.0
        return f"{total_size:.1f}TB"
    except Exception:
        return "Unknown"

# All Hugging Face repositories to clone
REPOSITORIES = [
    # 3D Generation Models
    "microsoft/TRELLIS-text-xlarge",
    "microsoft/TRELLIS-text-large", 
    "microsoft/TRELLIS-text-base",
    "microsoft/TRELLIS-image-large",
    "jetx/Hunyuan3D-2",
    "tencent/Hunyuan3D-2",
    "Stable-X/trellis-normal-v0-1",
    "Stable-X/stable-normal-v0-1",
    "cavargas10/TRELLIS",
    "hongfz16/3DTopia",
    "frozenburning/3DTopia-XL",
    "stabilityai/TripoSR",
    "stepfun-ai/Step1X-3D",
    "VAST-AI-Research/DetailGen3D",
    
    # Vision & Language Models
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl",
    "Intel/dpt-large",
    "facebook/dinov2-base",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov2-small-imagenet1k-1-layer",
    
    # Stable Diffusion & Image Generation
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-base",
    "runwayml/stable-diffusion-v1-5",
    "gokaygokay/flux-game",
    "madebyollin/taesdxl",
    
    # ControlNet Models
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11e_sd15_ip2p",
    "lllyasviel/control_v11p_sd15_inpaint",
    "lllyasviel/control_v11e_sd15_depth_aware_inpaint",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11f1e_sd15_tile",
    "lllyasviel/sd-controlnet-canny",
    
    # Specialized Models
    "ashawkey/stable-zero123-diffusers",
    "ashawkey/zero123-xl-diffusers",
    "Peng-Wang/ImageDream",
    "MVDream/MVDream",
    "Kijai/Hunyuan3D2_safetensors",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    
    # Validation & Utility Models
    "404-Gen/validation",
]

def setup_cleanup_handlers():
    """Setup cleanup handlers for graceful exit"""
    def cleanup_handler(signum=None, frame=None):
        print_warning("\nCleaning up temporary directories before exit...")
        cleanup_all_temp()
        sys.exit(0)
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_all_temp)
    
    # Handle signals for graceful cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

def check_prerequisites():
    """Check if all prerequisites are met"""
    try:
        # Check if user is authenticated
        user_info = whoami()
        print_success(f"Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print_error("Authentication failed. Please run 'huggingface-cli login' first.")
        print_error(f"Error: {str(e)}")
        return False

def clone_repository(api: HfApi, source_repo: str) -> bool:
    """Clone a single repository to the target organization"""
    repo_name = source_repo.split('/')[-1]
    target_repo = f"{ORG_NAME}/{repo_name}"
    temp_dir = f"{TEMP_DIR_PREFIX}{repo_name}_{int(time.time())}"
    
    print_status(f"Processing: {source_repo} -> {target_repo}")
    show_disk_usage()
    
    try:
        # Ensure clean start - remove any existing temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Download the original repository
        print_status(f"Downloading {source_repo}...")
        snapshot_download(
            repo_id=source_repo,
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        print_success(f"Downloaded {source_repo}")
        
        # Show size of downloaded repository
        repo_size = get_folder_size(temp_dir)
        print_status(f"Repository size: {repo_size}")
        
        # Create new repository in target organization
        print_status(f"Creating repository {target_repo}...")
        create_repo(
            repo_id=target_repo,
            repo_type="model",
            exist_ok=True
        )
        print_success(f"Created repository {target_repo}")
        
        # Upload to new repository
        print_status(f"Uploading to {target_repo}...")
        upload_folder(
            repo_id=target_repo,
            folder_path=temp_dir,
            commit_message=f"Cloned from {source_repo}",
            repo_type="model"
        )
        print_success(f"Uploaded to {target_repo}")
        
        # Immediate cleanup after successful upload
        print_status("Cleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print_success(f"✅ Completed and cleaned up: {source_repo} -> {target_repo}")
        
        # Show disk space after cleanup
        show_disk_usage()
        print("---")
        return True
        
    except Exception as e:
        print_error(f"Failed to clone {source_repo}: {str(e)}")
        # Clean up on failure
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print_status("Cleaned up temporary directory after failure")
            except Exception as cleanup_error:
                print_warning(f"Could not cleanup {temp_dir}: {cleanup_error}")
        print("---")
        return False

def main():
    """Main execution function"""
    print_status("Starting repository cloning process...")
    print_status(f"Target organization: {ORG_NAME}")
    print_status(f"Total repositories to clone: {len(REPOSITORIES)}")
    print("=" * 50)
    
    # Setup cleanup handlers
    setup_cleanup_handlers()
    
    # Initial cleanup
    cleanup_all_temp()
    
    # Check disk space
    if not check_disk_space():
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    print_warning("⚠️  IMPORTANT: Each repository will be downloaded, uploaded, and immediately deleted.")
    print_warning("⚠️  This process minimizes disk usage but requires stable internet connection.")
    print_warning("⚠️  If interrupted, you can safely re-run the script.")
    print("")
    
    # Tracking variables
    success_count = 0
    failure_count = 0
    failed_repos = []
    
    # Clone each repository
    for i, repo in enumerate(REPOSITORIES, 1):
        print_status(f"🚀 Starting repository {i}/{len(REPOSITORIES)}: {repo}")
        
        if clone_repository(api, repo):
            success_count += 1
            print_success(f"✅ Repository {success_count}/{len(REPOSITORIES)} completed successfully!")
        else:
            failure_count += 1
            failed_repos.append(repo)
            print_error("❌ Repository failed. Continuing with next...")
        
        # Force cleanup between repositories
        cleanup_all_temp()
        
        # Short pause to prevent rate limiting
        time.sleep(2)
    
    # Final cleanup
    cleanup_all_temp()
    
    # Summary
    print("=" * 50)
    print_status("🎉 Cloning process completed!")
    print_success(f"Successfully cloned: {success_count} repositories")
    
    if failure_count > 0:
        print_error(f"Failed to clone: {failure_count} repositories")
        print("Failed repositories:")
        for failed_repo in failed_repos:
            print(f"  - {failed_repo}")
        print("")
        print_warning("💡 You can re-run this script to retry failed repositories.")
    else:
        print_success("🎊 All repositories cloned successfully!")
    
    print("=" * 50)
    print_status(f"All repositories are now available in the {ORG_NAME} organization on Hugging Face!")
    show_disk_usage()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user")
        cleanup_all_temp()
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        cleanup_all_temp()
        sys.exit(1) 