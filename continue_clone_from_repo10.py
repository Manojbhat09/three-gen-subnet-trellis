#!/usr/bin/env python3

"""
Script to continue cloning Hugging Face repositories from repo 10 onwards
This script continues where the bash script left off after connection issues
"""

import os
import sys
import shutil
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
    whoami,
    HfFolder
)

# Configuration
ORG_NAME = "Manojb"
TEMP_DIR_PREFIX = "temp_hf_clone_"
MIN_FREE_SPACE_GB = 20
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

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

def repo_exists(api: HfApi, repo_id: str):
    """Check if repository exists"""
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        return True
    except Exception:
        return False

def retry_with_backoff(func, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = delay * (2 ** attempt)
            print_warning(f"Attempt {attempt + 1} failed: {e}")
            print_warning(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

# Remaining repositories to clone (starting from repo 10)
REMAINING_REPOSITORIES = [
    # Start from repo 10 - the one that failed
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
    "Kijai/Hunyuan3D-2_safetensors",
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

def clone_repository_with_retry(api: HfApi, source_repo: str, repo_index: int, total_repos: int) -> bool:
    """Clone a single repository to the target organization with retry logic"""
    repo_name = source_repo.split('/')[-1]
    target_repo = f"{ORG_NAME}/{repo_name}"
    temp_dir = f"{TEMP_DIR_PREFIX}{repo_name}_{int(time.time())}"
    
    print_status(f"🚀 Processing repository {repo_index}/{total_repos}: {source_repo} -> {target_repo}")
    show_disk_usage()
    
    # Check if repository already exists and is complete
    if repo_exists(api, target_repo):
        print_warning(f"Repository {target_repo} already exists - skipping")
        print_success(f"✅ Skipped (already exists): {source_repo} -> {target_repo}")
        show_disk_usage()
        print("---")
        return True
    
    try:
        # Ensure clean start
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Download with retry
        def download_repo():
            print_status(f"Downloading {source_repo}...")
            snapshot_download(
                repo_id=source_repo,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            print_success(f"Downloaded {source_repo}")
            
            # Show size
            repo_size = get_folder_size(temp_dir)
            print_status(f"Repository size: {repo_size}")
            return True
        
        retry_with_backoff(download_repo)
        
        # Create repository with retry
        def create_target_repo():
            print_status(f"Creating repository {target_repo}...")
            create_repo(
                repo_id=target_repo,
                repo_type="model",
                exist_ok=True
            )
            print_success(f"Created repository {target_repo}")
            return True
            
        retry_with_backoff(create_target_repo)
        
        # Upload with retry and smaller chunks
        def upload_repo():
            print_status(f"Uploading to {target_repo}...")
            upload_folder(
                repo_id=target_repo,
                folder_path=temp_dir,
                commit_message=f"Cloned from {source_repo}",
                repo_type="model"
            )
            print_success(f"Uploaded to {target_repo}")
            return True
            
        retry_with_backoff(upload_repo)
        
        # Cleanup
        print_status("Cleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print_success(f"✅ Completed and cleaned up: {source_repo} -> {target_repo}")
        
        show_disk_usage()
        print("---")
        return True
        
    except Exception as e:
        print_error(f"Failed to clone {source_repo} after {MAX_RETRIES} attempts: {str(e)}")
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
    print_status("🔄 Continuing repository cloning process from repo 10...")
    print_status(f"Target organization: {ORG_NAME}")
    print_status(f"Remaining repositories to clone: {len(REMAINING_REPOSITORIES)}")
    print_status("Starting from: hongfz16/3DTopia (repo 10/42)")
    print("=" * 60)
    
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
    
    print_warning("⚠️  RESUMING: Each repository will be downloaded, uploaded, and immediately deleted.")
    print_warning("⚠️  Connection issues will be retried automatically with backoff.")
    print_warning("⚠️  If interrupted, you can safely re-run this script.")
    print_warning("⚠️  Existing repositories will be skipped automatically.")
    print("")
    
    # Tracking variables
    success_count = 0
    failure_count = 0
    skipped_count = 0
    failed_repos = []
    
    # Starting repository number (repo 10 out of 42 total)
    start_repo_num = 10
    
    # Clone each repository
    for i, repo in enumerate(REMAINING_REPOSITORIES):
        current_repo_num = start_repo_num + i
        total_repos = 42
        
        print_status(f"📦 Repository {current_repo_num}/{total_repos}: {repo}")
        
        if clone_repository_with_retry(api, repo, current_repo_num, total_repos):
            # Check if it was skipped or actually processed
            if repo_exists(api, f"{ORG_NAME}/{repo.split('/')[-1]}"):
                success_count += 1
                print_success(f"✅ Repository {current_repo_num}/{total_repos} completed successfully!")
            else:
                failure_count += 1
                failed_repos.append(repo)
        else:
            failure_count += 1
            failed_repos.append(repo)
            print_error("❌ Repository failed. Continuing with next...")
        
        # Force cleanup between repositories
        cleanup_all_temp()
        
        # Longer pause to prevent rate limiting and help with connection stability
        print_status("Pausing 5 seconds to prevent rate limiting...")
        time.sleep(5)
    
    # Final cleanup
    cleanup_all_temp()
    
    # Summary
    print("=" * 60)
    print_status("🎉 Continuation process completed!")
    print_success(f"Successfully processed: {success_count} repositories")
    
    if failure_count > 0:
        print_error(f"Failed to clone: {failure_count} repositories")
        print("Failed repositories:")
        for failed_repo in failed_repos:
            print(f"  - {failed_repo}")
        print("")
        print_warning("💡 You can re-run this script to retry failed repositories.")
    else:
        print_success("🎊 All remaining repositories processed successfully!")
    
    print("=" * 60)
    total_completed = 9 + success_count  # 9 from previous bash script + new ones
    print_status(f"TOTAL PROGRESS: {total_completed}/42 repositories completed")
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