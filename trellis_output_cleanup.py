#!/usr/bin/env python3
"""
TRELLIS Output Directory Cleanup Script
Purpose: Safely clean up trellis_submit_outputs directory every 15 minutes
         Only cleans when server is not processing any requests to avoid breaking generations

Usage:
    python trellis_output_cleanup.py [--server-url http://localhost:8096] [--interval 900] [--dry-run]

Features:
- Monitors server job status via /job/status/ endpoint
- Only cleans when server status is "idle" or "completed" or "failed"
- Avoids cleaning during "processing" state to prevent generation failures
- Logs all cleanup activities with timestamps
- Supports dry-run mode for testing
- Configurable cleanup interval (default: 15 minutes = 900 seconds)
"""

import os
import sys
import time
import shutil
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trellis_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrellisOutputCleaner:
    """Safely clean up TRELLIS output directory based on server status"""
    
    def __init__(self, server_url: str = "http://localhost:8096", 
                 output_dir: str = "./trellis_submit_outputs",
                 cleanup_interval: int = 900,  # 15 minutes
                 dry_run: bool = False):
        """
        Initialize the cleanup script
        
        Args:
            server_url: URL of the TRELLIS server
            output_dir: Directory to clean up
            cleanup_interval: Interval between cleanup attempts in seconds
            dry_run: If True, only log what would be cleaned without actually deleting
        """
        self.server_url = server_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.cleanup_interval = cleanup_interval
        self.dry_run = dry_run
        
        # Statistics
        self.stats = {
            'cleanup_attempts': 0,
            'successful_cleanups': 0,
            'skipped_cleanups': 0,
            'failed_cleanups': 0,
            'total_files_removed': 0,
            'total_dirs_removed': 0,
            'total_space_freed_mb': 0,
            'last_cleanup_time': None,
            'start_time': time.time()
        }
        
        logger.info(f"🧹 TRELLIS Output Cleaner initialized")
        logger.info(f"   Server URL: {self.server_url}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Cleanup interval: {cleanup_interval} seconds ({cleanup_interval/60:.1f} minutes)")
        logger.info(f"   Dry run mode: {'ENABLED' if dry_run else 'DISABLED'}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
    def check_server_status(self) -> Optional[Dict[str, Any]]:
        """
        Check the current status of the TRELLIS server
        
        Returns:
            Dictionary with server status info, or None if check failed
        """
        try:
            # Check server health first
            health_url = f"{self.server_url}/health/"
            health_resp = requests.get(health_url, timeout=5)
            if health_resp.status_code != 200:
                logger.warning(f"⚠️ Server health check failed: HTTP {health_resp.status_code}")
                return None
            
            # Get job status
            job_status_url = f"{self.server_url}/job/status/"
            job_resp = requests.get(job_status_url, timeout=5)
            
            if job_resp.status_code == 200:
                job_data = job_resp.json()
                return job_data
            else:
                logger.warning(f"⚠️ Job status check failed: HTTP {job_resp.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Cannot connect to server at {self.server_url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Server status check timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error checking server status: {e}")
            return None
    
    def is_server_processing(self, job_status: Dict[str, Any]) -> bool:
        """
        Check if the server is currently processing a request
        
        Args:
            job_status: Job status dictionary from server
            
        Returns:
            True if server is processing, False otherwise
        """
        status = job_status.get('status', 'unknown')
        
        # Server is processing if status is "processing"
        if status == "processing":
            job_id = job_status.get('job_id', 'unknown')
            prompt = job_status.get('prompt', 'unknown')
            start_time = job_status.get('start_time')
            
            if start_time:
                processing_duration = time.time() - start_time
                logger.info(f"🔄 Server is processing job {job_id}: '{prompt[:50]}...' (duration: {processing_duration:.1f}s)")
            else:
                logger.info(f"🔄 Server is processing job {job_id}: '{prompt[:50]}...'")
            
            return True
        
        # Server is not processing if status is "idle", "completed", or "failed"
        if status in ["idle", "completed", "failed"]:
            logger.info(f"✅ Server is {status} - safe to clean up")
            return False
        
        # Unknown status - be conservative and assume it's processing
        logger.warning(f"⚠️ Unknown server status: {status} - assuming processing")
        return True
    
    def get_directory_size(self, path: Path) -> int:
        """
        Calculate total size of directory in bytes
        
        Args:
            path: Directory path
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"❌ Error calculating directory size: {e}")
        return total_size
    
    def count_files_and_dirs(self, path: Path) -> tuple[int, int]:
        """
        Count files and directories in a path
        
        Args:
            path: Directory path
            
        Returns:
            Tuple of (file_count, directory_count)
        """
        file_count = 0
        dir_count = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                dir_count += len(dirnames)
                file_count += len(filenames)
        except Exception as e:
            logger.error(f"❌ Error counting files/dirs: {e}")
        
        return file_count, dir_count
    
    def safe_cleanup(self) -> bool:
        """
        Safely clean up the output directory if server is not processing
        
        Returns:
            True if cleanup was performed, False if skipped or failed
        """
        self.stats['cleanup_attempts'] += 1
        
        # Check server status
        job_status = self.check_server_status()
        if job_status is None:
            logger.warning(f"⚠️ Could not check server status - skipping cleanup")
            self.stats['skipped_cleanups'] += 1
            return False
        
        # Check if server is processing
        if self.is_server_processing(job_status):
            logger.info(f"⏭️ Server is processing - skipping cleanup")
            self.stats['skipped_cleanups'] += 1
            return False
        
        # Server is not processing - safe to clean up
        logger.info(f"🧹 Starting cleanup of {self.output_dir}")
        
        try:
            # Calculate size before cleanup
            size_before = self.get_directory_size(self.output_dir)
            file_count_before, dir_count_before = self.count_files_and_dirs(self.output_dir)
            
            logger.info(f"   Before cleanup: {file_count_before} files, {dir_count_before} dirs, {size_before/1024/1024:.1f} MB")
            
            if self.dry_run:
                logger.info(f"   DRY RUN: Would remove {file_count_before} files and {dir_count_before} directories")
                logger.info(f"   DRY RUN: Would free {size_before/1024/1024:.1f} MB")
                self.stats['successful_cleanups'] += 1
                self.stats['total_files_removed'] += file_count_before
                self.stats['total_dirs_removed'] += dir_count_before
                self.stats['total_space_freed_mb'] += size_before / 1024 / 1024
                return True
            
            # Perform actual cleanup
            if self.output_dir.exists():
                # Remove all contents but keep the directory itself
                for item in self.output_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                
                logger.info(f"✅ Cleanup completed successfully")
                
                # Calculate size after cleanup
                size_after = self.get_directory_size(self.output_dir)
                file_count_after, dir_count_after = self.count_files_and_dirs(self.output_dir)
                
                files_removed = file_count_before - file_count_after
                dirs_removed = dir_count_before - dir_count_after
                space_freed_mb = (size_before - size_after) / 1024 / 1024
                
                logger.info(f"   After cleanup: {file_count_after} files, {dir_count_after} dirs, {size_after/1024/1024:.1f} MB")
                logger.info(f"   Removed: {files_removed} files, {dirs_removed} directories")
                logger.info(f"   Space freed: {space_freed_mb:.1f} MB")
                
                # Update statistics
                self.stats['successful_cleanups'] += 1
                self.stats['total_files_removed'] += files_removed
                self.stats['total_dirs_removed'] += dirs_removed
                self.stats['total_space_freed_mb'] += space_freed_mb
                self.stats['last_cleanup_time'] = time.time()
                
                return True
            else:
                logger.info(f"   Output directory does not exist - nothing to clean")
                self.stats['successful_cleanups'] += 1
                return True
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            self.stats['failed_cleanups'] += 1
            return False
    
    def print_statistics(self):
        """Print current cleanup statistics"""
        uptime_hours = (time.time() - self.stats['start_time']) / 3600
        
        logger.info("📊 CLEANUP STATISTICS")
        logger.info("="*50)
        logger.info(f"Uptime: {uptime_hours:.2f} hours")
        logger.info(f"Cleanup attempts: {self.stats['cleanup_attempts']}")
        logger.info(f"Successful cleanups: {self.stats['successful_cleanups']}")
        logger.info(f"Skipped cleanups: {self.stats['skipped_cleanups']}")
        logger.info(f"Failed cleanups: {self.stats['failed_cleanups']}")
        logger.info(f"Total files removed: {self.stats['total_files_removed']}")
        logger.info(f"Total directories removed: {self.stats['total_dirs_removed']}")
        logger.info(f"Total space freed: {self.stats['total_space_freed_mb']:.1f} MB")
        
        if uptime_hours > 0:
            cleanups_per_hour = self.stats['successful_cleanups'] / uptime_hours
            space_per_hour = self.stats['total_space_freed_mb'] / uptime_hours
            logger.info(f"Cleanups per hour: {cleanups_per_hour:.1f}")
            logger.info(f"Space freed per hour: {space_per_hour:.1f} MB")
        
        if self.stats['last_cleanup_time']:
            last_cleanup_ago = time.time() - self.stats['last_cleanup_time']
            logger.info(f"Last cleanup: {last_cleanup_ago/60:.1f} minutes ago")
        
        logger.info("="*50)
    
    def run_continuous_cleanup(self):
        """Run continuous cleanup loop"""
        logger.info(f"🚀 Starting continuous cleanup loop (interval: {self.cleanup_interval}s)")
        
        try:
            while True:
                # Perform cleanup
                self.safe_cleanup()
                
                # Print statistics every 10 cleanups
                if self.stats['cleanup_attempts'] % 10 == 0:
                    self.print_statistics()
                
                # Wait for next cleanup cycle
                logger.info(f"⏰ Waiting {self.cleanup_interval} seconds until next cleanup...")
                time.sleep(self.cleanup_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Cleanup interrupted by user")
        except Exception as e:
            logger.error(f"❌ Cleanup loop error: {e}")
        finally:
            self.print_statistics()
            logger.info("🏁 Cleanup stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TRELLIS Output Directory Cleanup Script")
    parser.add_argument("--server-url", default="http://localhost:8096", 
                       help="TRELLIS server URL (default: http://localhost:8096)")
    parser.add_argument("--output-dir", default="./trellis_submit_outputs",
                       help="Output directory to clean (default: ./trellis_submit_outputs)")
    parser.add_argument("--interval", type=int, default=900,
                       help="Cleanup interval in seconds (default: 900 = 15 minutes)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode - only log what would be cleaned without actually deleting")
    parser.add_argument("--once", action="store_true",
                       help="Run cleanup once and exit (instead of continuous loop)")
    
    args = parser.parse_args()
    
    # Validate interval
    if args.interval < 60:
        logger.warning(f"⚠️ Very short interval ({args.interval}s) - minimum recommended is 60s")
    
    # Create cleaner
    cleaner = TrellisOutputCleaner(
        server_url=args.server_url,
        output_dir=args.output_dir,
        cleanup_interval=args.interval,
        dry_run=args.dry_run
    )
    
    if args.once:
        # Run cleanup once
        logger.info("🔄 Running single cleanup...")
        cleaner.safe_cleanup()
        cleaner.print_statistics()
    else:
        # Run continuous cleanup
        cleaner.run_continuous_cleanup()

if __name__ == "__main__":
    main() 