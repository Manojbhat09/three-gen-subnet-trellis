#!/usr/bin/env python3
"""
Log Monitor V2 - Sliding Window Algorithm
Monitors the log file using a sliding window approach to capture complete generation blocks
"""

import time
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

class GenerationBlock:
    """Represents a complete generation block with all associated data"""
    
    def __init__(self):
        self.timestamp = None
        self.original = None
        self.optimized = None
        self.cleaned = None
        self.task_fidelity = None
        self.average_fidelity = None
        self.miner_reward = None
        self.generations_in_window = None
        self.block_lines = []
        self.start_line_number = None
        self.end_line_number = None
    
    def is_complete(self) -> bool:
        """Check if this generation block has all required data"""
        # Must have timestamp and at least one of the key identifiers
        if not self.timestamp:
            return False
        
        # Check if this is a complete submission result
        if (self.task_fidelity is not None and 
            self.generations_in_window is not None):
            return True
        
        # Check if this is a complete prompt-based generation
        if (self.original is not None and 
            self.optimized is not None):
            return True
        
        return False
    
    def add_line(self, line: str, line_number: int):
        """Add a line to this block and extract any relevant data"""
        self.block_lines.append(line)
        
        # Extract timestamp from INFO lines
        if " - INFO - " in line and not self.timestamp:
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if timestamp_match:
                self.timestamp = timestamp_match.group(1)
        
        # Extract original prompt
        if "Original:" in line and not self.original:
            original_match = re.search(r"Original: '([^']+)'", line)
            if original_match:
                self.original = original_match.group(1)
        
        # Extract optimized prompt
        if "Optimized:" in line and not self.optimized:
            optimized_match = re.search(r"Optimized: '([^']+)'", line)
            if optimized_match:
                self.optimized = optimized_match.group(1)
        
        # Extract cleaned prompt
        if "Cleaned:" in line and not self.cleaned:
            cleaned_match = re.search(r"Cleaned: '([^']+)'", line)
            if cleaned_match:
                self.cleaned = cleaned_match.group(1)
        
        # Extract task fidelity
        if "Task fidelity:" in line and self.task_fidelity is None:
            fidelity_match = re.search(r"Task fidelity: ([\d.]+)", line)
            if fidelity_match:
                self.task_fidelity = float(fidelity_match.group(1))
        
        # Extract average fidelity
        if "Average fidelity:" in line and self.average_fidelity is None:
            avg_fidelity_match = re.search(r"Average fidelity: ([\d.]+)", line)
            if avg_fidelity_match:
                self.average_fidelity = float(avg_fidelity_match.group(1))
        
        # Extract miner reward
        if "Miner reward:" in line and self.miner_reward is None:
            reward_match = re.search(r"Miner reward: ([\d.]+)", line)
            if reward_match:
                self.miner_reward = float(reward_match.group(1))
        
        # Extract generations in window
        if "Generations in window:" in line and self.generations_in_window is None:
            window_match = re.search(r"Generations in window: (\d+)", line)
            if window_match:
                self.generations_in_window = int(window_match.group(1))
    
    def display(self):
        """Display the complete generation block in a formatted way"""
        print("\n" + "="*80)
        print(f"🕐 Timestamp: {self.timestamp or 'N/A'}")
        
        # Show prompts if available
        if self.original:
            print(f"📝 Original: '{self.original}'")
        if self.optimized:
            print(f"🚀 Optimized: '{self.optimized}'")
        if self.cleaned:
            print(f"✨ Cleaned: '{self.cleaned}'")
        
        # Show results
        if self.task_fidelity is not None:
            print(f"📊 Task Fidelity: {self.task_fidelity}")
        if self.average_fidelity is not None:
            print(f"📈 Average Fidelity: {self.average_fidelity}")
        if self.miner_reward is not None:
            print(f"💰 Miner Reward: {self.miner_reward}")
        if self.generations_in_window is not None:
            print(f"🪟 Generations in Window: {self.generations_in_window}")
        
        print(f"📍 Block lines: {self.start_line_number}-{self.end_line_number}")
        print("="*80)

class SlidingWindowLogMonitor:
    """Log monitor using sliding window algorithm to capture complete generation blocks"""
    
    def __init__(self, log_file_path: str = "continuous_trellis.log"):
        self.log_file_path = log_file_path
        self.last_position = 0
        self.generation_blocks = []
        self.last_file_size = 0
        self.last_check_time = time.time()
        self.min_window_size = 10
        self.max_window_size = 100
        self.current_window_size = 20
        
    def get_file_size(self) -> int:
        """Get current file size"""
        try:
            return os.path.getsize(self.log_file_path)
        except OSError:
            return 0
    
    def read_log_lines(self) -> List[Tuple[int, str]]:
        """Read all lines from the log file with line numbers"""
        lines = []
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    lines.append((line_num, line.strip()))
        except FileNotFoundError:
            print(f"❌ Log file not found: {self.log_file_path}")
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
        
        return lines
    
    def find_generation_blocks(self, lines: List[Tuple[int, str]]) -> List[GenerationBlock]:
        """Find complete generation blocks using sliding window approach"""
        blocks = []
        i = len(lines) - 1  # Start from bottom (most recent)
        
        while i >= 0:
            # Try to find a complete generation block starting from this position
            block = self._extract_block_at_position(lines, i)
            
            if block and block.is_complete():
                blocks.append(block)
                # Skip the lines we've already processed
                i = block.start_line_number - 2  # Move up past this block
            else:
                i -= 1  # Move up one line
        
        return blocks
    
    def _extract_block_at_position(self, lines: List[Tuple[int, str]], start_pos: int) -> Optional[GenerationBlock]:
        """Extract a generation block starting from the given position using adaptive window size"""
        if start_pos < 0 or start_pos >= len(lines):
            return None
        
        # Start with minimum window size
        window_size = self.min_window_size
        best_block = None
        
        # Try different window sizes to find the best complete block
        while window_size <= self.max_window_size and (start_pos - window_size + 1) >= 0:
            # Extract window of lines
            window_start = max(0, start_pos - window_size + 1)
            window_end = start_pos + 1
            window_lines = lines[window_start:window_end]
            
            # Try to create a block from this window
            block = self._create_block_from_lines(window_lines, window_start + 1)
            
            if block and block.is_complete():
                # Found a complete block, check if it's better than previous
                if not best_block or self._is_better_block(block, best_block):
                    best_block = block
                    # Try to reduce window size to find more precise boundaries
                    window_size = max(self.min_window_size, window_size - 5)
                else:
                    break
            else:
                # Increase window size to try to capture more data
                window_size += 10
            
            # Prevent infinite loops
            if window_size > self.max_window_size:
                break
        
        return best_block
    
    def _create_block_from_lines(self, lines: List[Tuple[int, str]], start_line_num: int) -> Optional[GenerationBlock]:
        """Create a generation block from a list of lines"""
        if not lines:
            return None
        
        block = GenerationBlock()
        block.start_line_number = start_line_num
        block.end_line_number = start_line_num + len(lines) - 1
        
        # Add all lines to the block
        for line_num, line in lines:
            block.add_line(line, line_num)
        
        return block
    
    def _is_better_block(self, new_block: GenerationBlock, old_block: GenerationBlock) -> bool:
        """Determine if the new block is better than the old one"""
        # Prefer blocks with more complete data
        new_completeness = sum(1 for v in vars(new_block).values() if v is not None)
        old_completeness = sum(1 for v in vars(old_block).values() if v is not None)
        
        if new_completeness > old_completeness:
            return True
        
        # If same completeness, prefer smaller window size
        if new_completeness == old_completeness:
            new_window_size = new_block.end_line_number - new_block.start_line_number
            old_window_size = old_block.end_line_number - old_block.start_line_number
            
            if new_window_size < old_window_size:
                return True
        
        return False
    
    def print_status_update(self, current_size: int, new_blocks_count: int):
        """Print a status update"""
        current_time = datetime.now().strftime("%H:%M:%S")
        size_mb = current_size / (1024 * 1024) if current_size > 0 else 0
        
        print(f"\n[{current_time}] 📊 Status Update:")
        print(f"   📁 File: {self.log_file_path}")
        print(f"   📏 Size: {size_mb:.2f} MB ({current_size:,} bytes)")
        print(f"   🎯 New generation blocks: {new_blocks_count}")
        print(f"   📋 Total blocks captured: {len(self.generation_blocks)}")
        print(f"   🔧 Current window size: {self.current_window_size}")
        
        if current_size > self.last_file_size:
            growth = current_size - self.last_file_size
            growth_mb = growth / (1024 * 1024)
            print(f"   📈 File growth: +{growth_mb:.2f} MB (+{growth:,} bytes)")
        elif current_size < self.last_file_size:
            print(f"   ⚠️  File size decreased (possibly rotated)")
        
        print("-" * 60)
    
    def monitor_continuously(self, check_interval: float = 5.0, quiet_mode: bool = False):
        """Continuously monitor the log file for new generation blocks"""
        print("🔍 Starting sliding window log monitoring...")
        print(f"📁 Monitoring: {self.log_file_path}")
        print(f"⏱️  Check interval: {check_interval}s")
        print(f"🔧 Window size range: {self.min_window_size}-{self.max_window_size}")
        if quiet_mode:
            print("🔇 Quiet mode: Status updates hidden, only showing generations")
        print("Press Ctrl+C to stop\n")
        
        # Initialize file size tracking
        self.last_file_size = self.get_file_size()
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time for a status update
                if current_time - self.last_check_time >= check_interval:
                    # Check if file exists and has grown
                    current_size = self.get_file_size()
                    
                    if current_size > 0:
                        # Read all lines from the log
                        all_lines = self.read_log_lines()
                        
                        # Find generation blocks
                        new_blocks = self.find_generation_blocks(all_lines)
                        
                        # Filter out blocks we've already seen
                        new_unique_blocks = []
                        for block in new_blocks:
                            if not self._is_block_duplicate(block):
                                new_unique_blocks.append(block)
                        
                        # Print status update (unless in quiet mode)
                        if not quiet_mode:
                            self.print_status_update(current_size, len(new_unique_blocks))
                        
                        # Process new blocks
                        for block in new_unique_blocks:
                            self.generation_blocks.append(block)
                            print(f"🆕 New generation block detected!")
                            block.display()
                        
                        # Update tracking variables
                        self.last_file_size = current_size
                        self.last_check_time = current_time
                
                # Wait for next check
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print(f"📊 Total generation blocks captured: {len(self.generation_blocks)}")
            
            if self.generation_blocks:
                print("\n📋 Summary of all captured blocks:")
                for i, block in enumerate(self.generation_blocks, 1):
                    print(f"\n#{i}: {block.timestamp or 'N/A'} - Fidelity: {block.task_fidelity or 'N/A'} - Reward: {block.miner_reward or 'N/A'}")
    
    def _is_block_duplicate(self, new_block: GenerationBlock) -> bool:
        """Check if a block is a duplicate of an existing one"""
        for existing_block in self.generation_blocks:
            if (existing_block.timestamp == new_block.timestamp and
                existing_block.original == new_block.original and
                existing_block.optimized == new_block.optimized):
                return True
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sliding Window Log Monitor for Continuous Trellis")
    parser.add_argument("--log-file", default="continuous_trellis.log", 
                       help="Path to the log file to monitor")
    parser.add_argument("--interval", type=float, default=5.0,
                       help="Check interval in seconds (default: 5.0)")
    parser.add_argument("--min-window", type=int, default=10,
                       help="Minimum window size (default: 10)")
    parser.add_argument("--max-window", type=int, default=100,
                       help="Maximum window size (default: 100)")
    parser.add_argument("--quiet", action="store_true",
                       help="Hide status updates, only show generation outputs")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SlidingWindowLogMonitor(args.log_file)
    monitor.min_window_size = args.min_window
    monitor.max_window_size = args.max_window
    
    # Start continuous monitoring
    monitor.monitor_continuously(args.interval, args.quiet)

if __name__ == "__main__":
    main()
