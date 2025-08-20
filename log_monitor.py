#!/usr/bin/env python3
"""
Log Monitor for Continuous Trellis Orchestrator
Monitors the log file and extracts specific generation information
"""

import time
import re
import os
from datetime import datetime
from typing import Dict, List, Optional

class LogMonitor:
    def __init__(self, log_file_path: str = "continuous_trellis.log"):
        self.log_file_path = log_file_path
        self.last_position = 0
        self.generation_data = []
        self.last_file_size = 0
        self.last_check_time = time.time()
        
    def get_file_size(self) -> int:
        """Get current file size"""
        try:
            return os.path.getsize(self.log_file_path)
        except OSError:
            return 0
    
    def read_new_lines(self) -> List[str]:
        """Read only new lines from the log file"""
        new_lines = []
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                # Seek to last known position
                f.seek(self.last_position)
                
                # Read new lines
                for line in f:
                    new_lines.append(line.strip())
                
                # Update position
                self.last_position = f.tell()
                
        except FileNotFoundError:
            print(f"❌ Log file not found: {self.log_file_path}")
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            
        return new_lines
    
    def extract_generation_info(self, lines: List[str]) -> List[Dict]:
        """Extract generation information from log lines"""
        generations = []
        current_gen = {}
        in_generation_block = False
        
        for line in lines:
            # Look for generation start - start much earlier to capture prompts
            if ("🎯 FINAL OPTIMIZATION RESULT:" in line or 
                "✅ Submission successful to UID" in line or
                "Task fidelity:" in line or
                "Original:" in line or
                "Optimized:" in line or
                "Cleaned:" in line):
                
                # If we have a previous generation that's complete, save it
                if current_gen and self._is_generation_complete(current_gen):
                    if current_gen not in generations:
                        generations.append(current_gen)
                
                # Start new generation block
                current_gen = {"timestamp": None, "original": None, "optimized": None, "cleaned": None, 
                             "task_fidelity": None, "average_fidelity": None, "miner_reward": None, 
                             "generations_in_window": None}
                in_generation_block = True
                
                # Extract any fields that might be on this line
                if "Original:" in line and not current_gen.get("original"):
                    original_match = re.search(r"Original: '([^']+)'", line)
                    if original_match:
                        current_gen["original"] = original_match.group(1)
                
                if "Optimized:" in line and not current_gen.get("optimized"):
                    optimized_match = re.search(r"Optimized: '([^']+)'", line)
                    if optimized_match:
                        current_gen["optimized"] = optimized_match.group(1)
                
                if "Cleaned:" in line and not current_gen.get("cleaned"):
                    cleaned_match = re.search(r"Cleaned: '([^']+)'", line)
                    if cleaned_match:
                        current_gen["cleaned"] = cleaned_match.group(1)
                
                if "Task fidelity:" in line and current_gen.get("task_fidelity") is None:
                    fidelity_match = re.search(r"Task fidelity: ([\d.]+)", line)
                    if fidelity_match:
                        current_gen["task_fidelity"] = float(fidelity_match.group(1))
                
                continue
            
            # Only extract fields if we're in a generation block
            if not in_generation_block:
                continue
            
            # Extract timestamp from any INFO line
            if not current_gen.get("timestamp") and " - INFO - " in line:
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                if timestamp_match:
                    current_gen["timestamp"] = timestamp_match.group(1)
            
            # Extract original prompt
            if "Original:" in line and not current_gen.get("original"):
                original_match = re.search(r"Original: '([^']+)'", line)
                if original_match:
                    current_gen["original"] = original_match.group(1)
            
            # Extract optimized prompt
            if "Optimized:" in line and not current_gen.get("optimized"):
                optimized_match = re.search(r"Optimized: '([^']+)'", line)
                if optimized_match:
                    current_gen["optimized"] = optimized_match.group(1)
            
            # Extract cleaned prompt
            if "Cleaned:" in line and not current_gen.get("cleaned"):
                cleaned_match = re.search(r"Cleaned: '([^']+)'", line)
                if cleaned_match:
                    current_gen["cleaned"] = cleaned_match.group(1)
            
            # Extract task fidelity
            if "Task fidelity:" in line and current_gen.get("task_fidelity") is None:
                fidelity_match = re.search(r"Task fidelity: ([\d.]+)", line)
                if fidelity_match:
                    current_gen["task_fidelity"] = float(fidelity_match.group(1))
            
            # Extract average fidelity
            if "Average fidelity:" in line and current_gen.get("average_fidelity") is None:
                avg_fidelity_match = re.search(r"Average fidelity: ([\d.]+)", line)
                if avg_fidelity_match:
                    current_gen["average_fidelity"] = float(avg_fidelity_match.group(1))
            
            # Extract miner reward
            if "Miner reward:" in line and current_gen.get("miner_reward") is None:
                reward_match = re.search(r"Miner reward: ([\d.]+)", line)
                if reward_match:
                    current_gen["miner_reward"] = float(reward_match.group(1))
            
            # Extract generations in window - this marks the end of a generation block
            if "Generations in window:" in line:
                window_match = re.search(r"Generations in window: (\d+)", line)
                if window_match:
                    current_gen["generations_in_window"] = int(window_match.group(1))
                
                # This line marks the end of a generation block, so mark it as complete
                if current_gen and self._is_generation_complete(current_gen):
                    if current_gen not in generations:
                        generations.append(current_gen)
                current_gen = {}
                in_generation_block = False
                continue
            
            # Check if we have a complete generation (at least timestamp and task fidelity)
            if self._is_generation_complete(current_gen):
                if current_gen not in generations:
                    generations.append(current_gen)
                current_gen = {}
                in_generation_block = False
        
        # Add the last generation if it's complete enough
        if current_gen and self._is_generation_complete(current_gen):
            if current_gen not in generations:
                generations.append(current_gen)
        
        return generations
    
    def _is_generation_complete(self, gen: Dict) -> bool:
        """Check if a generation has enough data to be considered complete"""
        # Must have timestamp
        if not gen.get("timestamp"):
            return False
        
        # Check if this is a submission result (has task fidelity and generations in window)
        if gen.get("task_fidelity") is not None and gen.get("generations_in_window") is not None:
            return True
        
        # Check if this is a prompt-based generation (has original and optimized)
        if gen.get("original") and gen.get("optimized"):
            # For prompt-based generations, we consider them complete if they have the basic prompt data
            # They may not have all the result fields yet
            return True
        
        # Check if this is a partial generation with some prompt data
        if gen.get("original") or gen.get("optimized"):
            # If we have some prompt data, consider it complete enough to display
            # This helps capture generations that might be missing some fields
            return True
        
        return False
    
    def _is_generation_started(self, gen: Dict) -> bool:
        """Check if a generation has started (has basic info but may be incomplete)"""
        basic_fields = ["timestamp", "original", "optimized"]
        return all(gen.get(field) for field in basic_fields)
    
    def _is_more_complete(self, new_gen: Dict, existing_gen: Dict) -> bool:
        """Check if new generation has more complete data than existing one"""
        # Count non-None fields
        new_count = sum(1 for v in new_gen.values() if v is not None)
        existing_count = sum(1 for v in existing_gen.values() if v is not None)
        
        # Prefer the one with more data
        if new_count > existing_count:
            return True
        
        # If same count, prefer the one with task fidelity
        if new_count == existing_count:
            if new_gen.get('task_fidelity') is not None and existing_gen.get('task_fidelity') is None:
                return True
        
        return False
    
    def display_generation(self, gen: Dict):
        """Display a single generation in a formatted way"""
        print("\n" + "="*80)
        print(f"🕐 Timestamp: {gen.get('timestamp', 'N/A')}")
        
        # Check if this is a submission result or a prompt-based generation
        if gen.get("task_fidelity") and gen.get("generations_in_window"):
            # This is a submission result
            print(f"📊 Task Fidelity: {gen.get('task_fidelity')}")
            if gen.get('average_fidelity'):
                print(f"📈 Average Fidelity: {gen.get('average_fidelity')}")
            if gen.get('miner_reward'):
                print(f"💰 Miner Reward: {gen.get('miner_reward')}")
            print(f"🪟 Generations in Window: {gen.get('generations_in_window')}")
        else:
            # This is a prompt-based generation
            if gen.get('original'):
                print(f"📝 Original: '{gen.get('original')}'")
            if gen.get('optimized'):
                print(f"🚀 Optimized: '{gen.get('optimized')}'")
            if gen.get('cleaned'):
                print(f"✨ Cleaned: '{gen.get('cleaned')}'")
            
            # Show result fields
            if gen.get('task_fidelity') is not None:
                print(f"🎯 Task Fidelity: {gen.get('task_fidelity')}")
            if gen.get('average_fidelity') is not None:
                print(f"📊 Average Fidelity: {gen.get('average_fidelity')}")
            if gen.get('miner_reward') is not None:
                print(f"💰 Miner Reward: {gen.get('miner_reward')}")
            if gen.get('generations_in_window') is not None:
                print(f"🪟 Generations in Window: {gen.get('generations_in_window')}")
        
        print("="*80)
    
    def display_recent_generations(self, count: int = 3):
        """Display the most recent generations"""
        if not self.generation_data:
            print("📭 No generation data available yet")
            return
        
        print(f"\n🔄 Displaying last {min(count, len(self.generation_data))} generations:")
        
        # Get the last N generations
        recent_gens = self.generation_data[-count:]
        
        for i, gen in enumerate(recent_gens, 1):
            print(f"\n📋 Generation #{len(self.generation_data) - count + i}:")
            self.display_generation(gen)
    
    def print_status_update(self, current_size: int, new_lines_count: int, new_generations_count: int):
        """Print a status update every 5 seconds"""
        current_time = datetime.now().strftime("%H:%M:%S")
        size_mb = current_size / (1024 * 1024) if current_size > 0 else 0
        
        print(f"\n[{current_time}] 📊 Status Update:")
        print(f"   📁 File: {self.log_file_path}")
        print(f"   📏 Size: {size_mb:.2f} MB ({current_size:,} bytes)")
        print(f"   📈 New lines since last check: {new_lines_count}")
        print(f"   🎯 New generations detected: {new_generations_count}")
        print(f"   📋 Total generations captured: {len(self.generation_data)}")
        
        if current_size > self.last_file_size:
            growth = current_size - self.last_file_size
            growth_mb = growth / (1024 * 1024)
            print(f"   📈 File growth: +{growth_mb:.2f} MB (+{growth:,} bytes)")
        elif current_size < self.last_file_size:
            print(f"   ⚠️  File size decreased (possibly rotated)")
        
        print("-" * 60)
    
    def monitor_continuously(self, check_interval: float = 5.0, quiet_mode: bool = False):
        """Continuously monitor the log file for new generations"""
        print("🔍 Starting continuous log monitoring...")
        print(f"📁 Monitoring: {self.log_file_path}")
        print(f"⏱️  Check interval: {check_interval}s")
        if quiet_mode:
            print("🔇 Quiet mode: Status updates hidden, only showing generations")
        print("Press Ctrl+C to stop\n")
        
        # Initialize file size tracking
        self.last_file_size = self.get_file_size()
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time for a status update (every 5 seconds)
                if current_time - self.last_check_time >= check_interval:
                    # Check if file exists and has grown
                    current_size = self.get_file_size()
                    
                    # Read new lines
                    new_lines = self.read_new_lines()
                    new_lines_count = len(new_lines)
                    
                    # Extract new generation data
                    new_generations = self.extract_generation_info(new_lines)
                    new_generations_count = len(new_generations)
                    
                    # Print status update every 5 seconds (unless in quiet mode)
                    if not quiet_mode:
                        self.print_status_update(current_size, new_lines_count, new_generations_count)
                    
                    # Process new generations
                    if new_generations:
                        for new_gen in new_generations:
                            # Check if this generation is already in our data
                            is_duplicate = False
                            for existing_gen in self.generation_data:
                                if (existing_gen.get('timestamp') == new_gen.get('timestamp') and 
                                    existing_gen.get('original') == new_gen.get('original')):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                # Check if this generation is more complete than what we might already have
                                should_add = True
                                for existing_gen in self.generation_data:
                                    if (existing_gen.get('timestamp') == new_gen.get('timestamp') and 
                                        existing_gen.get('original') == new_gen.get('original')):
                                        # If we already have this generation, only replace if new one is more complete
                                        if self._is_more_complete(new_gen, existing_gen):
                                            self.generation_data.remove(existing_gen)
                                        else:
                                            should_add = False
                                        break
                                
                                if should_add:
                                    self.generation_data.append(new_gen)
                                    print(f"🆕 New generation detected!")
                                    self.display_generation(new_gen)
                    
                    # Update tracking variables
                    self.last_file_size = current_size
                    self.last_check_time = current_time
                
                # Wait for next check
                time.sleep(1)  # Check every second, but only update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print(f"📊 Total generations captured: {len(self.generation_data)}")
            
            if self.generation_data:
                print("\n📋 Summary of all captured generations:")
                for i, gen in enumerate(self.generation_data, 1):
                    print(f"\n#{i}: {gen.get('timestamp', 'N/A')} - Fidelity: {gen.get('task_fidelity', 'N/A')} - Reward: {gen.get('miner_reward', 'N/A')}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Continuous Trellis log file")
    parser.add_argument("--log-file", default="continuous_trellis.log", 
                       help="Path to the log file to monitor")
    parser.add_argument("--interval", type=float, default=5.0,
                       help="Check interval in seconds (default: 5.0)")
    parser.add_argument("--show-recent", type=int, default=3,
                       help="Number of recent generations to show initially (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                       help="Hide status updates, only show generation outputs")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = LogMonitor(args.log_file)
    
    # Show recent generations if available
    monitor.display_recent_generations(args.show_recent)
    
    # Start continuous monitoring
    monitor.monitor_continuously(args.interval, args.quiet)

if __name__ == "__main__":
    main()
