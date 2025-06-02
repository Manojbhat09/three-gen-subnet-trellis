import os
import sys
import time
import random
import uuid
import json
import shutil
import aiohttp
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRELLIS")
sys.path.append(TRELLIS_PATH)

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'  # Use xformers instead of flash-attn

import imageio
import torch
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# List of prompts to randomly choose from
PROMPTS = [
    "a blue monkey sitting on temple",
    "a red dragon perched on a castle",
    "a golden buddha statue in a garden",
    "an ancient mayan pyramid",
    "a futuristic space station",
    "a medieval castle with dragons",
    "an underwater city with mermaids",
    "a crystal cave with glowing minerals",
    "a steampunk airship floating in clouds",
    "a japanese zen garden with pagoda"
]

class TrellisMiner:
    def __init__(self, 
                 output_dir: str = "mined_outputs", 
                 model_path: str = "microsoft/TRELLIS-text-xlarge",
                 max_queue_size: int = 5,  # Keep only last 5 generations by default
                 archive_dir: Optional[str] = None,
                 prompt_endpoint: Optional[str] = "https://prompts.404.xyz/get"):  # Add prompt endpoint
        self.output_dir = output_dir
        self.max_queue_size = max_queue_size
        self.archive_dir = archive_dir
        self.prompt_endpoint = prompt_endpoint
        self.optimized_prompts = []
        
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize generation queue
        self.generation_queue = deque(maxlen=max_queue_size)
        
        # Load pipeline
        print("Loading TRELLIS pipeline...")
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(model_path)
        self.pipeline.cuda()
        print("Pipeline loaded successfully!")
        
        # Initialize mining stats
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "archived_generations": 0,
            "deleted_generations": 0,
            "start_time": datetime.now().isoformat(),
            "generations": []
        }
        self.save_stats()

    def save_stats(self):
        """Save mining statistics to a JSON file"""
        stats_file = os.path.join(self.output_dir, "mining_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def manage_queue(self, new_generation_id: str):
        """Manage the generation queue and handle old generations"""
        self.generation_queue.append(new_generation_id)
        
        # If queue is at max size, remove oldest generation
        if len(self.generation_queue) == self.max_queue_size and len(self.generation_queue) > 0:
            oldest_id = self.generation_queue[0]
            oldest_dir = os.path.join(self.output_dir, oldest_id)
            
            if os.path.exists(oldest_dir):
                if self.archive_dir:
                    # Archive the generation
                    archive_path = os.path.join(self.archive_dir, oldest_id)
                    shutil.move(oldest_dir, archive_path)
                    print(f"Archived generation {oldest_id} to {archive_path}")
                    self.stats["archived_generations"] += 1
                else:
                    # Delete the generation
                    shutil.rmtree(oldest_dir)
                    print(f"Deleted oldest generation {oldest_id}")
                    self.stats["deleted_generations"] += 1

    def get_queue_size(self) -> int:
        """Get current size of generation queue"""
        return len(self.generation_queue)

    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics in GB"""
        def get_dir_size(path):
            total = 0
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            return total
        
        stats = {
            "current_queue": get_dir_size(self.output_dir) / (1024 ** 3)  # Convert to GB
        }
        
        if self.archive_dir:
            stats["archive"] = get_dir_size(self.archive_dir) / (1024 ** 3)
            
        return stats

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def assess_quality(self, outputs) -> float:
        """Basic quality assessment of the generated outputs"""
        try:
            # Check if we have all required outputs
            if not all(k in outputs for k in ['gaussian', 'radiance_field', 'mesh']):
                return 0.0
                
            # Check if any outputs are None or empty
            if any(not outputs[k] for k in ['gaussian', 'radiance_field', 'mesh']):
                return 0.0
                
            # Check mesh quality (basic checks)
            mesh = outputs['mesh'][0]
            if mesh.vertices.shape[0] < 100 or mesh.faces.shape[0] < 100:  # Too simple mesh
                return 0.3
                
            # Check gaussian quality
            gaussian = outputs['gaussian'][0]
            if gaussian.get_xyz.shape[0] < 1000:  # Too few gaussian points
                return 0.4
                
            return 1.0  # Passed all basic quality checks
            
        except Exception as e:
            print(f"Quality assessment failed: {str(e)}")
            return 0.0

    def generate_3d_model(self, prompt: str, generation_id: str) -> Dict:
        """Generate a 3D model from a text prompt"""
        output_dir = os.path.join(self.output_dir, generation_id)
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Clear GPU memory before generation
            self.clear_gpu_memory()
            
            # Generate the 3D model
            outputs = self.pipeline.run(
                prompt,
                seed=random.randint(1, 1000000),
                sparse_structure_sampler_params={
                    "steps": 8,
                    "cfg_strength": 6.5,
                },
                slat_sampler_params={
                    "steps": 8,
                    "cfg_strength": 6.5,
                },
            )
            
            # Assess quality before proceeding
            quality_score = self.assess_quality(outputs)
            if quality_score < 0.5:
                raise Exception(f"Generated output failed quality check (score: {quality_score})")
            
            # Save preview videos
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            gaussian_video_path = os.path.join(output_dir, "preview_gaussian.mp4")
            imageio.mimsave(gaussian_video_path, video, fps=30)
            
            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            rf_video_path = os.path.join(output_dir, "preview_rf.mp4")
            imageio.mimsave(rf_video_path, video, fps=30)
            
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            mesh_video_path = os.path.join(output_dir, "preview_mesh.mp4")
            imageio.mimsave(mesh_video_path, video, fps=30)
            
            # Save PLY file (3D Gaussians)
            ply_path = os.path.join(output_dir, "model_gaussian.ply")
            outputs['gaussian'][0].save_ply(ply_path)
            
            # Save GLB file (textured mesh)
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb_path = os.path.join(output_dir, "model.glb")
            glb.export(glb_path)
            
            generation_time = time.time() - start_time
            
            # If generation took too long, log a warning
            if generation_time > 25:  # Warning at 25s to give buffer for the 30s limit
                print(f"WARNING: Generation took {generation_time:.2f}s, approaching 30s limit")
            
            # Update stats
            generation_info = {
                "id": generation_id,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "generation_time": generation_time,
                "status": "success",
                "files": {
                    "gaussian_video": f"preview_gaussian.mp4",
                    "rf_video": f"preview_rf.mp4",
                    "mesh_video": f"preview_mesh.mp4",
                    "ply": f"model_gaussian.ply",
                    "glb": f"model.glb"
                }
            }
            
            self.stats["generations"].append(generation_info)
            self.stats["successful_generations"] += 1
            self.stats["total_generations"] += 1
            
            # Manage queue after successful generation
            self.manage_queue(generation_id)
            self.save_stats()
            
            return generation_info
            
        except Exception as e:
            generation_info = {
                "id": generation_id,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }
            
            self.stats["generations"].append(generation_info)
            self.stats["failed_generations"] += 1
            self.stats["total_generations"] += 1
            self.save_stats()
            
            # Clean up failed generation directory
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            print(f"Generation failed for prompt '{prompt}': {str(e)}")
            return generation_info

    async def fetch_optimized_prompts(self):
        """Fetch optimized prompts from the prompt service"""
        if not self.prompt_endpoint:
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.prompt_endpoint}") as response:
                    if response.status == 200:
                        result = await response.json()
                        if "prompts" in result:
                            self.optimized_prompts = result["prompts"]
                            print(f"Fetched {len(self.optimized_prompts)} optimized prompts")
        except Exception as e:
            print(f"Failed to fetch optimized prompts: {str(e)}")

    def get_next_prompt(self) -> str:
        """Get next prompt, prioritizing optimized prompts"""
        if self.optimized_prompts:
            return self.optimized_prompts.pop(0)
        return random.choice(PROMPTS)

    def mine_continuously(self, custom_prompts: List[str] = None):
        """Continuously generate 3D models"""
        prompts = custom_prompts if custom_prompts else PROMPTS
        
        print(f"Starting continuous mining with {len(prompts)} prompts...")
        print(f"Keeping last {self.max_queue_size} generations in queue")
        if self.archive_dir:
            print(f"Archiving old generations to: {self.archive_dir}")
        print("Press Ctrl+C to stop mining")
        
        try:
            while True:
                # Fetch new optimized prompts if needed
                if not self.optimized_prompts and self.prompt_endpoint:
                    import asyncio
                    asyncio.run(self.fetch_optimized_prompts())
                
                prompt = self.get_next_prompt()
                generation_id = str(uuid.uuid4())
                
                print(f"\nStarting generation {self.stats['total_generations'] + 1}")
                print(f"Prompt: {prompt}")
                print(f"Generation ID: {generation_id}")
                
                generation_info = self.generate_3d_model(prompt, generation_id)
                
                if generation_info["status"] == "success":
                    print(f"Generation successful! Time taken: {generation_info['generation_time']:.2f}s")
                    print(f"Files saved in: {os.path.join(self.output_dir, generation_id)}")
                else:
                    print(f"Generation failed: {generation_info.get('error', 'Unknown error')}")
                
                # Print current stats
                print("\nMining Stats:")
                print(f"Total Generations: {self.stats['total_generations']}")
                print(f"Successful: {self.stats['successful_generations']}")
                print(f"Failed: {self.stats['failed_generations']}")
                print(f"Queue Size: {self.get_queue_size()}/{self.max_queue_size}")
                
                # Print disk usage
                disk_usage = self.get_disk_usage()
                print("\nDisk Usage:")
                print(f"Current Queue: {disk_usage['current_queue']:.2f} GB")
                if self.archive_dir:
                    print(f"Archive: {disk_usage['archive']:.2f} GB")
                
                # Clear GPU memory
                self.clear_gpu_memory()
                
        except KeyboardInterrupt:
            print("\nMining stopped by user")
            print("Final statistics saved in mining_stats.json")

def main():
    # Create miner with queue management
    miner = TrellisMiner(
        max_queue_size=5,  # Keep only last 5 generations
        archive_dir="archived_outputs",  # Optional: archive instead of delete
        prompt_endpoint="https://prompts.404.xyz/get"  # Add prompt endpoint
    )
    
    # You can provide custom prompts here if desired
    # custom_prompts = ["prompt1", "prompt2", ...]
    # miner.mine_continuously(custom_prompts)
    
    miner.mine_continuously()

if __name__ == "__main__":
    main() 