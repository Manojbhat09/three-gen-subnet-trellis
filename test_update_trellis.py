import os
import torch
import asyncio
import aiohttp
from omegaconf import OmegaConf
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

class OptimizedTrellisTest:
    def __init__(self, 
                 config_path: str = "generation/configs/text_mv.yaml",
                 model_path: str = "microsoft/TRELLIS-text-xlarge",
                 output_dir: str = "test_outputs",
                 prompt_endpoint: str = "https://prompts.404.xyz/get"):
        self.config_path = config_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.prompt_endpoint = prompt_endpoint
        self.pipeline = None
        self.optimized_prompts = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load config
        self.opt = OmegaConf.load(config_path)
        
        # Initialize pipeline
        self.initialize_pipeline()
        
    def initialize_pipeline(self):
        """Initialize the TRELLIS pipeline"""
        print("Loading TRELLIS pipeline...")
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_path)
        if torch.cuda.is_available():
            self.pipeline.cuda()
        print("Pipeline loaded successfully!")
        
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
    async def fetch_optimized_prompts(self):
        """Fetch optimized prompts from the prompt service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.prompt_endpoint) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "prompts" in result:
                            self.optimized_prompts = result["prompts"]
                            print(f"Fetched {len(self.optimized_prompts)} optimized prompts")
                            return self.optimized_prompts[0] if self.optimized_prompts else None
        except Exception as e:
            print(f"Failed to fetch optimized prompts: {str(e)}")
        return None
        
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
            
    async def generate_and_test(self, prompt: str = None):
        """Generate and test a 3D model with optimizations"""
        try:
            # If no prompt provided, try to fetch one
            if not prompt:
                prompt = await self.fetch_optimized_prompts()
                if not prompt:
                    prompt = "a blue monkey sitting on temple"  # Default fallback
                    
            print(f"Testing generation with prompt: {prompt}")
            
            # Clear GPU memory before generation
            self.clear_gpu_memory()
            
            # Generate with optimized parameters
            outputs = self.pipeline.run(
                prompt,
                seed=42,  # Fixed seed for testing
                sparse_structure_sampler_params={
                    "steps": 8,  # Reduced steps
                    "cfg_strength": 6.5,  # Optimized strength
                },
                slat_sampler_params={
                    "steps": 8,  # Reduced steps
                    "cfg_strength": 6.5,  # Optimized strength
                },
            )
            
            # Assess quality
            quality_score = self.assess_quality(outputs)
            print(f"Generation quality score: {quality_score}")
            
            if quality_score < 0.5:
                print("Warning: Generated output failed quality check")
                return False
                
            # Save outputs
            output_path = os.path.join(self.output_dir, "test_model")
            
            # Save preview videos
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(f"{output_path}_gaussian.mp4", video, fps=30)
            
            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(f"{output_path}_rf.mp4", video, fps=30)
            
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(f"{output_path}_mesh.mp4", video, fps=30)
            
            # Save PLY file
            outputs['gaussian'][0].save_ply(f"{output_path}.ply")
            
            # Save GLB file
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(f"{output_path}.glb")
            
            print("Test generation completed successfully!")
            print(f"Output files saved in: {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"Test generation failed: {str(e)}")
            return False
        finally:
            # Clear GPU memory after generation
            self.clear_gpu_memory()

async def main():
    # Initialize test environment
    tester = OptimizedTrellisTest()
    
    # Run test generation
    success = await tester.generate_and_test()
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")

if __name__ == "__main__":
    # Import imageio here to avoid potential import issues
    import imageio
    
    # Run the async main function
    asyncio.run(main()) 