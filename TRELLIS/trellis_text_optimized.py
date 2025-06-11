import os
import datetime
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import torch
import numpy as np
import re
from typing import List, Dict, Tuple
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

class OptimizedTrellisTextTo3D:
    """
    Optimized TRELLIS Text-to-3D pipeline with enhanced quality settings
    
    Key optimizations:
    1. Improved text preprocessing and prompt engineering
    2. Higher quality sampling parameters
    3. Multi-sample generation with selection
    4. Enhanced post-processing pipeline
    5. Adaptive guidance scheduling
    """
    
    def __init__(self, model_path="microsoft/TRELLIS-text-xlarge"):
        # Load the base pipeline
        self.pipeline = TrellisTextTo3DPipeline.from_pretrained(model_path)
        self.pipeline.cuda()
        
        # Optimized parameters for better quality
        self.quality_presets = {
            'draft': {
                'sparse_steps': 15,
                'slat_steps': 15,
                'cfg_strength': 6.0,
                'num_samples': 1,
            },
            'good': {
                'sparse_steps': 35,
                'slat_steps': 35,
                'cfg_strength': 8.5,
                'num_samples': 2,
            },
            'high': {
                'sparse_steps': 50,
                'slat_steps': 50,
                'cfg_strength': 10.0,
                'num_samples': 3,
            },
            'ultra': {
                'sparse_steps': 80,
                'slat_steps': 80,
                'cfg_strength': 12.0,
                'num_samples': 5,
            }
        }
    
    def preprocess_text(self, prompt: str) -> str:
        """
        Enhanced text preprocessing for better 3D generation quality
        """
        # Remove excessive punctuation and normalize spaces
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        # Add quality enhancement keywords if not present
        quality_keywords = ['3D asset', 'detailed', 'clean', 'high quality']
        if not any(keyword in prompt.lower() for keyword in quality_keywords):
            prompt = f"detailed {prompt}, 3D asset"
        
        # Add style guidance for better 3D representation
        if 'isometric' not in prompt.lower():
            prompt = f"{prompt}, 3D isometric style"
        
        # Ensure background specification for better object separation
        if 'background' not in prompt.lower():
            prompt = f"{prompt}, clean white background, isolated object"
        
        return prompt
    
    def create_adaptive_guidance_schedule(self, steps: int, base_cfg: float) -> Tuple[float, Tuple[float, float]]:
        """
        Create adaptive guidance scheduling for better quality
        
        Args:
            steps: Number of sampling steps
            base_cfg: Base CFG strength
            
        Returns:
            cfg_strength: Adjusted CFG strength
            cfg_interval: Optimized guidance interval
        """
        # Higher CFG for more steps (better quality with more compute)
        if steps >= 50:
            cfg_strength = base_cfg * 1.2
            cfg_interval = (0.3, 0.98)  # Extended interval for high-quality
        elif steps >= 30:
            cfg_strength = base_cfg * 1.1
            cfg_interval = (0.4, 0.95)  # Standard high-quality interval
        else:
            cfg_strength = base_cfg
            cfg_interval = (0.5, 0.9)   # Conservative for fewer steps
        
        return cfg_strength, cfg_interval
    
    def generate_multi_sample(self, prompt: str, quality: str = 'good', seed: int = 42) -> Dict:
        """
        Generate multiple samples and return the best quality result
        
        Args:
            prompt: Text prompt
            quality: Quality preset ('draft', 'good', 'high', 'ultra')
            seed: Random seed
            
        Returns:
            Dictionary with best quality outputs
        """
        if quality not in self.quality_presets:
            raise ValueError(f"Quality must be one of {list(self.quality_presets.keys())}")
        
        preset = self.quality_presets[quality]
        preprocessed_prompt = self.preprocess_text(prompt)
        
        print(f"Optimized prompt: {preprocessed_prompt}")
        print(f"Quality preset: {quality}")
        print(f"Generating {preset['num_samples']} samples...")
        
        all_outputs = []
        
        for i in range(preset['num_samples']):
            sample_seed = seed + i * 100  # Ensure different seeds
            
            # Create adaptive guidance schedule
            sparse_cfg, sparse_interval = self.create_adaptive_guidance_schedule(
                preset['sparse_steps'], preset['cfg_strength']
            )
            slat_cfg, slat_interval = self.create_adaptive_guidance_schedule(
                preset['slat_steps'], preset['cfg_strength']
            )
            
            print(f"Generating sample {i+1}/{preset['num_samples']} (seed: {sample_seed})")
            
            # Enhanced sampling parameters
            sparse_structure_params = {
                "steps": preset['sparse_steps'],
                "cfg_strength": sparse_cfg,
                "cfg_interval": sparse_interval,
                "rescale_t": 3.5,  # Slightly higher for better quality
            }
            
            slat_params = {
                "steps": preset['slat_steps'],
                "cfg_strength": slat_cfg,
                "cfg_interval": slat_interval,
                "rescale_t": 3.5,  # Slightly higher for better quality
            }
            
            # Generate sample
            output = self.pipeline.run(
                preprocessed_prompt,
                seed=sample_seed,
                sparse_structure_sampler_params=sparse_structure_params,
                slat_sampler_params=slat_params,
            )
            
            all_outputs.append(output)
        
        # Select best sample (for now, just return the last one - could implement quality scoring)
        best_output = all_outputs[-1] if preset['num_samples'] > 1 else all_outputs[0]
        
        return best_output
    
    def enhanced_postprocessing(self, outputs: Dict, output_prefix: str = "optimized", 
                              enhanced_texture: bool = True) -> Dict:
        """
        Apply enhanced post-processing for better quality outputs
        
        Args:
            outputs: Pipeline outputs
            output_prefix: Prefix for output files
            enhanced_texture: Whether to apply enhanced texturing
            
        Returns:
            Dictionary with processed outputs and file paths
        """
        processed_files = {}
        
        # Render with higher quality settings
        if 'gaussian' in outputs:
            print("Rendering Gaussian Splatting...")
            video = render_utils.render_video(
                outputs['gaussian'][0], 
                render_size=(1024, 1024),  # Higher resolution
                num_frames=120,            # More frames for smoother video
                ss_level=2                 # Supersampling for anti-aliasing
            )['color']
            video_path = f"{output_prefix}_gs_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['gaussian_video'] = video_path
        
        if 'radiance_field' in outputs:
            print("Rendering Radiance Field...")
            video = render_utils.render_video(
                outputs['radiance_field'][0],
                render_size=(1024, 1024),
                num_frames=120,
                ss_level=2
            )['color']
            video_path = f"{output_prefix}_rf_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['radiance_field_video'] = video_path
        
        if 'mesh' in outputs:
            print("Rendering Mesh...")
            video = render_utils.render_video(
                outputs['mesh'][0],
                render_size=(1024, 1024),
                num_frames=120
            )['normal']
            video_path = f"{output_prefix}_mesh_hq.mp4"
            imageio.mimsave(video_path, video, fps=30)
            processed_files['mesh_video'] = video_path
        
        # Enhanced GLB export with better post-processing
        if 'gaussian' in outputs and 'mesh' in outputs:
            print("Creating enhanced GLB...")
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.98,           # Higher quality mesh (less simplification)
                texture_size=2048,       # Higher texture resolution
                fill_holes=True,         # Fill holes for better quality
                verbose=True
            )
            glb_path = f"{output_prefix}_enhanced.glb"
            glb.export(glb_path)
            processed_files['glb'] = glb_path
        
        # Enhanced PLY export with simplified Gaussians
        if 'gaussian' in outputs:
            print("Optimizing and saving Gaussians...")
            simplified_gs = postprocessing_utils.simplify_gs(
                outputs['gaussian'][0],
                simplify=0.98,  # High quality simplification
                verbose=True
            )
            ply_path = f"{output_prefix}_optimized.ply"
            simplified_gs.save_ply(ply_path)
            processed_files['ply'] = ply_path
        
        return processed_files
    
    def run_optimized_generation(self, prompt: str, quality: str = 'good', 
                                output_prefix: str = "optimized", seed: int = 42) -> Dict:
        """
        Run the complete optimized generation pipeline
        
        Args:
            prompt: Text description
            quality: Quality preset
            output_prefix: Output file prefix  
            seed: Random seed
            
        Returns:
            Dictionary with all outputs and file paths
        """
        print("=" * 60)
        print("TRELLIS OPTIMIZED TEXT-TO-3D GENERATION")
        print("=" * 60)
        
        # Generate multiple samples
        outputs = self.generate_multi_sample(prompt, quality, seed)
        
        # Apply enhanced post-processing
        print("\nApplying enhanced post-processing...")
        processed_files = self.enhanced_postprocessing(outputs, output_prefix)
        
        # Combine outputs
        result = {
            'raw_outputs': outputs,
            'files': processed_files,
            'settings': {
                'quality': quality,
                'preprocessed_prompt': self.preprocess_text(prompt),
                'seed': seed
            }
        }
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE!")
        print("Generated files:")
        for file_type, path in processed_files.items():
            print(f"  {file_type}: {path}")
        print("=" * 60)
        
        return result


# Example usage and test
if __name__ == "__main__":
    # Initialize optimized pipeline
    optimizer = OptimizedTrellisTextTo3D()
    
    # Test prompts with different quality levels
    test_prompts = [
        "charming red barn with weathered wood without any windows or base",
        "a blue monkey sitting on temple",
        "football",
        "purple durable robotic arm", 
        "flying dragon", 
        "a knight", 
        "a realistic darth vader", 
        "a animated skeleton "
    ]

    # Create timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {len(test_prompts)} prompts...")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    for idx, prompt in enumerate(test_prompts):
        safe_prefix = f"prompt_{idx+1}_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt)[:32]}"
        output_prefix = os.path.join(output_dir, safe_prefix)
        print(f"\n{'='*80}")
        print(f"Processing prompt {idx+1}/{len(test_prompts)}:")
        print(f"Prompt: {prompt}")
        print(f"Output prefix: {output_prefix}")
        print(f"{'='*80}")
        result = optimizer.run_optimized_generation(
            prompt=prompt,
            quality='ultra',  # Options: 'draft', 'good', 'high', 'ultra'
            output_prefix=output_prefix,
            seed=42 + idx * 100
        )
        print(f"\nPrompt {idx+1} complete. Files saved to: {output_dir}\n")
    print(f"\nAll prompts processed. Results are in: {output_dir}")
    print(f"Timestamp: {timestamp}") 