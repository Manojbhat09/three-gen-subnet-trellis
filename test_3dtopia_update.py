import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# Add necessary paths to Python path
topia_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DTopia-XL")
sys.path.append(topia_dir)  # Add the parent directory first

# Now import the local modules
from dva.io import load_from_config
from dva.ray_marcher import RayMarcher
from dva.visualize import visualize_primvolume
from models.primsdf import PrimSDF
from models.diffusion import create_diffusion
from models.conditioner.text import TextConditioner
import open_clip

def main():
    parser = argparse.ArgumentParser(description='Generate 3D model from text prompt')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for 3D model generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DTopia-XL/configs/inference_dit_text.yml")
    config = OmegaConf.load(config_path)

    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.float16 if config.inference.precision == "fp16" else torch.float32

    # Create model configuration
    model_config = {
        'num_prims': config.model.num_prims,
        'dim_feat': config.model.dim_feat,
        'prim_shape': config.model.prim_shape,
        'init_scale': config.model.init_scale,
        'sdf2alpha_var': config.model.sdf2alpha_var,
        'auto_scale_init': config.model.auto_scale_init,
        'init_sampling': config.model.init_sampling
    }

    # Load the full PrimSDF model
    model = PrimSDF(**model_config)
    
    # Load the text-to-3D checkpoint
    checkpoint_path = os.path.join(topia_dir, config.checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in state_dict:
        state_dict = state_dict['ema']
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Tokenize text prompt
    text_tokens = model.conditioner.tokenizer(args.prompt).to(device)

    # Generate noise for diffusion sampling
    noise = torch.randn(1, config.model.num_prims, config.model.generator.in_channels, device=device)

    # Create diffusion model
    diffusion = create_diffusion(
        timestep_respacing=f"ddim{config.inference.ddim}",
        noise_schedule=config.diffusion.noise_schedule,
        parameterization=config.diffusion.parameterization,
        diffusion_steps=config.diffusion.diffusion_steps
    )

    # Generate 3D model
    with torch.no_grad():
        # Sample from diffusion model
        sample_fn = diffusion.ddim_sample_loop_progressive
        fwd_fn = model.forward_with_cfg

        # Set up model kwargs
        model_kwargs = dict(
            y=model.conditioner({"caption_token": text_tokens}, None),
            cfg_scale=config.inference.cfg
        )

        # Run sampling
        for samples in sample_fn(fwd_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device):
            latents = samples["sample"]

        # Decode the latents using the VAE
        params = model.decode_latents(latents)
        print(f"Decoded params shape: {params.shape}")

        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/3dtopia_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure the output directory has write permissions
        os.chmod(output_dir, 0o777)

        # Save parameters
        torch.save(params, os.path.join(output_dir, "result.pt"))

        # Create RayMarcher instance
        raymarcher = RayMarcher(
            image_height=config.image_height,
            image_width=config.image_width,
            volradius=config.rm.volradius,
            dt=config.rm.dt
        ).to(device)

        # Print debug information
        print(f"Output directory: {output_dir}")
        print(f"Output file path: {os.path.join(output_dir, 'result.jpg')}")

        # Visualize intermediate steps
        visualize_primvolume(
            os.path.join(output_dir, "result.jpg"),
            {"caption_token": text_tokens.unsqueeze(0)},
            params,  # Use the decoded params directly
            raymarcher,
            device
        )

        print("Visualization complete. Check the output directory for results.")

if __name__ == "__main__":
    main() 