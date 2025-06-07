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
from models.diffusion import create_diffusion
from models.conditioner.text import TextConditioner
from models.dit_crossattn import DiT
from models.vae3d_dib import VAE
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

    # Load VAE
    vae = load_from_config(config.model.vae)
    vae_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DTopia-XL", config.model.vae_checkpoint_path)
    vae_state_dict = torch.load(vae_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in vae_state_dict:
        vae_state_dict = vae_state_dict['model_state_dict']
    vae.load_state_dict(vae_state_dict)
    vae = vae.to(device).eval()

    # Load DiT
    dit = load_from_config(config.model.generator)
    dit_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DTopia-XL/pretrained/model_sview_dit_fp16.pt")
    dit_state_dict = torch.load(dit_checkpoint_path, map_location='cpu')
    if 'ema' in dit_state_dict:
        dit_state_dict = dit_state_dict['ema']
    dit.load_state_dict(dit_state_dict)
    dit = dit.to(device).eval()

    # Load CLIP model and tokenizer
    model_spec = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"  # Using a pretrained model from OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(model_spec, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_spec)

    # Tokenize text prompt
    text_tokens = tokenizer(args.prompt).to(device)  # [1, seq_len]

    # Create conditioner
    conditioner = TextConditioner(
        encoder_config={
            "class_name": "models.conditioner.text.CLIPTextEncoder",
            "model_spec": model_spec,
            "pretrained_path": pretrained
        }
    )
    conditioner = conditioner.to(device).eval()

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
        fwd_fn = dit.forward_with_cfg

        # Set up model kwargs
        model_kwargs = dict(
            y=conditioner({"caption_token": text_tokens}, None),
            cfg_scale=config.inference.cfg
        )

        # Run sampling
        for samples in sample_fn(fwd_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device):
            latents = samples["sample"]

        # Print initial latents shape
        print(f"Latents shape before reshape: {latents.shape}")
        print(f"Total elements: {latents.numel()}")

        # Reshape latents for VAE decoder
        batch_size = latents.shape[0]  # 1
        num_prims = latents.shape[1]   # 2048
        in_channels = latents.shape[2] # 68

        # Reshape to [batch, num_prims, -1]
        recon_param = latents.reshape(batch_size, num_prims, -1)
        print(f"After reshape to [batch, num_prims, -1]: {recon_param.shape}")

        # Split SRT and feature params
        srt_param = recon_param[:, :, :4]
        feat_param = recon_param[:, :, 4:]  # [batch, 2048, 64]
        print(f"SRT param shape: {srt_param.shape}")
        print(f"Feature param shape: {feat_param.shape}")

        # Decode each batch element
        decoded_list = []
        for b in range(batch_size):
            # [2048, 64] -> [2048, 1, 4, 4, 4]
            feat = feat_param[b].reshape(num_prims, 1, 4, 4, 4)
            print(f"Decoded input shape for VAE: {feat.shape}")
            decoded = vae.decode(feat)
            decoded_list.append(decoded)
        # Use the first batch (since batch_size=1)
        params = decoded_list[0]
        print(f"Decoded VAE output shape: {params.shape}")

        # Reshape params to match expected format for visualize_primvolume
        # params shape: [2048, 6, 8, 8, 8]
        # We need to reshape it to [1, 2048, 4 + 6*8^3]
        # First 4 channels are position and scale, rest are features
        bs = 1  # batch size
        num_prims = params.shape[0]  # 2048
        prim_shape = params.shape[2]  # 8
        
        # Split geometry and texture features
        feat_geo = params[:, 3:, :, :, :]  # [2048, 3, 8, 8, 8]
        feat_tex = params[:, :3, :, :, :]  # [2048, 3, 8, 8, 8]
        
        # Reshape features
        feat_geo = feat_geo.reshape(bs, num_prims, -1)  # [1, 2048, 3*8*8*8]
        feat_tex = feat_tex.reshape(bs, num_prims, -1)  # [1, 2048, 3*8*8*8]
        
        # Create position and scale tensors (using default values)
        pos = torch.zeros(bs, num_prims, 3, device=device)  # [1, 2048, 3]
        scale = torch.ones(bs, num_prims, 1, device=device)  # [1, 2048, 1]
        
        # Concatenate all features
        prim_volume = torch.cat([
            scale,  # [1, 2048, 1]
            pos,    # [1, 2048, 3]
            feat_geo,  # [1, 2048, 3*8*8*8]
            feat_tex,  # [1, 2048, 3*8*8*8]
        ], dim=2)  # [1, 2048, 4 + 6*8*8*8]
        
        print(f"Final prim_volume shape: {prim_volume.shape}")

        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/3dtopia_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure the output directory has write permissions
        os.chmod(output_dir, 0o777)

        # Save parameters
        torch.save(params, os.path.join(output_dir, "result.pt"))

        # Create RayMarcher instance
        raymarcher = RayMarcher(
            image_height=518,
            image_width=518,
            volradius=config.rm.volradius,
            dt=config.rm.dt
        ).to(device)  # Move RayMarcher to device

        # Ensure text tokens are on the correct device
        text_tokens = text_tokens.to(device)

        # Print debug information
        print(f"Output directory: {output_dir}")
        print(f"Output file path: {os.path.join(output_dir, 'result.jpg')}")

        # Visualize intermediate steps
        visualize_primvolume(
            os.path.join(output_dir, "result.jpg"),
            {"caption_token": text_tokens.unsqueeze(0)},
            prim_volume,  # Use the reshaped tensor
            raymarcher,
            device
        )

        print("Visualization complete. Check the output directory for results.")

if __name__ == "__main__":
    main() 