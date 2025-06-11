#  "a blue monkey sitting on temple"
'''
python generate_3d.py "a blue monkey sitting on temple" --guidance-scale 3.5 --ss-guidance-strength 7.5 --ss-sampling-steps 12 --slat-guidance-strength 3.0 --slat-sampling-steps 12
python generate_3d.py "football" --guidance-scale 3.5 --ss-guidance-strength 7.5 --ss-sampling-steps 12 --slat-guidance-strength 3.0 --slat-sampling-steps 12

'''
import os
import sys
import argparse
import imageio
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import gc

# Add TRELLIS to Python path
TRELLIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trellis")
sys.path.append(TRELLIS_PATH)

# Set spconv algorithm to native for single run and use xformers for attention
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Constants from app.py
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def generate_3d_model(prompt, output_dir="outputs", seed=42, width=1024, height=1024, guidance_scale=3.5,
                     ss_guidance_strength=7.5, ss_sampling_steps=12,
                     slat_guidance_strength=3.0, slat_sampling_steps=12):
    """
    Generate a 3D model from a text prompt using Flux and TRELLIS.
    
    Args:
        prompt (str): Text description of the 3D model to generate
        output_dir (str): Directory to save the outputs
        seed (int): Random seed for reproducibility
        width (int): Image width
        height (int): Image height
        guidance_scale (float): Flux guidance scale
        ss_guidance_strength (float): Sparse structure guidance strength
        ss_sampling_steps (int): Sparse structure sampling steps
        slat_guidance_strength (float): Structured latent guidance strength
        slat_sampling_steps (int): Structured latent sampling steps
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize Flux pipeline
    print("Loading Flux pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16
    file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
    file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
    single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
    quantization_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(single_file_base_model, subfolder="text_encoder_2", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config_tf, token=huggingface_token)
    
    if ".gguf" in file_url:
        transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=single_file_base_model)
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, token=huggingface_token)
        transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config, token=huggingface_token)
    
    flux_pipeline = FluxPipeline.from_pretrained(single_file_base_model, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype, token=huggingface_token)
    flux_pipeline.to("cuda")
    
    # Generate image using Flux with exact settings from app.py
    print(f"Generating image from prompt: {prompt}")
    # prompt = "wbgmsst, " + prompt + ", 3D isometric, white background"
    # prompt = "wbgmsst, " + prompt + ", 3D isometric cute object asset, white background"
    prompt = "wbgmsst, " + prompt + ", 3D isometric object asset, white background"
    generator = torch.Generator(device=device).manual_seed(seed)
    image = flux_pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=NUM_INFERENCE_STEPS,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    # Save the generated image
    image_path = output_path / "generated_image.png"
    image.save(image_path)
    print(f"Saved generated image to {image_path}")
    
    # Clear Flux pipeline from GPU memory
    print("Clearing GPU memory...")
    del flux_pipeline
    del transformer
    del text_encoder_2
    clear_gpu_memory()
    
    # Initialize Trellis pipeline
    print("Loading TRELLIS pipeline...")
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("cavargas10/TRELLIS")
    trellis_pipeline.cuda()
    
    try:
        trellis_pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
    except:
        pass
    
    # Generate 3D model from image with exact settings from app.py
    print("Generating 3D model from image...")
    outputs = trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    
    # Save the outputs
    print("Saving outputs...")
    
    # Save preview videos with exact settings from app.py
    print("Rendering preview videos...")
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    imageio.mimsave(output_path / "preview.mp4", video, fps=15)
    
    # Save as PLY file (3D Gaussians)
    model_name = prompt.replace(" ", "_").lower()
    outputs['gaussian'][0].save_ply(output_path / f"{model_name}_gaussian.ply")
    
    # Save as GLB file (textured mesh)
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
        verbose=False
    )
    glb.export(output_path / f"{model_name}.glb")
    
    # Clear final memory
    del trellis_pipeline
    del outputs
    clear_gpu_memory()
    
    print("\nGeneration complete! Files saved in", output_dir)
    print("- generated_image.png (Initial 2D image)")
    print(f"- {model_name}_gaussian.ply (3D Gaussians)")
    print(f"- {model_name}.glb (Textured mesh)")
    print("- preview.mp4 (Combined preview video)")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D models from text prompts using Flux and TRELLIS")
    parser.add_argument("prompt", type=str, help="Text description of the 3D model to generate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save the outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--huggingface-token", type=str, help="Hugging Face API token for accessing models")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Flux guidance scale")
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5, help="Sparse structure guidance strength")
    parser.add_argument("--ss-sampling-steps", type=int, default=12, help="Sparse structure sampling steps")
    parser.add_argument("--slat-guidance-strength", type=float, default=3.0, help="Structured latent guidance strength")
    parser.add_argument("--slat-sampling-steps", type=int, default=12, help="Structured latent sampling steps")
    
    args = parser.parse_args()
    
    if args.huggingface_token:
        os.environ["HUGGINGFACE_TOKEN"] = args.huggingface_token
    
    generate_3d_model(
        args.prompt,
        args.output_dir,
        args.seed,
        args.width,
        args.height,
        args.guidance_scale,
        args.ss_guidance_strength,
        args.ss_sampling_steps,
        args.slat_guidance_strength,
        args.slat_sampling_steps
    )

if __name__ == "__main__":
    main() 