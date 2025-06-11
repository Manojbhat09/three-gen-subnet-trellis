# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import gc

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover

# Import Flux components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
import random 
# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def text_to_3d(prompt='a car', output_dir='outputs', seed=42): # seed=42random.randint(1, 100)
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check and install required dependencies
    try:
        from google import protobuf
    except ImportError:
        print("Installing required protobuf package...")
        import subprocess
        subprocess.check_call(["pip", "install", "--no-cache-dir", "protobuf"])
        from google import protobuf
    
    # Initialize Flux pipeline
    print("Loading Flux pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16
    
    # Use the correct model configuration
    file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
    file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
    single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
    
    try:
        # Load text encoder with 8-bit quantization
        quantization_config_tf = BitsAndBytesConfigTF(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            single_file_base_model, 
            subfolder="text_encoder_2", 
            torch_dtype=dtype, 
            quantization_config=quantization_config_tf, 
            token=huggingface_token
        )
        
        # Load transformer with GGUF configuration
        transformer = FluxTransformer2DModel.from_single_file(
            file_url, 
            subfolder="transformer", 
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
            torch_dtype=dtype, 
            config=single_file_base_model
        )
        
        # Initialize pipeline
        flux_pipeline = FluxPipeline.from_pretrained(
            single_file_base_model, 
            transformer=transformer, 
            text_encoder_2=text_encoder_2, 
            torch_dtype=dtype, 
            token=huggingface_token
        )
        flux_pipeline.to("cuda")
        
        # Generate image using Flux
        print(f"Generating image from prompt: {prompt}")
        # prompt = "wbgmsst, " + prompt + ", 3D isometric cute object asset, white background"
        prompt = "wbgmsst, " + prompt + ", 3D isometric asset, clean white background, isolated object"
        generator = torch.Generator(device=device).manual_seed(seed)
        image = flux_pipeline(
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=NUM_INFERENCE_STEPS,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
        
        # Save the generated image
        original_image_path = output_path / "t2i_original.png"
        image.save(original_image_path)
        print(f"Saved original image to {original_image_path}")
        
        # Clear Flux pipeline from GPU memory
        print("Clearing GPU memory...")
        del flux_pipeline
        del transformer
        del text_encoder_2
        clear_gpu_memory()
        
        # Initialize Hunyuan3D pipeline
        print("Loading Hunyuan3D pipeline...")
        rembg = BackgroundRemover()
        model_path = 'jetx/Hunyuan3D-2'
        
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, use_safetensors=True)
            actual_model_path = pipeline.kwargs['from_pretrained_kwargs']['model_path']
        except Exception as e:
            print(f"Error loading model with safetensors: {e}")
            try:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, use_safetensors=False)
                actual_model_path = pipeline.kwargs['from_pretrained_kwargs']['model_path']
            except Exception as e:
                print(f"Error loading model without safetensors: {e}")
                raise RuntimeError("Failed to load model. Please check your installation and model files.")
        
        try:
            print("Removing background...")
            image_no_bg = rembg(image)
            print("Background removed successfully")
            
            # Save the image after background removal
            no_bg_image_path = output_path / "t2i_no_bg.png"
            image_no_bg.save(no_bg_image_path)
            print(f"Saved background-removed image to {no_bg_image_path}")
            
            print("Generating 3D mesh...")
            mesh = pipeline(image=image_no_bg, num_inference_steps=30, mc_algo='mc',
                          generator=torch.manual_seed(seed))[0]
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh)
            mesh.export(output_path / 't2i_final_1.glb')
            print("Successfully generated and exported t2i_final_1.glb")
            
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                print("Starting texture generation...")
                print(f"Using model path: {actual_model_path}")
                try:
                    import dataclasses_json
                except ImportError:
                    print("dataclasses_json is required for texture generation. Please install it with: pip install dataclasses-json")
                    print("Continuing with untextured mesh...")
                    return
                pipeline = Hunyuan3DPaintPipeline.from_pretrained(actual_model_path)
                mesh = pipeline(mesh, image=image_no_bg)
                mesh.export(output_path / 't2i_texture.glb')
                print("Successfully generated and exported t2i_texture.glb")
            except Exception as e:
                print(f"Texture generation failed: {e}")
                print("Continuing with untextured mesh...")
        except Exception as e:
            print(f"Error during mesh generation: {e}")
            raise
            
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise
    finally:
        clear_gpu_memory()

if __name__ == '__main__':
    # text_to_3d("a blue monkey sitting on temple") 
    # text_to_3d("woman monkey with red breasts") 
    # text_to_3d("football") 
    # text_to_3d("purple durable robotic arm") 
    # text_to_3d("3 eyed, red and green mystical creature with red horns guarding a big gate with spikes on it", seed=121) 
    # text_to_3d("flying dragon")
    # text_to_3d("a knight") 
    # text_to_3d("a realistic darth vader") 
    # text_to_3d("a animated skeleton ") 
    # text_to_3d("tin whistle with wooden mouthpiece ") 
    text_to_3d("charming red barn with weathered wood without any windows or base") 