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
import trimesh
import yaml
import random
os.environ['SPCONV_ALGO'] = 'native'
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover

# Import Flux components
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF

# Import BPT components
from hy3dgen.shapegen.bpt.model.model import MeshTransformer
from hy3dgen.shapegen.bpt.utils import Dataset, apply_normalize, sample_pc, joint_filter
from hy3dgen.shapegen.bpt.model.serializaiton import BPT_deserialize
from torch.utils.data import DataLoader

# Constants
NUM_INFERENCE_STEPS = 8
MAX_SEED = np.iinfo(np.int32).max

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def create_bpt_config():
    """Create BPT model configuration"""
    config = {
        'dim': 1024,
        'max_seq_len': 8192,
        'flash_attn': True,
        'attn_depth': 24,
        'attn_dim_head': 64,
        'attn_heads': 16,
        'attn_kwargs': {
            'ff_glu': True,
            'num_mem_kv': 4,
            'attn_qk_norm': True,
        },
        'dropout': 0.0,
        'pad_id': -1,
        'coor_continuous_range': (-1., 1.),
        'num_discrete_coors': 128,  # 2^7
        'block_size': 8,
        'offset_size': 16,
        'mode': 'vertices',
        'special_token': -2,
        'use_special_block': True,
        'conditioned_on_pc': True,
        'encoder_name': 'miche-256-feature',
        'encoder_freeze': False,
        'cond_dim': 768
    }
    return config

def load_bpt_model(model_path=None, config=None, device="cuda"):
    """Load BPT model for mesh enhancement"""
    if config is None:
        config = create_bpt_config()
    
    # Initialize model
    model = MeshTransformer(**config)
    
    # Load pretrained weights if available
    if model_path is None:
        # Use the default BPT weights path, making it relative to the script's location
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "hy3dgen" / "shapegen" / "bpt" / "weights" / "bpt-8-16-500m.pt"
    
    if os.path.exists(model_path):
        print(f"Loading BPT model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: BPT model weights not found at {model_path}, using randomly initialized weights")
    
    model.to(device)
    model.eval()
    return model

def enhance_mesh_with_bpt(mesh_path, bpt_model, device="cuda", temperature=0.5, batch_size=1):
    """Enhance mesh using BPT to generate high-detail version"""
    print("Enhancing mesh with BPT...")
    
    # Create dataset from the mesh
    dataset = Dataset(input_type='mesh', input_list=[str(mesh_path)])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    enhanced_meshes = []
    
    with torch.no_grad():
        for batch in dataloader:
            pc_normal = batch['pc_normal'].to(device)  # [B, 4096, 6]
            uid = batch['uid']
            
            print(f"Processing mesh: {uid[0]}")
            
            # Generate enhanced mesh using BPT
            try:
                # Use the generate method from BPT model
                codes = bpt_model.generate(
                    pc=pc_normal,
                    batch_size=batch_size,
                    temperature=temperature,
                    filter_logits_fn=joint_filter,
                    filter_kwargs={'k': 50, 'p': 0.95},
                    max_seq_len=bpt_model.max_seq_len,
                    cache_kv=True,
                )
                
                print(f"Generated codes shape: {codes.shape}")
                
                # Deserialize the codes to mesh faces
                try:
                    coordinates = BPT_deserialize(
                        codes.cpu().numpy().flatten(),
                        block_size=bpt_model.block_size,
                        offset_size=bpt_model.offset_size,
                        compressed=True,
                        special_token=-2,
                        use_special_block=bpt_model.use_special_block
                    )
                    
                    # Convert coordinates to trimesh object
                    if len(coordinates) > 0 and len(coordinates) % 3 == 0:
                        # BPT_deserialize returns coordinates for triangular faces
                        # Reshape to get individual triangle vertices
                        vertices = coordinates.reshape(-1, 3)
                        # Create faces by grouping every 3 vertices into triangles
                        num_faces = len(vertices) // 3
                        faces = np.arange(len(vertices)).reshape(num_faces, 3)
                        
                        # Create mesh
                        enhanced_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        enhanced_mesh = apply_normalize(enhanced_mesh)
                        enhanced_meshes.append((enhanced_mesh, uid[0]))
                        print(f"Enhanced mesh created with {len(vertices)} vertices and {len(faces)} faces")
                    else:
                        print("Invalid coordinates generated, using original mesh")
                        enhanced_meshes.append((None, uid[0]))
                    
                except Exception as e:
                    print(f"Error deserializing mesh: {e}")
                    enhanced_meshes.append((None, uid[0]))
                    
            except Exception as e:
                print(f"Error generating enhanced mesh: {e}")
                enhanced_meshes.append((None, uid[0]))
    
    return enhanced_meshes

def text_to_3d_with_bpt(prompt='a car', output_dir='outputs_bpt', seed=42, use_bpt=True, bpt_temperature=0.5):
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
            
            print("Generating initial 3D mesh with Hunyuan3D...")
            mesh = pipeline(image=image_no_bg, num_inference_steps=30, mc_algo='mc',
                          generator=torch.manual_seed(seed))[0]
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh)
            
            # Save initial mesh
            initial_mesh_path = output_path / 't2i_initial.glb'
            mesh.export(initial_mesh_path)
            print(f"Successfully generated initial mesh: {initial_mesh_path}")
            print(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Clear Hunyuan3D pipeline memory
            del pipeline
            clear_gpu_memory()
            
            # Enhance with BPT if requested
            if use_bpt:
                print("Loading BPT model for mesh enhancement...")
                try:
                    bpt_model = load_bpt_model(device=device)
                    
                    # Convert GLB to OBJ for BPT processing
                    temp_obj_path = output_path / 't2i_temp.obj'
                    mesh.export(temp_obj_path)
                    
                    enhanced_meshes = enhance_mesh_with_bpt(
                        temp_obj_path, 
                        bpt_model, 
                        device=device, 
                        temperature=bpt_temperature,
                        batch_size=1
                    )
                    
                    if enhanced_meshes and enhanced_meshes[0][0] is not None:
                        enhanced_mesh = enhanced_meshes[0][0]
                        enhanced_mesh_path = output_path / 't2i_enhanced_bpt.glb'
                        enhanced_mesh.export(enhanced_mesh_path)
                        print(f"Successfully generated BPT enhanced mesh: {enhanced_mesh_path}")
                        print(f"Enhanced mesh: {len(enhanced_mesh.vertices)} vertices, {len(enhanced_mesh.faces)} faces")
                    else:
                        print("BPT enhancement failed, keeping original mesh")
                    
                    # Clean up temporary file
                    temp_obj_path.unlink()
                    
                    # Clear BPT model memory
                    del bpt_model
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"BPT enhancement failed: {e}")
                    print("Continuing with Hunyuan3D mesh only...")
            
            # Try texture generation
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                print("Starting texture generation...")
                try:
                    import dataclasses_json
                except ImportError:
                    print("dataclasses_json is required for texture generation. Please install it with: pip install dataclasses-json")
                    print("Continuing with untextured mesh...")
                    return
                    
                paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(actual_model_path)
                textured_mesh = paint_pipeline(mesh, image=image_no_bg)
                textured_mesh.export(output_path / 't2i_textured.glb')
                print("Successfully generated and exported t2i_textured.glb")
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
    # Examples with BPT enhancement
    # text_to_3d_with_bpt("a blue monkey sitting on temple", use_bpt=True, bpt_temperature=0.5) 
    # text_to_3d_with_bpt("woman monkey with red breasts", use_bpt=True, bpt_temperature=0.3) 
    # text_to_3d_with_bpt("football", use_bpt=True, bpt_temperature=0.4) 
    text_to_3d_with_bpt("purple durable robotic arm", use_bpt=True, bpt_temperature=0.5, seed=342) 
    # text_to_3d_with_bpt("3 eyed, red and green mystical creature with red horns guarding a big gate with spikes on it", seed=121, use_bpt=True) 
    # text_to_3d_with_bpt("flying dragon", use_bpt=True)
    # text_to_3d_with_bpt("a knight", use_bpt=True) 
    # text_to_3d_with_bpt("a realistic darth vader", use_bpt=True) 
    # text_to_3d_with_bpt("a animated skeleton", use_bpt=True) 
    # text_to_3d_with_bpt("tin whistle with wooden mouthpiece", use_bpt=True) 
    # text_to_3d_with_bpt("charming red barn with weathered wood without any windows or base", use_bpt=True, bpt_temperature=0.4) 