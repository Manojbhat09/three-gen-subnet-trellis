# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

import torch
from PIL import Image
import os
import sys

# Add the parent directory to sys.path to allow importing from hy3dgen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def image_to_3d(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    shape_model_path = 'jetx/Hunyuan3D-2'
    texture_model_path = 'jetx/Hunyuan3D-2-texture'  # Separate path for texture model

    image = Image.open(image_path)
    image = image.resize((1024, 1024))

    if image.mode == 'RGB':
        image = rembg(image)

    try:
        # Load shape generation pipeline
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(shape_model_path, use_safetensors=True)
        actual_model_path = pipeline.kwargs['from_pretrained_kwargs']['model_path']
    except Exception as e:
        print(f"Error loading model with safetensors: {e}")
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(shape_model_path, use_safetensors=False)
            actual_model_path = pipeline.kwargs['from_pretrained_kwargs']['model_path']
        except Exception as e:
            print(f"Error loading model without safetensors: {e}")
            raise RuntimeError("Failed to load model. Please check your installation and model files.")

    try:
        # Generate mesh
        mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                       generator=torch.manual_seed(2025))[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)
        mesh.export('mesh.glb')
        print("Successfully generated and exported mesh.glb")

        # Generate texture
        try:
            print("Starting texture generation...")
            print(f"Using texture model path: {texture_model_path}")
            
            # Check for required dependencies
            try:
                import dataclasses_json
            except ImportError:
                print("dataclasses_json is required for texture generation. Please install it with: pip install dataclasses-json")
                print("Continuing with untextured mesh...")
                return

            # Load the UNet2p5DConditionModel
            try:
                unet = UNet2p5DConditionModel.from_pretrained(
                    texture_model_path,
                    torch_dtype=torch.float16  # Use float16 for better memory efficiency
                )
                
                # Create the texture generation pipeline with the custom UNet
                pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    texture_model_path,
                    unet=unet
                )
                
                # Generate texture
                mesh = pipeline(mesh, image=image)
                mesh.export('texture.glb')
                print("Successfully generated and exported texture.glb")
                
            except Exception as e:
                print(f"Error during texture generation: {e}")
                print("Continuing with untextured mesh...")
                
        except Exception as e:
            print(f"Texture generation failed: {e}")
            print("Continuing with untextured mesh...")
            
    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise


def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled')
    shape_model_path = 'jetx/Hunyuan3D-2'
    texture_model_path = 'jetx/Hunyuan3D-2-texture'  # Separate path for texture model
    
    # Load shape generation pipeline
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(shape_model_path)

    # Generate image from text
    image = t2i(prompt)
    image = rembg(image)
    
    # Generate mesh
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_final_1.glb')
    
    # Generate texture
    try:
        print("Starting texture generation for text-to-3D...")
        print(f"Using texture model path: {texture_model_path}")
        
        # Load the UNet2p5DConditionModel
        unet = UNet2p5DConditionModel.from_pretrained(
            texture_model_path,
            torch_dtype=torch.float16
        )
        
        # Create the texture generation pipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            texture_model_path,
            unet=unet
        )
        
        # Generate texture
        mesh = pipeline(mesh, image=image)
        mesh.export('t2i_final_1_textured.glb')
        print("Successfully generated and exported textured mesh")
        
    except Exception as e:
        print(f"Texture generation failed: {e}")
        print("Continuing with untextured mesh...")


if __name__ == '__main__':
    image_to_3d()
    # text_to_3d("a blue monkey sitting on temple") 