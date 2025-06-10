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

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from PIL import Image
import os

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'jetx/Hunyuan3D-2'

    image = Image.open(image_path)
    image = image.resize((1024, 1024))

    if image.mode == 'RGB':
        image = rembg(image)

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
        mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                        generator=torch.manual_seed(2025))[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)
        mesh.export('mesh.glb')
        print("Successfully generated and exported mesh.glb")

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
            mesh = pipeline(mesh, image=image)
            mesh.export('texture.glb')
            print("Successfully generated and exported texture.glb")
        except Exception as e:
            print(f"Texture generation failed: {e}")
            print("Continuing with untextured mesh...")
    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise


def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    model_path = 'jetx/Hunyuan3D-2'
    cache_dir = '/home/mbhat/miniconda/envs/hunyuan3d/.cache/huggingface/hub/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled/snapshots/527cf2ecce7c04021975938f8b0e44e35d2b1ed9'

    try:
        # Try loading from local cache first
        print(f"Attempting to load model from cache: {cache_dir}")
        t2i = HunyuanDiTPipeline(cache_dir)
    except Exception as e:
        print(f"Error loading from cache: {e}")
        print("Trying HuggingFace model...")
        try:
            # Try loading from HuggingFace
            t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            raise RuntimeError("Failed to load text-to-image model. Please check your installation and model files.")

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
        print("Generating image from text prompt...")
        image = t2i(prompt)
        print("Image generated successfully")
        
        # Save the original generated image
        original_image_path = 't2i_original.png'
        image.save(original_image_path)
        print(f"Saved original image to {original_image_path}")
        
        print("Removing background...")
        image_no_bg = rembg(image)
        print("Background removed successfully")
        
        # Save the image after background removal
        no_bg_image_path = 't2i_no_bg.png'
        image_no_bg.save(no_bg_image_path)
        print(f"Saved background-removed image to {no_bg_image_path}")
        
        print("Generating 3D mesh...")
        mesh = pipeline(image=image_no_bg, num_inference_steps=30, mc_algo='mc',
                       generator=torch.manual_seed(42))[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)
        mesh.export('t2i_final_1.glb')
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
            mesh.export('t2i_texture.glb')
            print("Successfully generated and exported t2i_texture.glb")
        except Exception as e:
            print(f"Texture generation failed: {e}")
            print("Continuing with untextured mesh...")
    except Exception as e:
        print(f"Error during mesh generation: {e}")
        raise


if __name__ == '__main__':
    # image_to_3d()
    text_to_3d("a blue monkey sitting on temple")
    # text_to_3d("a blue monkey sitting on temple, 3D isometric, white background")
