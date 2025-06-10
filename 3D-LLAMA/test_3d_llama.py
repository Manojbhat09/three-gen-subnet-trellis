import os
import sys
import torch
import trimesh
from PIL import Image
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face

def main():
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize the models
    print("Loading 3D-LLAMA models...")
    geometry_model = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", 
        subfolder="Step1X-3D-Geometry-Label-1300m"
    ).to("cuda")
    
    texture_model = Step1X3DTexturePipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", 
        subfolder="Step1X-3D-Texture"
    )
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    print(f"Generating 3D model from prompt: {prompt}")
    
    # First, we need to generate an image from the text prompt
    # For this example, we'll use a placeholder image
    # In a real scenario, you would need to integrate with a text-to-image model
    input_image_path = "examples/images/000.png"  # Replace with your image generation pipeline
    
    # Generate the 3D model
    print("Generating 3D model...")
    geometry_output = geometry_model(
        input_image_path,
        label={"symmetry": "asymmetry", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=384,
        max_facenum=400000,
        num_inference_steps=50,
    )
    
    # Save the geometry mesh
    geometry_mesh = geometry_output.mesh[0]
    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh)
    geometry_save_path = "outputs/blue_monkey_geometry.glb"
    geometry_mesh.export(geometry_save_path)
    
    # Generate and save the textured mesh
    print("Generating textured mesh...")
    textured_mesh = texture_model(input_image_path, geometry_mesh)
    textured_save_path = "outputs/blue_monkey_textured.glb"
    textured_mesh.export(textured_save_path)
    
    print("\nGeneration complete! Files saved in outputs/:")
    print("- blue_monkey_geometry.glb (Geometry mesh)")
    print("- blue_monkey_textured.glb (Textured mesh)")

if __name__ == "__main__":
    main() 