import os
import sys
import warnings
from PIL import Image

# Filter out specific warnings
warnings.filterwarnings("ignore", message=".*torch.range.*")
warnings.filterwarnings("ignore", message=".*timestep schedule.*")

# Import Hunyuan3D components
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def main():
    # Initialize the Hunyuan3D pipelines
    print("Loading Hunyuan3D pipelines...")
    model_path = 'tencent/Hunyuan3D-2'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Load and process input image
    image_path = 'assets/demo.png'  # Default path
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        print("Please either:")
        print("1. Create an 'assets' directory and place your input image as 'demo.png'")
        print("2. Or modify the 'image_path' variable in the script to point to your image")
        sys.exit(1)
        
    print(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Remove background if image is RGB
    if image.mode == 'RGB':
        print("Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # Generate 3D mesh
    print("Generating 3D mesh...")
    mesh = pipeline_shapegen(image=image)[0]
    
    # Generate texture
    print("Generating texture...")
    mesh = pipeline_texgen(mesh, image=image)
    
    # Save outputs
    print("Saving outputs...")
    
    # Save as GLB file (textured mesh)
    output_path = "outputs/generated_model.glb"
    mesh.export(output_path)
    
    print("\nGeneration complete! Files saved in outputs/:")
    print(f"- {output_path} (Textured mesh)")

if __name__ == "__main__":
    main() 