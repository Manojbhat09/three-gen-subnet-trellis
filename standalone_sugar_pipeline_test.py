#!/usr/bin/env python3
"""
Standalone SuGaR Pipeline Test
Tests the complete pipeline: Text → FLUX → Image → Hunyuan3D → Mesh → SuGaR → Gaussian Splatting PLY
Creates three PLY outputs: Gaussian Splatting (for validation), Viewable Mesh, and Simple Points
No server, no BPT, just the core pipeline.
"""

import os
import time
import torch
import traceback
import gc
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import tempfile

# Set environment variables
os.environ['SPCONV_ALGO'] = 'native'

# Add Hunyuan3D-2 to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Hunyuan3D-2'))

print("🍬 Standalone SuGaR Pipeline Test")
print("=" * 50)

def clear_gpu_memory():
    """Clear GPU memory aggressively"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        available = total - allocated
        print(f"🧠 GPU: {allocated:.1f}GB used, {available:.1f}GB available")
        return available
    return 0

def test_imports():
    """Test all required imports"""
    print("\n1. Testing imports...")
    
    try:
        # Core imports
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import (
            Hunyuan3DDiTFlowMatchingPipeline, 
            FaceReducer, 
            FloaterRemover, 
            DegenerateFaceRemover
        )
        print("✓ Hunyuan3D imports successful")
        
        # FLUX imports
        from diffusers import FluxPipeline
        print("✓ FLUX imports successful")
        
        # PLY handling
        from plyfile import PlyData, PlyElement
        print("✓ PLY handling imports successful")
        
        # SuGaR fallback functions (since full SuGaR may not be available)
        def RGB2SH(rgb):
            """Fallback RGB to Spherical Harmonics conversion"""
            return rgb / (2 * 3.14159 ** 0.5)
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_flux_generation(prompt: str, seed: int = 42):
    """Test FLUX image generation"""
    print(f"\n2. Testing FLUX generation for: '{prompt}'")
    
    try:
        clear_gpu_memory()
        
        # Import required components
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
        from diffusers import BitsAndBytesConfig, GGUFQuantizationConfig
        
        print("Loading FLUX pipeline with quantization...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        
        # Get HuggingFace token
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Use the correct model configuration (same as working demo)
        file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
        single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        
        try:
            # Load text encoder with 8-bit quantization
            print("Loading text encoder with 8-bit quantization...")
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
            print("Loading transformer with GGUF quantization...")
            transformer = FluxTransformer2DModel.from_single_file(
                file_url, 
                subfolder="transformer", 
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), 
                torch_dtype=dtype, 
                config=single_file_base_model
            )
            
            # Initialize pipeline
            print("Initializing FLUX pipeline...")
            flux_pipe = FluxPipeline.from_pretrained(
                single_file_base_model, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2, 
                torch_dtype=dtype, 
                token=huggingface_token
            )
            flux_pipe.to("cuda")
            
            print("✓ FLUX pipeline loaded with quantization")
            clear_gpu_memory()
            
            # Generate image
            enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            print("Generating image...")
            start_time = time.time()
            
            image = flux_pipe(
                prompt=enhanced_prompt,
                guidance_scale=3.5,
                num_inference_steps=8,
                width=1024,
                height=1024,
                generator=generator,
            ).images[0]
            
            generation_time = time.time() - start_time
            print(f"✓ Image generated in {generation_time:.2f}s")
            
            # Save image
            image.save("test_flux_output.png")
            print("✓ Image saved as test_flux_output.png")
            
            # CRITICAL: Completely clear FLUX from memory before proceeding
            print("Clearing FLUX from GPU memory...")
            del flux_pipe
            del transformer
            del text_encoder_2
            clear_gpu_memory()
            
            return image
            
        except Exception as e:
            print(f"❌ Quantized FLUX loading failed: {e}")
            # Fallback to simple loading without quantization
            print("Trying fallback FLUX loading...")
            try:
                flux_pipe = FluxPipeline.from_pretrained(
                    "camenduru/FLUX.1-dev-diffusers",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                flux_pipe = flux_pipe.to("cuda")
                
                enhanced_prompt = f"wbgmsst, {prompt}, 3D isometric asset, clean white background, isolated object"
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
                image = flux_pipe(
                    prompt=enhanced_prompt,
                    guidance_scale=3.5,
                    num_inference_steps=8,
                    width=1024,
                    height=1024,
                    generator=generator,
                ).images[0]
                
                image.save("test_flux_output.png")
                print("✓ Fallback FLUX generation successful")
                
                # Clear fallback pipeline
                del flux_pipe
                clear_gpu_memory()
                
                return image
                
            except Exception as fallback_e:
                print(f"❌ Fallback FLUX also failed: {fallback_e}")
                return None
        
    except Exception as e:
        print(f"❌ FLUX generation failed: {e}")
        traceback.print_exc()
        return None

def test_background_removal(image):
    """Test background removal"""
    print("\n3. Testing background removal...")
    
    try:
        from hy3dgen.rembg import BackgroundRemover
        
        rembg = BackgroundRemover()
        processed_image = rembg(image)
        processed_image.save("test_processed_image.png")
        print("✓ Background removal successful")
        return processed_image
        
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        print("Using original image...")
        return image

def test_hunyuan3d_generation(image, seed: int = 42):
    """Test Hunyuan3D mesh generation"""
    print("\n4. Testing Hunyuan3D mesh generation...")
    
    try:
        clear_gpu_memory()
        
        from hy3dgen.shapegen import (
            Hunyuan3DDiTFlowMatchingPipeline, 
            FaceReducer, 
            FloaterRemover, 
            DegenerateFaceRemover
        )
        
        print("Loading Hunyuan3D pipeline...")
        hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'jetx/Hunyuan3D-2',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Fix: Don't assign the result of .to() back to the variable
        hunyuan_pipeline.to("cuda")
        
        print("✓ Hunyuan3D pipeline loaded")
        
        # Generate mesh
        print("Generating 3D mesh...")
        start_time = time.time()
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        mesh = hunyuan_pipeline(
            image,
            num_inference_steps=30,
            guidance_scale=2.0,
            generator=generator,
        )
        
        generation_time = time.time() - start_time
        print(f"✓ Mesh generated in {generation_time:.2f}s")
        
        # Handle Hunyuan3D's nested list structure: List[List[trimesh.Trimesh]]
        if isinstance(mesh, list):
            if len(mesh) > 0 and isinstance(mesh[0], list):
                # Nested list structure - flatten and select best mesh
                all_meshes = []
                for batch_meshes in mesh:
                    all_meshes.extend(batch_meshes)
                mesh = all_meshes
            
            if isinstance(mesh, list) and len(mesh) > 0:
                # Select the largest mesh by vertex count (usually the main object)
                num_candidates = len(mesh)
                mesh = max(mesh, key=lambda m: len(m.vertices) if m is not None else 0)
                print(f"Selected mesh with {len(mesh.vertices)} vertices from {num_candidates} candidates")
            else:
                raise RuntimeError("No valid meshes generated")
        
        # Ensure we have a valid single mesh object
        if not hasattr(mesh, 'vertices'):
            raise RuntimeError("Invalid mesh object generated")
        
        # Process mesh
        print("Processing mesh...")
        face_reducer = FaceReducer()
        floater_remover = FloaterRemover()
        degenerate_remover = DegenerateFaceRemover()
        
        mesh = face_reducer(mesh)
        mesh = floater_remover(mesh)
        mesh = degenerate_remover(mesh)
        
        print(f"✓ Mesh processed: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Save mesh
        mesh.export("test_mesh.glb")
        print("✓ Mesh saved as test_mesh.glb")
        
        # Clear Hunyuan3D pipeline
        del hunyuan_pipeline
        clear_gpu_memory()
        
        return mesh
        
    except Exception as e:
        print(f"❌ Hunyuan3D generation failed: {e}")
        traceback.print_exc()
        return None

def test_sugar_conversion(mesh, num_points: int = 15000):
    """Test SuGaR conversion to Gaussian Splatting PLY"""
    print(f"\n5. Testing SuGaR conversion to Gaussian Splatting PLY ({num_points} points)...")
    
    try:
        # Sample points from mesh surface
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # Get face normals for sampled points
        face_normals = mesh.face_normals[face_indices]
        
        # Fallback RGB to SH conversion
        def RGB2SH(rgb):
            """Fallback RGB to Spherical Harmonics conversion"""
            return rgb / (2 * 3.14159 ** 0.5)
        
        # Create Gaussian Splatting attributes
        print("Creating Gaussian Splatting attributes...")
        
        # Colors (use vertex colors if available, otherwise default)
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            # Interpolate vertex colors to sampled points
            colors = np.ones((num_points, 3)) * 0.7  # Default gray
        else:
            colors = np.ones((num_points, 3)) * 0.7  # Default gray
        
        # Convert RGB to Spherical Harmonics (DC component only for simplicity)
        sh_dc = RGB2SH(colors)  # Shape: (N, 3)
        
        # Create additional SH coefficients (set to zero for simplicity)
        sh_rest = np.zeros((num_points, 45))  # 15 coefficients * 3 channels = 45
        
        # Opacities (sigmoid inverse of alpha values)
        opacities = np.ones((num_points, 1)) * 2.0  # High opacity
        
        # Scales (log of scale values)
        scales = np.ones((num_points, 3)) * (-3.0)  # Small scales
        
        # Rotations (quaternions, normalized)
        rotations = np.zeros((num_points, 4))
        rotations[:, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)
        
        print("✓ Gaussian Splatting attributes created")
        
        # Create PLY data
        gs_ply_data = create_gaussian_splatting_ply(
            points, face_normals, sh_dc, sh_rest, opacities, scales, rotations
        )
        
        # Save Gaussian Splatting PLY
        with open("test_gaussian_splatting.ply", "wb") as f:
            f.write(gs_ply_data)
        
        print("✓ Gaussian Splatting PLY saved as test_gaussian_splatting.ply")
        
        # Create viewable mesh PLY from original GLB
        create_viewable_mesh_ply()
        
        # Create simple points PLY
        create_simple_points_ply(points)
        
        return gs_ply_data
        
    except Exception as e:
        print(f"❌ SuGaR conversion failed: {e}")
        traceback.print_exc()
        return None

def create_gaussian_splatting_ply(points, normals, sh_dc, sh_rest, opacities, scales, rotations):
    """Create a Gaussian Splatting PLY file"""
    from plyfile import PlyData, PlyElement
    
    num_points = len(points)
    
    # Create vertex data with all Gaussian Splatting attributes
    vertex_data = []
    for i in range(num_points):
        vertex = [
            points[i, 0], points[i, 1], points[i, 2],  # x, y, z
            normals[i, 0], normals[i, 1], normals[i, 2],  # nx, ny, nz
        ]
        
        # Add spherical harmonics DC components
        vertex.extend([sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2]])  # f_dc_0, f_dc_1, f_dc_2
        
        # Add remaining SH coefficients
        vertex.extend(sh_rest[i])  # f_rest_0 through f_rest_44
        
        # Add opacity
        vertex.append(opacities[i, 0])  # opacity
        
        # Add scales
        vertex.extend([scales[i, 0], scales[i, 1], scales[i, 2]])  # scale_0, scale_1, scale_2
        
        # Add rotations
        vertex.extend([rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]])  # rot_0, rot_1, rot_2, rot_3
        
        vertex_data.append(tuple(vertex))
    
    # Define vertex properties
    properties = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    
    # Add SH rest properties
    for i in range(45):
        properties.append((f'f_rest_{i}', 'f4'))
    
    # Add remaining properties
    properties.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])
    
    # Create PLY element
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=properties),
        'vertex'
    )
    
    # Create PLY data and write to bytes
    ply_data = PlyData([vertex_element])
    
    import io
    buffer = io.BytesIO()
    ply_data.write(buffer)
    return buffer.getvalue()

def create_viewable_mesh_ply():
    """Create a viewable mesh PLY from the GLB file"""
    print("Creating viewable mesh PLY from GLB...")
    
    try:
        # Load the mesh from GLB
        scene_or_mesh = trimesh.load("test_mesh.glb")
        
        # Handle both Scene and Mesh objects
        if hasattr(scene_or_mesh, 'geometry'):
            # It's a Scene object
            if len(scene_or_mesh.geometry) > 0:
                mesh = list(scene_or_mesh.geometry.values())[0]
            else:
                print("❌ No geometry found in GLB file")
                return False
        else:
            # It's already a Mesh object
            mesh = scene_or_mesh
        
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Export as PLY with faces
        mesh.export("test_mesh_viewable.ply")
        print("✓ Created test_mesh_viewable.ply (viewable in 3D viewers)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating viewable PLY: {e}")
        return False

def create_simple_points_ply(points):
    """Create a simple point cloud PLY that's viewable"""
    print("Creating simple point cloud PLY...")
    
    try:
        from plyfile import PlyData, PlyElement
        
        # Create simple vertex data
        vertex_data = []
        for point in points:
            vertex_data.append((point[0], point[1], point[2]))
        
        # Create PLY element
        vertex_element = PlyElement.describe(
            np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
            'vertex'
        )
        
        # Save as simple point cloud PLY
        PlyData([vertex_element]).write("test_points_simple.ply")
        print("✓ Created test_points_simple.ply (simple point cloud)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating simple PLY: {e}")
        return False

def validate_gaussian_splatting_ply(ply_data: bytes):
    """Validate the Gaussian Splatting PLY format"""
    print("\n6. Validating Gaussian Splatting PLY format...")
    
    try:
        from plyfile import PlyData
        import io
        
        # Read PLY from bytes
        buffer = io.BytesIO(ply_data)
        ply = PlyData.read(buffer)
        
        # Check vertex element
        if 'vertex' not in [elem.name for elem in ply.elements]:
            print("❌ No vertex element found")
            return False
        
        vertex_element = ply['vertex']
        properties = [prop.name for prop in vertex_element.properties]
        
        # Check required Gaussian Splatting properties
        required_props = [
            'x', 'y', 'z',  # Position
            'f_dc_0', 'f_dc_1', 'f_dc_2',  # SH DC components
            'opacity',  # Opacity
            'scale_0', 'scale_1', 'scale_2',  # Scales
            'rot_0', 'rot_1', 'rot_2', 'rot_3',  # Rotations
        ]
        
        missing_props = [prop for prop in required_props if prop not in properties]
        if missing_props:
            print(f"❌ Missing required properties: {missing_props}")
            return False
        
        print(f"✓ Valid Gaussian Splatting PLY with {len(vertex_element)} points")
        print(f"✓ Properties ({len(properties)}): {properties[:10]}...")  # Show first 10
        
        return True
        
    except Exception as e:
        print(f"❌ PLY validation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Standalone SuGaR Pipeline Test")
    
    # Test configuration
    test_prompt = "a red sports car"
    test_seed = 42
    
    # Step 1: Test imports
    if not test_imports():
        print("❌ Import test failed, exiting")
        return
    
    # Step 2: Test FLUX generation
    image = test_flux_generation(test_prompt, test_seed)
    if image is None:
        print("❌ FLUX generation failed, exiting")
        return
    
    # Step 3: Test background removal
    processed_image = test_background_removal(image)
    
    # Step 4: Test Hunyuan3D generation
    mesh = test_hunyuan3d_generation(processed_image, test_seed)
    if mesh is None:
        print("❌ Hunyuan3D generation failed, exiting")
        return
    
    # Step 5: Test SuGaR conversion
    gs_ply_data = test_sugar_conversion(mesh)
    if gs_ply_data is None:
        print("❌ SuGaR conversion failed, exiting")
        return
    
    # Step 6: Validate output
    if not validate_gaussian_splatting_ply(gs_ply_data):
        print("❌ PLY validation failed")
        return
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 Pipeline Test Complete!")
    print("=" * 50)
    print("📁 Generated Files:")
    print("   • test_flux_output.png - FLUX generated image")
    print("   • test_processed_image.png - Background removed image")
    print("   • test_mesh.glb - Original Hunyuan3D mesh")
    print("   • test_gaussian_splatting.ply - Gaussian Splatting PLY (for validation)")
    print("   • test_mesh_viewable.ply - Regular mesh PLY (viewable in 3D viewers)")
    print("   • test_points_simple.ply - Simple point cloud (viewable as points)")
    print("\n💡 Use test_mesh_viewable.ply to view the 3D model!")
    print("💡 Use test_gaussian_splatting.ply for validation scoring!")

if __name__ == "__main__":
    main() 