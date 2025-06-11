import os
import torch
from safetensors.torch import save_file
from tqdm import tqdm

def convert_bin_to_safetensors(model_dir):
    """Convert all .bin files in a directory to safetensors format."""
    print(f"Converting files in {model_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.bin'):
                bin_path = os.path.join(root, file)
                safetensors_path = bin_path.replace('.bin', '.safetensors')
                
                # Skip if safetensors already exists
                if os.path.exists(safetensors_path):
                    print(f"Skipping {file} - safetensors already exists")
                    continue
                
                print(f"Converting {file} to safetensors...")
                try:
                    # Load the state dict from .bin file
                    state_dict = torch.load(bin_path, map_location='cpu')
                    
                    # Save as safetensors
                    save_file(state_dict, safetensors_path)
                    print(f"Successfully converted {file} to safetensors")
                except Exception as e:
                    print(f"Error converting {file}: {str(e)}")

def convert_text_encoder(model_dir):
    """Convert text encoder model files."""
    text_encoder_dir = os.path.join(model_dir, 'text_encoder')
    if not os.path.exists(text_encoder_dir):
        print(f"Text encoder directory not found: {text_encoder_dir}")
        return

    # Convert pytorch_model.bin to model.safetensors
    bin_path = os.path.join(text_encoder_dir, 'pytorch_model.bin')
    safetensors_path = os.path.join(text_encoder_dir, 'model.safetensors')
    
    if os.path.exists(bin_path):
        print(f"Converting text encoder model...")
        try:
            state_dict = torch.load(bin_path, map_location='cpu')
            save_file(state_dict, safetensors_path)
            print(f"Successfully converted text encoder model to safetensors")
        except Exception as e:
            print(f"Error converting text encoder model: {str(e)}")
    else:
        print(f"Text encoder model file not found: {bin_path}")

def main():
    # Use the HuggingFace cache path - updated for current environment
    model_dir = '/root/.cache/huggingface/hub/models--jetx--Hunyuan3D-2/snapshots/bdc739e6add05a2393532a3b893c95f466c17cc3'
    
    # Convert files in both model directories
    delight_dir = os.path.join(model_dir, 'hunyuan3d-delight-v2-0')
    paint_dir = os.path.join(model_dir, 'hunyuan3d-paint-v2-0')
    
    if os.path.exists(delight_dir):
        print("\nConverting delight model files...")
        convert_bin_to_safetensors(delight_dir)
        convert_text_encoder(delight_dir)
    else:
        print(f"Delight model directory not found: {delight_dir}")
    
    if os.path.exists(paint_dir):
        print("\nConverting paint model files...")
        convert_bin_to_safetensors(paint_dir)
        convert_text_encoder(paint_dir)
    else:
        print(f"Paint model directory not found: {paint_dir}")

if __name__ == "__main__":
    main() 