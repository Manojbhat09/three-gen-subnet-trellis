import requests
import base64
import torch
import sys
import os

# Add validation directory to Python path (append so project root takes precedence)
validation_path = os.path.join(os.path.dirname(__file__), 'validation')
sys.path.append(validation_path)

from validation.engine.validation_engine import ValidationEngine
from validation.engine.data_structures import RequestData, ResponseData
from validation.engine.io.ply import PlyLoader
from validation.engine.rendering.renderer import Renderer
from validation.serve import decode_and_validate_txt
import zstandard

def generate_model(prompt, return_compressed=True):
    url = "http://127.0.0.1:8096/generate/"
    data = {"prompt": prompt, "return_compressed": return_compressed}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.content  # binary data
    else:
        print(f"Generation failed: {response.status_code} - {response.text}")
        return None

def main():
    prompt = "pinkbicycle"
    # Request compressed data (default)
    generated_data = generate_model(prompt, return_compressed=True)
    if not generated_data:
        return

    # Base64 encode the received data
    base64_data = base64.b64encode(generated_data).decode('utf-8')

    # Prepare RequestData - set compression=2 for SPZ
    request = RequestData(
        prompt=prompt,
        data=base64_data,
        compression=2,  # SPZ compression
        generate_preview=False
    )

    # Initialize components
    validator = ValidationEngine()
    validator.load_pipelines()
    zstd_decompressor = zstandard.ZstdDecompressor()
    renderer = Renderer()
    ply_data_loader = PlyLoader()

    # Validate
    result = decode_and_validate_txt(
        request=request,
        ply_data_loader=ply_data_loader,
        renderer=renderer,
        zstd_decompressor=zstd_decompressor,
        validator=validator
    )

    # Print detailed results
    print("Validation Results:")
    print(f"Score: {result.response_data.score}")
    print(f"IQA: {result.response_data.iqa}")
    print(f"Alignment Score: {result.response_data.alignment_score}")
    print(f"SSIM: {result.response_data.ssim}")
    print(f"LPIPS: {result.response_data.lpips}")

    # Cleanup
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 