import os
import torch
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils import BitsAndBytesConfigTF

def test_flux_schnell():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16

    model_repo = "manbeast3b/flux.1-schnell-full1"
    revision_hash = "cb1b599b0d712b9aab2c4df3ad27b050a27ec146"

    print("Loading FLUX text encoder with 8-bit quantization...")
    quant_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)
    text_encoder = T5EncoderModel.from_pretrained(
        model_repo,
        revision=revision_hash,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        quantization_config=quant_config_tf,
        token=huggingface_token,
    )

    print("Loading FLUX transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_repo,
        revision=revision_hash,
        subfolder="transformer",
        torch_dtype=dtype,
        token=huggingface_token
    )

    print("Initializing FLUX pipeline...")
    flux_pipeline = FluxPipeline.from_pretrained(
        model_repo,
        transformer=transformer,
        text_encoder_2=text_encoder,
        torch_dtype=dtype,
        token=huggingface_token,
    ).to(device)

    prompt = "A serene landscape with mountains and a river"
    print(f"Generating image for prompt: '{prompt}'")

    image = flux_pipeline(prompt)
    image.save("output.png")
    print("Image saved to output.png")

if __name__ == "__main__":
    test_flux_schnell()
