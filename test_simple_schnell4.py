import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderTiny
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image

CHECKPOINT = "manbeast3b/Flux.1.schnell-quant2"
REVISION = "44eb293715147878512da10bf3bc47cd14ec8c55"
TinyVAE = "madebyollin/taef1"
TinyVAE_REV = "2d552378e58c9c94201075708d7de4e1163b2689"

def load_pipeline():
    # Load transformer from local cache directory
    transformer_path = os.path.join(
        HF_HUB_CACHE,
        "models--manbeast3b--Flux.1.schnell-quant2",
        "snapshots", REVISION,
        "transformer"
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        transformer_path,
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )

    # Load VAE model
    vae = AutoencoderTiny.from_pretrained(
        TinyVAE,
        revision=TinyVAE_REV,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )

    # Load main FLUX pipeline with transformer and vae
    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        transformer=transformer,
        vae=vae,
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipeline.to(memory_format=torch.channels_last)
    return pipeline

def test_generate():
    pipeline = load_pipeline()
    prompt = "A beautiful sunrise over the mountains"
    image = pipeline(prompt, guidance_scale=7.5, num_inference_steps=10).images[0]
    image.save("test_output.png")
    print("Image saved as test_output.png")

if __name__ == "__main__":
    test_generate()
