import os
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline

# Model details
CHECKPOINT = "manbeast3b/Flux.1.schnell-quant2"
REVISION = "44eb293715147878512da10bf3bc47cd14ec8c55"

# Use your Hugging Face cache directory or default location
HF_HUB_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

def load_pipeline():
    # Construct local path to transformer's checkpoint files in the cache
    path = os.path.join(
        HF_HUB_CACHE,
        "models--manbeast3b--Flux.1.schnell-quant2",
        "snapshots",
        REVISION,
        "transformer"
    )
    print(f"Loading transformer from local path: {path}")
    
    # Load the Flux transformer model from local files only
    transformer = FluxTransformer2DModel.from_pretrained(
        path,
        use_safetensors=False,
        #local_files_only=True,
        torch_dtype=torch.bfloat16
    )

    # You will also need the VAE model; load similarly or from repo
    #vae = None
    #vae_path = os.path.join(
    #    HF_HUB_CACHE,
    #    "models--madebyollin--taef1",
    #    "snapshots",
    #    "2d552378e58c9c94201075708d7de4e1163b2689",
    #    ""
    #)
    #print(f"Loading VAE from local path {vae_path} or change as needed")
    # For example, you could do:
    # from diffusers import AutoencoderKL
    # vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, torch_dtype=torch.bfloat16)

    # Load the FLUX pipeline with transformer and VAE
    pipeline = FluxPipeline.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        transformer=transformer,
        #vae=vae,  # supply your vae here if loaded
        #local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Memory format optimization
    pipeline.to(memory_format=torch.channels_last)

    return pipeline

if __name__ == "__main__":
    pipe = load_pipeline()
    prompt = "A serene landscape with mountains and a river"
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt).images[0]
    image.save("output.png")
    print("Image saved as output.png")
