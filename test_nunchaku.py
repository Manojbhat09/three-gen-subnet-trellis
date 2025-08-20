import os
import gc
import torch
from diffusers import FluxPipeline
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
#from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from torch import Generator
from PIL import Image

# Environment and torch configs from your reference
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

# Model and revision specifics from your reference
CHECKPOINT = "black-forest-labs/FLUX.1-schnell"
REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"

def load_pipeline():
    model_id = "manbeast3b/flux.1-schnell-full1"
    model_revision = "cb1b599b0d712b9aab2c4df3ad27b050a27ec146"

    # Load transformer pretrained model directory from HF cache path
    hub_model_dir = os.path.join(
        torch.hub._get_torch_home(),
        #"diffusers",  # Adjust path for your cache location if needed
        f"models--{model_id.replace('/', '--')}",
        "snapshots",
        model_revision,
        "transformer"
    )
    print(hub_model_dir)
    # Load transformer
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(hub_model_dir, torch_dtype=torch.bfloat16)

    # Load diffusion pipeline with the transformer
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        revision=model_revision,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    #pipeline = apply_cache_on_pipe(pipeline, residual_diff_threshold=0.56)
    pipeline.to("cuda", memory_format=torch.channels_last)

    # Warm up call
    _ = pipeline(prompt="A cat holding a sign that says hello world", width=1024, height=1024, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256, output_type="pil")

    return pipeline

@torch.no_grad()
def infer(pipeline, prompt, generator):
    # Clear cache, reset memory tracking for first sample inference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    image = pipeline(
        prompt,
        generator=generator,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=1024,
        width=1024,
        output_type="pil"
    ).images[0]
    return image

if __name__ == "__main__":
    pipeline = load_pipeline()
    generator = Generator(device="cuda").manual_seed(42)

    prompts = [
        "A scenic mountain valley at sunset",
        "An astronaut cat exploring space"
    ]

    for i, prompt in enumerate(prompts):
        img = infer(pipeline, prompt, generator)
        img.save(f"output_image_{i+1}.png")
        print(f"Saved output_image_{i+1}.png")
