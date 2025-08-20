import os
import gc
import torch
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from diffusers import FluxPipeline

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True

model_id = "manbeast3b/flux.1-schnell-full1"
model_revision = "cb1b599b0d712b9aab2c4df3ad27b050a27ec146"

gc.collect()
torch.cuda.empty_cache()

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-schnell",
    torch_dtype=torch.bfloat16
)

pipeline = FluxPipeline.from_pretrained(
    model_id,
    revision=model_revision,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipeline.to("cuda", memory_format=torch.channels_last)

# Warmup call
_ = pipeline(
    prompt="A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    output_type="pil"
)

prompts = [
    "A futuristic city skyline at sunset",
    "A mystical forest with glowing plants"
]

for i, prompt in enumerate(prompts):
    gc.collect()
    torch.cuda.empty_cache()
    img = pipeline(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=1024,
        width=1024,
        output_type="pil"
    ).images[0]
    img.save(f"output_{i+1}.png")
    print(f"Saved output_{i+1}.png")
