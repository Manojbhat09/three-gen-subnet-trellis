import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderTiny
from transformers import T5EncoderModel
from huggingface_hub.constants import HF_HUB_CACHE
import os
import gc



# Clear GPU cache
def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

# Allow unpickling of custom affine quantized tensor
#from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
#torch.serialization.add_safe_globals([AffineQuantizedTensor])

import torch
import torchao.dtypes.affine_quantized_tensor as aqt
import inspect

# Allow all classes/functions in the affine_quantized_tensor module to be safe globals
safe_globals = [obj for name, obj in vars(aqt).items() if inspect.isclass(obj) or inspect.isfunction(obj)]
torch.serialization.add_safe_globals(safe_globals)

def test_load_and_infer():
    empty_cache()
    dtype = torch.bfloat16
    device = "cuda"

    # Load text encoder
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "city96/t5-v1_1-xxl-encoder-bf16",
        revision="1b9c856aadb864af93c1dcdc226c2774fa67bc86",
        torch_dtype=dtype
    ).to(memory_format=torch.channels_last, device=device)

    # Load VAE model
    vae = AutoencoderTiny.from_pretrained(
        "RobertML/FLUX.1-schnell-vae_fx",
        revision="00c83cdfdfe46992eb0ed45921eee34261fcb56e",
        torch_dtype=dtype
    ).to(device)

    # Load Flux transformer model from local cache path
    path = os.path.join(
        HF_HUB_CACHE,
        "models--RobertML--FLUX.1-schnell-int8wo",
        "snapshots",
        "307e0777d92df966a3c0f99f31a6ee8957a9857a"
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        path,
        torch_dtype=dtype,
        use_safetensors=False,
    ).to(memory_format=torch.channels_last, device=device)

    # Load pipeline
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
        vae=vae,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    ).to(device)

    pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")

    prompt = "A beautiful landscape with mountains and a river at sunrise"

    # Run inference
    for _ in range(3):
        image = pipeline(prompt, width=512, height=512, guidance_scale=0.0, num_inference_steps=4).images[0]

    image.save("output_test.png")
    print("Image saved as output_test.png")

if __name__ == "__main__":
    test_load_and_infer()
