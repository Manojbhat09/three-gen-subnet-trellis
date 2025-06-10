import os
import shlex
import spaces
import subprocess
def install_cuda_toolkit():
    CUDA_TOOLKIT_URL = "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
    CUDA_TOOLKIT_FILE = "/tmp/%s" % os.path.basename(CUDA_TOOLKIT_URL)
    subprocess.call(["wget", "-q", CUDA_TOOLKIT_URL, "-O", CUDA_TOOLKIT_FILE])
    subprocess.call(["chmod", "+x", CUDA_TOOLKIT_FILE])
    subprocess.call([CUDA_TOOLKIT_FILE, "--silent", "--toolkit"])

    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "%s/bin:%s" % (os.environ["CUDA_HOME"], os.environ["PATH"])
    os.environ["LD_LIBRARY_PATH"] = "%s/lib:%s" % (
        os.environ["CUDA_HOME"],
        "" if "LD_LIBRARY_PATH" not in os.environ else os.environ["LD_LIBRARY_PATH"],
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
install_cuda_toolkit()
os.system("pip list | grep torch")
os.system('nvcc -V')
print("cd /home/user/app/step1x3d_texture/differentiable_renderer/ && python setup.py install")
os.system("cd /home/user/app/step1x3d_texture/differentiable_renderer/ && python setup.py install")

subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"), check=True)
import time
import uuid
import torch
import trimesh
import argparse
import numpy as np
import gradio as gr
from gradio_client import Client
from PIL import Image
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face


parser = argparse.ArgumentParser()
parser.add_argument(
    "--geometry_model", type=str, default="Step1X-3D-Geometry-Label-1300m"
)
parser.add_argument(
    "--texture_model", type=str, default="Step1X-3D-Texture"
)
parser.add_argument("--cache_dir", type=str, default="cache")
args = parser.parse_args()

os.makedirs(args.cache_dir, exist_ok=True)

geometry_model = Step1X3DGeometryPipeline.from_pretrained(
    "stepfun-ai/Step1X-3D", subfolder=args.geometry_model
).to("cuda")

texture_model = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder=args.texture_model)

# Initialize text-to-image client
t2i_client = Client(os.getenv("H100_3D_URL")) 


def generate_image_from_text(prompt, height, width, steps, scales, seed):
    """Generate image from text using the external API"""
    try:
        result = t2i_client.predict(
            height=height,
            width=width,
            steps=steps,
            scales=scales,
            prompt=prompt,
            seed=seed if seed != -1 else None,
            api_name="/process_and_save_image"
        )
        # Result contains a dict with 'path' key pointing to the generated image
        if isinstance(result, dict) and 'path' in result:
            return result['path']
        elif isinstance(result, str):
            return result
        else:
            raise Exception("Unexpected result format from text-to-image API")
    except Exception as e:
        print(f"Error generating image from text: {e}")
        return None


def get_random_seed():
    """Get a random seed from the external API"""
    try:
        result = t2i_client.predict(api_name="/update_random_seed")
        return result
    except Exception as e:
        print(f"Error getting random seed: {e}")
        return -1


@spaces.GPU(duration=240)
def generate_3d_func(
    input_image_path, guidance_scale, inference_steps, max_facenum, symmetry, edge_type
):
    # geometry_model = geometry_model.to("cuda")
    if "Label" in args.geometry_model:
        symmetry_values = ["x", "asymmetry"]
        out = geometry_model(
            input_image_path,
            label={"symmetry": symmetry_values[int(symmetry)], "edge_type": edge_type},
            guidance_scale=float(guidance_scale),
            octree_resolution=384,
            max_facenum=int(max_facenum),
            num_inference_steps=int(inference_steps),
        )
    else:
        out = geometry_model(
            input_image_path,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(inference_steps),
            max_facenum=int(max_facenum),
        )

    save_name = str(uuid.uuid4())
    print(save_name)
    geometry_save_path = f"{args.cache_dir}/{save_name}.glb"
    geometry_mesh = out.mesh[0]
    geometry_mesh.export(geometry_save_path)

    geometry_mesh = remove_degenerate_face(geometry_mesh)
    geometry_mesh = reduce_face(geometry_mesh)
    textured_mesh = texture_model(input_image_path, geometry_mesh)
    textured_save_path = f"{args.cache_dir}/{save_name}-textured.glb"
    textured_mesh.export(textured_save_path)

    torch.cuda.empty_cache()
    print("Generate finish")
    return geometry_save_path, textured_save_path


def update_image_display(uploaded_image, generated_image):
    """Update the displayed image based on which source has content"""
    if generated_image is not None:
        return generated_image
    elif uploaded_image is not None:
        return uploaded_image
    else:
        return None


with gr.Blocks(title="3D-LLAMA V2") as demo:
    gr.Markdown("# 3D-LLAMA V2 with Step1X-3D")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Image Input")
            with gr.Tab("Upload Image"):
                uploaded_image = gr.Image(label="Upload Image", type="filepath")
            
            with gr.Tab("Generate from Text"):
                text_prompt = gr.Textbox(label="Image Description", placeholder="Enter your image description here...")
                with gr.Row():
                    t2i_height = gr.Slider(label="Height", minimum=512, maximum=2048, value=1024, step=64)
                    t2i_width = gr.Slider(label="Width", minimum=512, maximum=2048, value=1024, step=64)
                with gr.Row():
                    t2i_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=8, step=1)
                    t2i_scales = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, value=3.5, step=0.5)
                with gr.Row():
                    t2i_seed = gr.Number(label="Seed (optional, -1 for random)", value=-1)
                    random_seed_btn = gr.Button("Get Random Seed", scale=0)
                generate_image_btn = gr.Button("Generate Image", variant="primary")
            
            # Display the current working image
            current_image = gr.Image(label="Current Image (for 3D generation)", type="filepath", interactive=False)
            generated_image_path = gr.State(value=None)
            
            gr.Markdown("## 3D Generation Settings")
            guidance_scale = gr.Number(label="3D Guidance Scale", value="7.5")
            inference_steps = gr.Slider(
                label="3D Inference Steps", minimum=1, maximum=100, value=50
            )
            max_facenum = gr.Number(label="Max Face Num", value="400000")
            symmetry = gr.Radio(
                choices=["symmetry", "asymmetry"],
                label="Symmetry Type",
                value="symmetry",
                type="index",
            )
            edge_type = gr.Radio(
                choices=["sharp", "normal", "smooth"],
                label="Edge Type",
                value="sharp",
                type="value",
            )
            btn_3d = gr.Button("Generate 3D", variant="primary")
            
        with gr.Column(scale=4):
            textured_preview = gr.Model3D(label="Textured", height=380)
            geometry_preview = gr.Model3D(label="Geometry", height=380)
            
        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    ["examples/images/000.png"],
                    ["examples/images/001.png"],
                    ["examples/images/004.png"],
                    ["examples/images/008.png"],
                    ["examples/images/028.png"],
                    ["examples/images/032.png"],
                    ["examples/images/061.png"],
                    ["examples/images/107.png"],
                ],
                inputs=[uploaded_image],
                cache_examples=False,
                label="Example Images"
            )

    # Event handlers
    def on_generate_image(prompt, height, width, steps, scales, seed):
        if not prompt:
            gr.Warning("Please enter a text prompt")
            return None, None
        
        generated_path = generate_image_from_text(prompt, height, width, steps, scales, seed)
        if generated_path:
            return generated_path, generated_path
        else:
            gr.Warning("Failed to generate image from text")
            return None, None
    
    def on_upload_image(image_path):
        return image_path
    
    def get_current_image(uploaded, generated):
        if generated is not None:
            return generated
        elif uploaded is not None:
            return uploaded
        else:
            return None
    
    # Connect event handlers
    generate_image_btn.click(
        on_generate_image,
        inputs=[text_prompt, t2i_height, t2i_width, t2i_steps, t2i_scales, t2i_seed],
        outputs=[generated_image_path, current_image]
    )
    
    random_seed_btn.click(
        get_random_seed,
        inputs=[],
        outputs=[t2i_seed]
    )
    
    uploaded_image.change(
        on_upload_image,
        inputs=[uploaded_image],
        outputs=[current_image]
    )
    
    btn_3d.click(
        lambda img, gs, is_, mf, sym, et: generate_3d_func(img, gs, is_, mf, sym, et) if img else (None, None),
        inputs=[
            current_image,
            guidance_scale,
            inference_steps,
            max_facenum,
            symmetry,
            edge_type,
        ],
        outputs=[geometry_preview, textured_preview],
    )

demo.launch(ssr_mode=False)