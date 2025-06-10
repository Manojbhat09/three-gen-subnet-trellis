import gradio as gr
import spaces

import os
import shutil
import random
import uuid
from datetime import datetime
from diffusers import DiffusionPipeline
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
NUM_INFERENCE_STEPS = 8
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Funciones auxiliares
def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)

def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = trellis_pipeline.preprocess_image(image)
    return processed_image

def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    return gs, mesh

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

@spaces.GPU
def generate_flux_image(
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    req: gr.Request,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Image.Image:
    """Generate image using Flux pipeline"""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt = "wbgmsst, " + prompt + ", 3D isometric, white background"
    image = flux_pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=NUM_INFERENCE_STEPS,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{unique_id}.png"
    filepath = os.path.join(user_dir, filename)
    image.save(filepath)
    return image

@spaces.GPU
def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    req: gr.Request,
) -> Tuple[dict, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path

@spaces.GPU(duration=90)
def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("""
    # UTPL - Conversión de Texto a Imagen a objetos 3D usando IA  
    ### Tesis: *"Objetos tridimensionales creados por IA: Innovación en entornos virtuales"*  
    **Autor:** Carlos Vargas  
    **Base técnica:** Adaptación de [TRELLIS](https://trellis3d.github.io/) y [FLUX](https://huggingface.co/camenduru/FLUX.1-dev-diffusers) (herramientas de código abierto para generación 3D)  
    **Propósito educativo:** Demostraciones académicas e Investigación en modelado 3D automático  
    """)
    
    with gr.Row():
        with gr.Column():
            # Flux image generation inputs
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your game asset description")
            with gr.Accordion("Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    width = gr.Slider(512, 1024, label="Width", value=1024, step=16)
                    height = gr.Slider(512, 1024, label="Height", value=1024, step=16)
                with gr.Row():
                    guidance_scale = gr.Slider(0.0, 10.0, label="Guidance Scale", value=3.5, step=0.1)
            # Botones separados
            generate_image_btn = gr.Button("Generar Imagen")
            generate_video_btn = gr.Button("Generar Video", interactive=False)
        with gr.Column():
            generated_image = gr.Image(label="Generated Asset", type="pil")
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True)
        model_output = gr.Model3D(label="Extracted GLB/Gaussian", height=400)
    
    with gr.Row():
        extract_glb_btn = gr.Button("Extract GLB", interactive=False)
    
    with gr.Row():
        download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
    
    # Variables adicionales para la generación 3D
    with gr.Accordion("3D Generation Settings", open=False):
        gr.Markdown("Stage 1: Sparse Structure Generation")
        with gr.Row():
            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
        gr.Markdown("Stage 2: Structured Latent Generation")
        with gr.Row():
            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
    
    # Variables para la extracción de GLB
    with gr.Accordion("GLB Extraction Settings", open=False):
        mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
        texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
    
    output_buf = gr.State()
    
    # Event handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    # Generar imagen
    generate_image_btn.click(
        generate_flux_image,
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale],
        outputs=[generated_image]
    ).then(
        lambda: gr.Button(interactive=True),  
        outputs=[generate_video_btn],
    )
    
    # Generar video
    generate_video_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        preprocess_image,
        inputs=[generated_image],
        outputs=[generated_image],
    ).then(
        image_to_3d,
        inputs=[
            generated_image, 
            seed, 
            ss_guidance_strength, 
            ss_sampling_steps, 
            slat_guidance_strength, 
            slat_sampling_steps
        ],
        outputs=[output_buf, video_output],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[extract_glb_btn],
    )
    
    video_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[extract_glb_btn],
    )
    
    # Extraer GLB
    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )
    
    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )

# Initialize both pipelines
if __name__ == "__main__":
    from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
    from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF
    
    # Initialize Flux pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    dtype = torch.bfloat16
    file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
    file_url = file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
    single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
    quantization_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(single_file_base_model, subfolder="text_encoder_2", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config_tf, token=huggingface_token)
    
    if ".gguf" in file_url:
        transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=single_file_base_model)
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, token=huggingface_token)
        transformer = FluxTransformer2DModel.from_single_file(file_url, subfolder="transformer", torch_dtype=dtype, config=single_file_base_model, quantization_config=quantization_config, token=huggingface_token)
    
    flux_pipeline = FluxPipeline.from_pretrained(single_file_base_model, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=dtype, token=huggingface_token)
    flux_pipeline.to("cuda")
    
    # Initialize Trellis pipeline
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("cavargas10/TRELLIS")
    trellis_pipeline.cuda()
    
    try:
        trellis_pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
    except:
        pass
    
    demo.launch(show_error=True)