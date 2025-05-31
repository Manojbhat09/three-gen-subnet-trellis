import os
import torch
from omegaconf import OmegaConf
from generation.DreamGaussianLib.GaussianProcessor import GaussianProcessor
from generation.DreamGaussianLib.ModelsPreLoader import preload_model

def main():
    # Load the text-to-3D config
    config_path = "generation/configs/text_mv.yaml"
    # config_path = "generation/configs/imagedream.yaml"
    opt = OmegaConf.load(config_path)
    
    # Set your text prompt
    prompt = "a blue monkey sitting on temple"
    opt.prompt = prompt
    
    # Set output directory
    opt.outdir = "outputs"
    os.makedirs(opt.outdir, exist_ok=True)
    
    # Initialize the models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = preload_model(opt, device)
    
    # Initialize the Gaussian processor
    gaussian_processor = GaussianProcessor(opt, prompt)
    
    # Train the model
    gaussian_processor.train(models, opt.iters)
    
    # Save the generated 3D model
    output_path = os.path.join(opt.outdir, "blue_monkey.ply")
    with open(output_path, "wb") as f:
        gaussian_processor.get_gs_model().save_ply(f)
    
    print(f"Generated 3D model saved to: {output_path}")

if __name__ == "__main__":
    main() 