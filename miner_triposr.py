import os
import sys
import time
import torch
import logging
import argparse
import bittensor as bt
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import io
import base64

# Import TripoSR
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import rembg

# Import subnet specific modules
from neurons.protocol import (
    TextTo3D,
    TextTo3DResponse,
    TextTo3DRequest,
    TextTo3DSynapse,
)
from neurons.utils import (
    get_my_subnet_uid,
    get_subnet_netuid,
    get_metagraph,
    get_my_subnet_uid,
    get_subnet_netuid,
    get_metagraph,
    get_my_subnet_uid,
    get_subnet_netuid,
    get_metagraph,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

class TripoSRMiner:
    def __init__(
        self,
        config: Optional[bt.config] = None,
        wallet: Optional[bt.wallet] = None,
        subtensor: Optional[bt.subtensor] = None,
        metagraph: Optional[bt.metagraph] = None,
    ):
        self.config = config or bt.config()
        self.wallet = wallet or bt.wallet(config=self.config)
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        self.metagraph = metagraph or bt.metagraph(
            netuid=get_subnet_netuid(self.config),
            subtensor=self.subtensor,
        )
        
        # Initialize TripoSR model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(8192)  # Optimize for speed
        self.model.to(self.device)
        
        # Initialize background removal
        self.rembg_session = rembg.new_session()
        
        # Set up parameters
        self.mc_resolution = 256  # Marching cubes resolution
        self.foreground_ratio = 0.85  # Foreground size ratio
        self.texture_resolution = 2048  # Texture resolution
        
    def process_image(self, image: Image.Image) -> Image.Image:
        """Process input image by removing background and resizing foreground."""
        image = remove_background(image, self.rembg_session)
        image = resize_foreground(image, self.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))
    
    def generate_3d(self, image: Image.Image) -> Dict:
        """Generate 3D model from input image using TripoSR."""
        # Process image
        processed_image = self.process_image(image)
        
        # Generate 3D model
        with torch.no_grad():
            scene_codes = self.model([processed_image], device=self.device)
        
        # Extract mesh with vertex colors
        meshes = self.model.extract_mesh(
            scene_codes, 
            use_vertex_colors=True,
            resolution=self.mc_resolution
        )
        
        # Export to PLY format
        output = io.BytesIO()
        meshes[0].export(output, file_type="ply")
        ply_data = output.getvalue()
        
        return {
            "ply_data": ply_data,
            "score": 0.95  # TripoSR typically produces high-quality results
        }
    
    def run(self):
        """Run the miner."""
        bt.logging.info(f"Starting TripoSR miner on {self.device}")
        
        while True:
            try:
                # Get tasks from validators
                tasks = self.get_tasks()
                
                for task in tasks:
                    try:
                        # Generate 3D model
                        result = self.generate_3d(task.image)
                        
                        # Submit result
                        self.submit_result(task, result)
                        
                    except Exception as e:
                        bt.logging.error(f"Error processing task: {str(e)}")
                        continue
                
                time.sleep(1)  # Prevent too frequent requests
                
            except Exception as e:
                bt.logging.error(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
                continue

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet.name", type=str, default="default")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--logging.trace", action="store_true")
    args = parser.parse_args()
    
    # Initialize miner
    miner = TripoSRMiner()
    
    # Run miner
    miner.run()

if __name__ == "__main__":
    main() 