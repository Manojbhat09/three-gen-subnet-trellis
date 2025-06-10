import os
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import psutil
import GPUtil
from dataclasses import dataclass
import numpy as np
from PIL import Image
import shutil
import traceback

# Import from Hunyuan3D-2
from Hunyuan3D_2.minimal_flux_demo import (
    text_to_3d,
    BackgroundRemover,
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
    Hunyuan3DPaintPipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    total_time: float
    vram_usage: float
    cpu_usage: float
    mesh_vertices: int
    mesh_faces: int
    texture_resolution: Tuple[int, int]
    quality_score: float

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start(self):
        self.start_time = time.time()
        self.metrics['initial_vram'] = self._get_vram_usage()
        self.metrics['initial_cpu'] = psutil.cpu_percent()
    
    def end(self) -> Dict:
        self.metrics['total_time'] = time.time() - self.start_time
        self.metrics['final_vram'] = self._get_vram_usage()
        self.metrics['final_cpu'] = psutil.cpu_percent()
        self.metrics['vram_usage'] = self.metrics['final_vram'] - self.metrics['initial_vram']
        return self.metrics
    
    @staticmethod
    def _get_vram_usage() -> float:
        try:
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        except:
            return 0.0

class QualityChecker:
    @staticmethod
    def check_mesh_quality(mesh_path: str) -> Tuple[bool, str]:
        """Check if the generated mesh meets quality standards."""
        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            
            # Basic quality checks
            if len(mesh.vertices) < 100:
                return False, "Mesh has too few vertices"
            
            if len(mesh.faces) < 100:
                return False, "Mesh has too few faces"
            
            # Check for non-manifold edges
            if not mesh.is_watertight:
                return False, "Mesh is not watertight"
            
            # Check for degenerate faces
            if mesh.is_empty:
                return False, "Mesh is empty"
            
            return True, "Mesh quality checks passed"
        except Exception as e:
            return False, f"Error checking mesh quality: {str(e)}"

    @staticmethod
    def check_texture_quality(texture_path: str) -> Tuple[bool, str]:
        """Check if the generated texture meets quality standards."""
        try:
            img = Image.open(texture_path)
            width, height = img.size
            
            # Check texture resolution
            if width < 512 or height < 512:
                return False, "Texture resolution too low"
            
            # Check if texture is not empty
            if np.mean(np.array(img)) < 1.0:
                return False, "Texture appears to be empty"
            
            return True, "Texture quality checks passed"
        except Exception as e:
            return False, f"Error checking texture quality: {str(e)}"

class DiskSpaceManager:
    def __init__(self, directory: Path, max_size_gb: float = 10.0):
        self.directory = directory
        self.max_size_bytes = max_size_gb * 1024 ** 3

    def get_total_size(self) -> int:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def cleanup(self):
        """Delete oldest generation folders until under the size limit."""
        if not self.directory.exists():
            return
        generations = [d for d in self.directory.iterdir() if d.is_dir() and d.name.startswith('generation_')]
        generations.sort(key=lambda d: d.stat().st_mtime)  # oldest first
        total_size = self.get_total_size()
        removed = 0
        while total_size > self.max_size_bytes and generations:
            oldest = generations.pop(0)
            try:
                shutil.rmtree(oldest)
                logger.warning(f"Removed old generation folder: {oldest}")
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove {oldest}: {e}")
            total_size = self.get_total_size()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old generations to free disk space.")

class TextTo3DGenerator:
    def __init__(self, output_dir: str = "outputs", max_output_gb: float = 10.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.performance_monitor = PerformanceMonitor()
        self.quality_checker = QualityChecker()
        self.disk_manager = DiskSpaceManager(self.output_dir, max_output_gb)
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required models."""
        try:
            self.rembg = BackgroundRemover()
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                'jetx/Hunyuan3D-2',
                use_safetensors=True
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def generate(self, prompt: str, seed: int = 42) -> Tuple[bool, Dict, str]:
        """
        Generate 3D model from text prompt.
        Returns: (success, metrics, message)
        """
        # Clean up disk space before generation
        try:
            self.disk_manager.cleanup()
        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}\n{traceback.format_exc()}")
            return False, {}, f"Disk cleanup failed: {e}"
        # Check if enough space is available
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < 2 * 1024 ** 3:  # Require at least 2GB free
            logger.error("Not enough disk space to generate new 3D model.")
            return False, {}, "Not enough disk space to generate new 3D model."
        self.performance_monitor.start()
        try:
            # Generate 3D model
            output_path = self.output_dir / f"generation_{int(time.time())}"
            output_path.mkdir(exist_ok=True)
            # Run text-to-3D generation
            text_to_3d(
                prompt=prompt,
                output_dir=str(output_path),
                seed=seed
            )
            # Get performance metrics
            metrics = self.performance_monitor.end()
            # Check quality
            mesh_path = output_path / 't2i_final_1.glb'
            texture_path = output_path / 't2i_texture.glb'
            mesh_quality, mesh_message = self.quality_checker.check_mesh_quality(str(mesh_path))
            texture_quality, texture_message = self.quality_checker.check_texture_quality(str(texture_path))
            if not mesh_quality or not texture_quality:
                return False, metrics, f"Quality check failed: {mesh_message}, {texture_message}"
            return True, metrics, "Generation successful"
        except Exception as e:
            metrics = self.performance_monitor.end() if self.performance_monitor.start_time else {}
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            return False, metrics, f"Generation failed: {str(e)}"
    
    def clear_memory(self):
        """Clear GPU memory."""
        torch.cuda.empty_cache()
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'rembg'):
            del self.rembg

def main():
    # Example usage
    generator = TextTo3DGenerator(max_output_gb=10.0)  # 10GB limit by default
    try:
        success, metrics, message = generator.generate(
            prompt="a red sports car",
            seed=42
        )
        logger.info(f"Generation {'successful' if success else 'failed'}")
        logger.info(f"Message: {message}")
        logger.info("Performance metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    finally:
        generator.clear_memory()

if __name__ == "__main__":
    main() 