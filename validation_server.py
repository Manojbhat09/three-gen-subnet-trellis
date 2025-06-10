#!/usr/bin/env python3
# Subnet 17 (404-GEN) - Validation Server
# Purpose: HTTP server for validating 3D models (PLY files) and returning quality scores

import os
import time
import base64
import tempfile
import traceback
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import trimesh
import pyspz
from PIL import Image
import torch

# Configuration
VALIDATION_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'temp_dir': './validation_temp',
    'min_vertices': 100,
    'min_faces': 100,
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'render_resolution': 512,
    'num_render_views': 8
}

@dataclass
class ValidationMetrics:
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    average_validation_time: float = 0.0
    last_validation_time: float = 0.0

class ValidationRequest(BaseModel):
    prompt: str
    data: str  # base64 encoded PLY data
    compression: int = 0  # 0: none, 1: zstd, 2: spz
    data_ver: int = 0
    generate_preview: bool = False

class ValidationResponse(BaseModel):
    score: float
    details: Dict[str, Any]
    preview_images: Optional[List[str]] = None  # base64 encoded images
    validation_time: float

class PLYValidator:
    def __init__(self):
        self.metrics = ValidationMetrics()
        self.validation_lock = threading.Lock()
        
        # Ensure temp directory exists
        Path(VALIDATION_CONFIG['temp_dir']).mkdir(exist_ok=True)
        
        print("PLY Validator initialized")
    
    def decompress_data(self, data: str, compression: int) -> bytes:
        """Decompress the input data based on compression type."""
        raw_data = base64.b64decode(data)
        
        if compression == 0:  # No compression
            return raw_data
        elif compression == 1:  # zstd compression
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(raw_data)
        elif compression == 2:  # SPZ compression
            return pyspz.decompress(raw_data)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
    
    def validate_mesh_structure(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Validate basic mesh structure and quality."""
        validation_results = {
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'is_watertight': mesh.is_watertight,
            'is_valid': mesh.is_valid,
            'has_vertex_normals': mesh.vertex_normals is not None,
            'has_face_normals': mesh.face_normals is not None,
            'bounding_box_volume': float(mesh.bounding_box.volume),
            'surface_area': float(mesh.area)
        }
        
        # Check for minimum requirements
        validation_results['meets_min_vertices'] = len(mesh.vertices) >= VALIDATION_CONFIG['min_vertices']
        validation_results['meets_min_faces'] = len(mesh.faces) >= VALIDATION_CONFIG['min_faces']
        
        # Check for degenerate faces
        face_areas = mesh.area_faces
        validation_results['has_degenerate_faces'] = np.any(face_areas < 1e-10)
        validation_results['degenerate_face_count'] = int(np.sum(face_areas < 1e-10))
        
        # Check mesh bounds
        bounds = mesh.bounds
        validation_results['mesh_size'] = float(np.linalg.norm(bounds[1] - bounds[0]))
        
        return validation_results
    
    def render_mesh_views(self, mesh: trimesh.Trimesh) -> List[np.ndarray]:
        """Render multiple views of the mesh for visual quality assessment."""
        try:
            # Create scene
            scene = trimesh.Scene([mesh])
            
            # Generate camera positions in a sphere around the object
            angles = np.linspace(0, 2 * np.pi, VALIDATION_CONFIG['num_render_views'], endpoint=False)
            elevation = np.pi / 4  # 45 degrees
            radius = 2.0
            
            rendered_images = []
            
            for angle in angles:
                # Calculate camera position
                camera_pos = np.array([
                    radius * np.cos(angle) * np.cos(elevation),
                    radius * np.sin(angle) * np.cos(elevation),
                    radius * np.sin(elevation)
                ])
                
                # Set camera transform
                camera_transform = trimesh.scene.cameras.look_at(
                    camera_pos,
                    mesh.centroid,
                    [0, 0, 1]
                )
                
                # Render the scene
                try:
                    rendered = scene.save_image(
                        resolution=(VALIDATION_CONFIG['render_resolution'], VALIDATION_CONFIG['render_resolution']),
                        camera_transform=camera_transform
                    )
                    if rendered is not None:
                        rendered_images.append(np.array(rendered))
                except Exception as e:
                    print(f"Warning: Failed to render view at angle {angle}: {e}")
                    continue
            
            return rendered_images
        
        except Exception as e:
            print(f"Warning: Mesh rendering failed: {e}")
            return []
    
    def calculate_quality_score(
        self, 
        mesh_validation: Dict[str, Any], 
        rendered_images: List[np.ndarray],
        prompt: str
    ) -> float:
        """Calculate overall quality score based on various metrics."""
        score = 0.0
        
        # Structural quality (40% of score)
        structural_score = 0.0
        
        # Basic requirements
        if mesh_validation['meets_min_vertices']:
            structural_score += 0.2
        if mesh_validation['meets_min_faces']:
            structural_score += 0.2
        if mesh_validation['is_valid']:
            structural_score += 0.2
        if mesh_validation['is_watertight']:
            structural_score += 0.2
        
        # Degenerate faces penalty
        if not mesh_validation['has_degenerate_faces']:
            structural_score += 0.2
        else:
            # Penalize based on percentage of degenerate faces
            degenerate_ratio = mesh_validation['degenerate_face_count'] / max(mesh_validation['face_count'], 1)
            structural_score += 0.2 * (1.0 - min(degenerate_ratio * 10, 1.0))
        
        score += structural_score * 0.4
        
        # Geometric quality (30% of score)
        geometric_score = 0.0
        
        # Size reasonableness (not too small or too large)
        mesh_size = mesh_validation['mesh_size']
        if 0.1 <= mesh_size <= 10.0:
            geometric_score += 0.5
        elif 0.01 <= mesh_size <= 100.0:
            geometric_score += 0.3
        else:
            geometric_score += 0.1
        
        # Surface area to volume ratio (complexity indicator)
        volume = mesh_validation['bounding_box_volume']
        area = mesh_validation['surface_area']
        if volume > 0:
            complexity_ratio = area / (volume ** (2/3))
            if 1.0 <= complexity_ratio <= 50.0:
                geometric_score += 0.5
            else:
                geometric_score += 0.2
        
        score += geometric_score * 0.3
        
        # Visual quality (30% of score)
        visual_score = 0.0
        
        if rendered_images:
            # Basic check: images were successfully rendered
            visual_score += 0.5
            
            # Check for image content (not just black or white)
            for img in rendered_images:
                if len(img.shape) == 3:
                    mean_intensity = np.mean(img)
                    std_intensity = np.std(img)
                    # Good images should have reasonable intensity distribution
                    if 30 <= mean_intensity <= 225 and std_intensity >= 10:
                        visual_score += 0.5 / len(rendered_images)
        
        score += visual_score * 0.3
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))
    
    def validate_ply(
        self, 
        prompt: str, 
        data: str, 
        compression: int = 0,
        data_ver: int = 0,
        generate_preview: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a PLY file and return quality score and details.
        """
        with self.validation_lock:
            start_time = time.time()
            self.metrics.total_validations += 1
            
            try:
                # Decompress data
                ply_bytes = self.decompress_data(data, compression)
                
                # Check file size
                if len(ply_bytes) > VALIDATION_CONFIG['max_file_size']:
                    raise ValueError(f"File too large: {len(ply_bytes)} bytes")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False, dir=VALIDATION_CONFIG['temp_dir']) as tmp_file:
                    tmp_file.write(ply_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Load mesh
                    mesh = trimesh.load(tmp_path)
                    
                    if not isinstance(mesh, trimesh.Trimesh):
                        raise ValueError("Loaded file is not a valid mesh")
                    
                    # Validate mesh structure
                    mesh_validation = self.validate_mesh_structure(mesh)
                    
                    # Render mesh views
                    rendered_images = []
                    preview_images = []
                    
                    if generate_preview:
                        rendered_images = self.render_mesh_views(mesh)
                        # Convert rendered images to base64
                        for img in rendered_images:
                            img_pil = Image.fromarray(img)
                            import io
                            img_buffer = io.BytesIO()
                            img_pil.save(img_buffer, format='PNG')
                            img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            preview_images.append(img_b64)
                    
                    # Calculate quality score
                    quality_score = self.calculate_quality_score(
                        mesh_validation, rendered_images, prompt
                    )
                    
                    # Update metrics
                    validation_time = time.time() - start_time
                    self.metrics.successful_validations += 1
                    self.metrics.last_validation_time = validation_time
                    self.metrics.average_validation_time = (
                        (self.metrics.average_validation_time * (self.metrics.successful_validations - 1) + validation_time) /
                        self.metrics.successful_validations
                    )
                    
                    return {
                        'score': quality_score,
                        'details': {
                            'mesh_validation': mesh_validation,
                            'file_size': len(ply_bytes),
                            'compression_used': compression,
                            'data_version': data_ver,
                            'num_preview_images': len(preview_images)
                        },
                        'preview_images': preview_images if generate_preview else None,
                        'validation_time': validation_time
                    }
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
            except Exception as e:
                self.metrics.failed_validations += 1
                print(f"Validation error: {e}")
                traceback.print_exc()
                raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get validator status and metrics."""
        return {
            'device': VALIDATION_CONFIG['device'],
            'total_validations': self.metrics.total_validations,
            'successful_validations': self.metrics.successful_validations,
            'failed_validations': self.metrics.failed_validations,
            'success_rate': (
                self.metrics.successful_validations / max(self.metrics.total_validations, 1) * 100
            ),
            'average_validation_time': self.metrics.average_validation_time,
            'last_validation_time': self.metrics.last_validation_time,
            'config': VALIDATION_CONFIG
        }

# Initialize validator
validator = PLYValidator()

# FastAPI app
app = FastAPI(title="Subnet 17 Validation Server", version="1.0.0")

@app.post("/validate_txt_to_3d_ply/", response_model=ValidationResponse)
async def validate_ply_endpoint(request: ValidationRequest):
    """Validate a PLY file and return quality score."""
    try:
        result = validator.validate_ply(
            prompt=request.prompt,
            data=request.data,
            compression=request.compression,
            data_ver=request.data_ver,
            generate_preview=request.generate_preview
        )
        
        return ValidationResponse(
            score=result['score'],
            details=result['details'],
            preview_images=result['preview_images'],
            validation_time=result['validation_time']
        )
        
    except Exception as e:
        print(f"Error in validation endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/status/")
async def get_status():
    """Get validation server status."""
    return validator.get_status()

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/clear_cache/")
async def clear_cache():
    """Clear temporary files."""
    try:
        temp_dir = Path(VALIDATION_CONFIG['temp_dir'])
        for file in temp_dir.glob("*.ply"):
            file.unlink()
        return {"message": "Temporary files cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Subnet 17 Validation Server...")
    print(f"Device: {VALIDATION_CONFIG['device']}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8094,
        log_level="info"
    ) 