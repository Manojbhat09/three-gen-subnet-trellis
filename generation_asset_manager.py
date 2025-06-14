#!/usr/bin/env python3
"""
Generation Asset Manager for Subnet 17
Comprehensive data structure for storing all generation outputs with compression and mining integration
"""

import os
import time
import uuid
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pyspz
import trimesh
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetType(Enum):
    """Types of assets generated in the pipeline"""
    ORIGINAL_IMAGE = "original_image"
    GENERATED_IMAGE = "generated_image"  # Added for FLUX output
    BACKGROUND_REMOVED_IMAGE = "background_removed_image"
    PROCESSED_IMAGE = "processed_image"  # Added for processed images
    DEPTH_MAP = "depth_map"
    NORMAL_MAP = "normal_map"
    INITIAL_MESH_GLB = "initial_mesh_glb"
    INITIAL_MESH_PLY = "initial_mesh_ply"
    BPT_ENHANCED_MESH_GLB = "bpt_enhanced_mesh_glb"
    BPT_ENHANCED_MESH_PLY = "bpt_enhanced_mesh_ply"
    TEXTURED_MESH_GLB = "textured_mesh_glb"
    TEXTURED_MESH_PLY = "textured_mesh_ply"
    GAUSSIAN_SPLATTING_PLY = "gaussian_splatting_ply"  # Added for SuGaR output
    VIEWABLE_MESH_PLY = "viewable_mesh_ply"  # Added for viewable mesh
    SIMPLE_POINTS_PLY = "simple_points_ply"  # Added for simple points
    COMPRESSED_PLY = "compressed_ply"
    VALIDATION_PREVIEW = "validation_preview"
    MULTI_VIEW_RENDERS = "multi_view_renders"

class GenerationStatus(Enum):
    """Status of generation process"""
    INITIALIZED = "initialized"
    INITIALIZING = "initializing"  # Added for starting generation
    IMAGE_GENERATING = "image_generating"
    IMAGE_PROCESSING = "image_processing"
    MESH_GENERATING = "mesh_generating"
    MESH_PROCESSING = "mesh_processing"  # Added for mesh post-processing
    MESH_ENHANCING = "mesh_enhancing"
    MESH_TEXTURING = "mesh_texturing"
    CONVERTING_TO_PLY = "converting_to_ply"  # Added for PLY conversion
    COMPRESSING = "compressing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AssetFile:
    """Individual asset file information"""
    asset_type: AssetType
    file_path: str
    file_size: int
    checksum: str
    mime_type: str
    created_at: float
    data: Optional[bytes] = None  # In-memory data for small files
    
    def __post_init__(self):
        if self.file_path and os.path.exists(self.file_path):
            self.file_size = os.path.getsize(self.file_path)
            if self.file_size < 10 * 1024 * 1024:  # Load files < 10MB into memory
                with open(self.file_path, 'rb') as f:
                    self.data = f.read()
                    self.checksum = hashlib.sha256(self.data).hexdigest()[:16]

@dataclass
class GenerationParameters:
    """Parameters used for generation"""
    prompt: str
    seed: int
    num_inference_steps_image: int = 8
    num_inference_steps_shape: int = 30
    guidance_scale_image: float = 3.5
    guidance_scale_shape: float = 7.5
    image_width: int = 1024
    image_height: int = 1024
    use_bpt_enhancement: bool = False
    bpt_temperature: float = 0.5
    use_texture_generation: bool = False
    mc_algorithm: str = "mc"
    flux_model: str = "flux-dev"
    shape_model: str = "Hunyuan3D-2"

@dataclass
class ValidationMetrics:
    """Validation and quality metrics"""
    local_validation_score: float = 0.0
    validator_scores: Dict[str, float] = field(default_factory=dict)
    mesh_quality_score: float = 0.0
    visual_quality_score: float = 0.0
    geometric_complexity: float = 0.0
    texture_quality: float = 0.0
    face_count: int = 0
    vertex_count: int = 0
    is_manifold: bool = False
    has_texture: bool = False
    validation_timestamp: float = 0.0
    validation_notes: str = ""

@dataclass
class PerformanceMetrics:
    """Performance and timing metrics"""
    total_generation_time: float = 0.0
    image_generation_time: float = 0.0
    background_removal_time: float = 0.0
    mesh_generation_time: float = 0.0
    bpt_enhancement_time: float = 0.0
    texture_generation_time: float = 0.0
    compression_time: float = 0.0
    validation_time: float = 0.0
    memory_peak_usage: float = 0.0
    gpu_memory_peak: float = 0.0
    cpu_usage_avg: float = 0.0

@dataclass
class MiningInfo:
    """Mining-specific information"""
    task_id: str = ""
    validator_hotkey: str = ""
    validator_uid: int = -1
    submission_timestamp: float = 0.0
    reward_received: float = 0.0
    submission_status: str = "pending"
    network_uid: int = 17
    competition_score: float = 0.0
    market_rank: int = 0

class GenerationAsset:
    """Comprehensive asset container for all generation outputs"""
    
    def __init__(self, 
                 prompt: str, 
                 seed: int = None,
                 generation_id: str = None,
                 output_dir: str = "./generation_assets"):
        
        # Core identification
        self.generation_id = generation_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.updated_at = self.created_at
        
        # Generation parameters
        if seed is None:
            seed = int(time.time()) % (2**31)
        self.parameters = GenerationParameters(prompt=prompt, seed=seed)
        
        # Status tracking
        self.status = GenerationStatus.INITIALIZED
        self.error_message = ""
        
        # Asset storage
        self.assets: Dict[AssetType, AssetFile] = {}
        self.asset_directory = Path(output_dir) / self.generation_id
        self.asset_directory.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.validation_metrics = ValidationMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.mining_info = MiningInfo()
        
        # Compressed data for submission
        self.compressed_ply_data: Optional[bytes] = None
        self.compression_ratio: float = 0.0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Created GenerationAsset {self.generation_id} for prompt: '{prompt[:50]}...'")

    def update_status(self, status: GenerationStatus, error_message: str = ""):
        """Update generation status with thread safety"""
        with self._lock:
            self.status = status
            self.error_message = error_message
            self.updated_at = time.time()
            logger.info(f"Asset {self.generation_id}: Status -> {status.value}")

    def add_asset(self, 
                  asset_type: AssetType, 
                  data: Union[bytes, str, np.ndarray, trimesh.Trimesh, Image.Image],
                  file_extension: str = None,
                  save_to_disk: bool = True) -> AssetFile:
        """Add an asset to the collection with automatic type handling"""
        
        with self._lock:
            # Determine file extension and MIME type
            if file_extension is None:
                file_extension = self._get_default_extension(asset_type)
            
            mime_type = self._get_mime_type(asset_type, file_extension)
            
            # Generate filename
            safe_prompt = "".join(c for c in self.parameters.prompt if c.isalnum() or c in (' ', '-', '_'))[:50]
            filename = f"{safe_prompt}_{self.parameters.seed}_{asset_type.value}{file_extension}"
            file_path = self.asset_directory / filename
            
            # Convert data to bytes
            file_data = self._convert_to_bytes(data, asset_type, file_extension)
            
            # Save to disk if requested
            if save_to_disk:
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                actual_path = str(file_path)
            else:
                actual_path = ""
            
            # Create asset file
            asset_file = AssetFile(
                asset_type=asset_type,
                file_path=actual_path,
                file_size=len(file_data),
                checksum=hashlib.sha256(file_data).hexdigest()[:16],
                mime_type=mime_type,
                created_at=time.time(),
                data=file_data if len(file_data) < 10 * 1024 * 1024 else None
            )
            
            self.assets[asset_type] = asset_file
            logger.info(f"Asset {self.generation_id}: Added {asset_type.value} ({len(file_data)} bytes)")
            return asset_file

    def get_asset(self, asset_type: AssetType) -> Optional[AssetFile]:
        """Get an asset by type"""
        return self.assets.get(asset_type)

    def get_asset_data(self, asset_type: AssetType) -> Optional[bytes]:
        """Get asset data, loading from disk if necessary"""
        asset = self.get_asset(asset_type)
        if not asset:
            return None
        
        if asset.data:
            return asset.data
        
        if asset.file_path and os.path.exists(asset.file_path):
            with open(asset.file_path, 'rb') as f:
                return f.read()
        
        return None

    def compress_ply_asset(self, 
                          asset_type: AssetType = AssetType.INITIAL_MESH_PLY,
                          compression_level: int = 1,
                          workers: int = 1) -> bool:
        """Compress PLY asset for network submission"""
        
        start_time = time.time()
        
        try:
            ply_data = self.get_asset_data(asset_type)
            if not ply_data:
                logger.error(f"Asset {self.generation_id}: No PLY data found for {asset_type.value}")
                return False
            
            # Try compression using pyspz
            try:
                compressed_data = pyspz.compress(ply_data, compression_level, workers)
                
                # Store compressed data
                self.compressed_ply_data = compressed_data
                self.compression_ratio = len(ply_data) / len(compressed_data)
                
                # Add compressed asset
                self.add_asset(
                    AssetType.COMPRESSED_PLY,
                    compressed_data,
                    ".spz",
                    save_to_disk=True
                )
                
                compression_time = time.time() - start_time
                self.performance_metrics.compression_time = compression_time
                
                logger.info(f"Asset {self.generation_id}: Compressed PLY - "
                           f"Original: {len(ply_data)}, Compressed: {len(compressed_data)}, "
                           f"Ratio: {self.compression_ratio:.2f}x, Time: {compression_time:.2f}s")
                
                return True
                
            except Exception as pyspz_error:
                logger.warning(f"Asset {self.generation_id}: pyspz compression failed: {pyspz_error}")
                
                # Fallback: store uncompressed PLY as "compressed" for submission
                # This ensures the mining system still works even if compression fails
                self.compressed_ply_data = ply_data
                self.compression_ratio = 1.0  # No compression
                
                # Add "compressed" asset (actually uncompressed)
                self.add_asset(
                    AssetType.COMPRESSED_PLY,
                    ply_data,
                    ".ply",  # Keep as PLY since compression failed
                    save_to_disk=True
                )
                
                compression_time = time.time() - start_time
                self.performance_metrics.compression_time = compression_time
                
                logger.info(f"Asset {self.generation_id}: Using uncompressed PLY as fallback - "
                           f"Size: {len(ply_data)} bytes, Time: {compression_time:.2f}s")
                
                return True  # Still return True since we have data for submission
                
        except Exception as e:
            logger.error(f"Asset {self.generation_id}: Compression process failed - {e}")
            return False

    def get_compressed_ply_base64(self) -> str:
        """Get base64 encoded compressed PLY for network submission"""
        if not self.compressed_ply_data:
            return ""
        return base64.b64encode(self.compressed_ply_data).decode('utf-8')

    def update_validation_metrics(self, 
                                local_score: float = None,
                                validator_scores: Dict[str, float] = None,
                                mesh_metrics: Dict[str, Any] = None):
        """Update validation metrics"""
        
        with self._lock:
            if local_score is not None:
                self.validation_metrics.local_validation_score = local_score
            
            if validator_scores:
                self.validation_metrics.validator_scores.update(validator_scores)
            
            if mesh_metrics:
                for key, value in mesh_metrics.items():
                    if hasattr(self.validation_metrics, key):
                        setattr(self.validation_metrics, key, value)
            
            self.validation_metrics.validation_timestamp = time.time()

    def update_mining_info(self, 
                          task_id: str = None,
                          validator_hotkey: str = None,
                          validator_uid: int = None,
                          submission_status: str = None,
                          reward_received: float = None):
        """Update mining information"""
        
        with self._lock:
            if task_id:
                self.mining_info.task_id = task_id
            if validator_hotkey:
                self.mining_info.validator_hotkey = validator_hotkey
            if validator_uid is not None:
                self.mining_info.validator_uid = validator_uid
            if submission_status:
                self.mining_info.submission_status = submission_status
            if reward_received is not None:
                self.mining_info.reward_received = reward_received
                
            if submission_status:
                self.mining_info.submission_timestamp = time.time()

    def save_metadata(self) -> str:
        """Save comprehensive metadata to JSON file"""
        
        metadata = {
            "generation_id": self.generation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "error_message": self.error_message,
            "parameters": asdict(self.parameters),
            "validation_metrics": asdict(self.validation_metrics),
            "performance_metrics": asdict(self.performance_metrics),
            "mining_info": asdict(self.mining_info),
            "compression_ratio": self.compression_ratio,
            "assets": {
                asset_type.value: {
                    "file_path": asset.file_path,
                    "file_size": asset.file_size,
                    "checksum": asset.checksum,
                    "mime_type": asset.mime_type,
                    "created_at": asset.created_at
                }
                for asset_type, asset in self.assets.items()
            }
        }
        
        metadata_path = self.asset_directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)

    def load_metadata(self, metadata_path: str) -> bool:
        """Load metadata from JSON file"""
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.generation_id = metadata["generation_id"]
            self.created_at = metadata["created_at"]
            self.updated_at = metadata["updated_at"]
            self.status = GenerationStatus(metadata["status"])
            self.error_message = metadata["error_message"]
            
            # Reconstruct dataclasses
            self.parameters = GenerationParameters(**metadata["parameters"])
            self.validation_metrics = ValidationMetrics(**metadata["validation_metrics"])
            self.performance_metrics = PerformanceMetrics(**metadata["performance_metrics"])
            self.mining_info = MiningInfo(**metadata["mining_info"])
            self.compression_ratio = metadata.get("compression_ratio", 0.0)
            
            # Reconstruct assets
            self.assets = {}
            for asset_type_str, asset_data in metadata["assets"].items():
                asset_type = AssetType(asset_type_str)
                self.assets[asset_type] = AssetFile(
                    asset_type=asset_type,
                    file_path=asset_data["file_path"],
                    file_size=asset_data["file_size"],
                    checksum=asset_data["checksum"],
                    mime_type=asset_data["mime_type"],
                    created_at=asset_data["created_at"]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return False

    def cleanup(self, keep_compressed: bool = True, keep_metadata: bool = True):
        """Clean up asset files, optionally keeping compressed and metadata"""
        
        for asset_type, asset in self.assets.items():
            if asset_type == AssetType.COMPRESSED_PLY and keep_compressed:
                continue
                
            if asset.file_path and os.path.exists(asset.file_path):
                try:
                    os.remove(asset.file_path)
                    logger.info(f"Cleaned up {asset.file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up {asset.file_path}: {e}")
        
        if not keep_metadata:
            metadata_path = self.asset_directory / "metadata.json"
            if metadata_path.exists():
                try:
                    os.remove(metadata_path)
                except Exception as e:
                    logger.error(f"Failed to clean up metadata: {e}")

    def _get_default_extension(self, asset_type: AssetType) -> str:
        """Get default file extension for asset type"""
        extension_map = {
            AssetType.ORIGINAL_IMAGE: ".png",
            AssetType.GENERATED_IMAGE: ".png",  # Added for FLUX output
            AssetType.BACKGROUND_REMOVED_IMAGE: ".png",
            AssetType.PROCESSED_IMAGE: ".png",  # Added for processed images
            AssetType.DEPTH_MAP: ".png",
            AssetType.NORMAL_MAP: ".png",
            AssetType.INITIAL_MESH_GLB: ".glb",
            AssetType.INITIAL_MESH_PLY: ".ply",
            AssetType.BPT_ENHANCED_MESH_GLB: ".glb",
            AssetType.BPT_ENHANCED_MESH_PLY: ".ply",
            AssetType.TEXTURED_MESH_GLB: ".glb",
            AssetType.TEXTURED_MESH_PLY: ".ply",
            AssetType.GAUSSIAN_SPLATTING_PLY: ".ply",  # Added for SuGaR output
            AssetType.VIEWABLE_MESH_PLY: ".ply",  # Added for viewable mesh
            AssetType.SIMPLE_POINTS_PLY: ".ply",  # Added for simple points
            AssetType.COMPRESSED_PLY: ".spz",
            AssetType.VALIDATION_PREVIEW: ".png",
            AssetType.MULTI_VIEW_RENDERS: ".zip"
        }
        return extension_map.get(asset_type, ".bin")

    def _get_mime_type(self, asset_type: AssetType, extension: str) -> str:
        """Get MIME type for asset"""
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".glb": "model/gltf-binary",
            ".ply": "application/x-ply",
            ".spz": "application/octet-stream",
            ".zip": "application/zip"
        }
        return mime_map.get(extension, "application/octet-stream")

    def _convert_to_bytes(self, 
                         data: Union[bytes, str, np.ndarray, trimesh.Trimesh, Image.Image],
                         asset_type: AssetType,
                         file_extension: str) -> bytes:
        """Convert various data types to bytes"""
        
        if isinstance(data, bytes):
            return data
        
        if isinstance(data, str):
            return data.encode('utf-8')
        
        if isinstance(data, Image.Image):
            from io import BytesIO
            buffer = BytesIO()
            format_map = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
            fmt = format_map.get(file_extension, "PNG")
            data.save(buffer, format=fmt)
            return buffer.getvalue()
        
        if isinstance(data, np.ndarray):
            from io import BytesIO
            buffer = BytesIO()
            if file_extension == ".png":
                Image.fromarray(data.astype(np.uint8)).save(buffer, format="PNG")
            else:
                np.save(buffer, data)
            return buffer.getvalue()
        
        if isinstance(data, trimesh.Trimesh):
            from io import BytesIO
            
            if file_extension == ".ply":
                # Use trimesh's PLY export function directly
                try:
                    return trimesh.exchange.ply.export_ply(data)
                except Exception as e:
                    logger.warning(f"PLY export via exchange.ply failed: {e}, trying fallback")
                    # Fallback: use export with file_type specified
                    try:
                        buffer = BytesIO()
                        data.export(buffer, file_type='ply')
                        return buffer.getvalue()
                    except Exception as e2:
                        logger.error(f"PLY export fallback failed: {e2}")
                        raise e2
            elif file_extension == ".glb":
                try:
                    buffer = BytesIO()
                    data.export(buffer, file_type='glb')
                    return buffer.getvalue()
                except Exception as e:
                    logger.warning(f"GLB export failed: {e}, trying direct export")
                    return data.export(file_type='glb')
            else:
                # Default to PLY for unknown extensions
                try:
                    return trimesh.exchange.ply.export_ply(data)
                except Exception:
                    buffer = BytesIO()
                    data.export(buffer, file_type='ply')
                    return buffer.getvalue()
        
        # Fallback: try to serialize as string
        return str(data).encode('utf-8')


class AssetManager:
    """Manager for multiple generation assets with mining integration"""
    
    def __init__(self, base_output_dir: str = "./generation_assets"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.assets: Dict[str, GenerationAsset] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()
        
        logger.info(f"AssetManager initialized with base directory: {base_output_dir}")

    def create_asset(self, prompt: str, seed: int = None, generation_id: str = None) -> GenerationAsset:
        """Create a new generation asset"""
        
        asset = GenerationAsset(
            prompt=prompt,
            seed=seed,
            generation_id=generation_id,
            output_dir=str(self.base_output_dir)
        )
        
        with self._lock:
            self.assets[asset.generation_id] = asset
        
        return asset

    def get_asset(self, generation_id: str) -> Optional[GenerationAsset]:
        """Get asset by generation ID"""
        return self.assets.get(generation_id)

    def get_assets_by_prompt(self, prompt: str) -> List[GenerationAsset]:
        """Get all assets matching a prompt"""
        return [asset for asset in self.assets.values() 
                if asset.parameters.prompt == prompt]

    def get_completed_assets(self) -> List[GenerationAsset]:
        """Get all completed assets"""
        return [asset for asset in self.assets.values() 
                if asset.status == GenerationStatus.COMPLETED]

    def get_submission_ready_assets(self) -> List[GenerationAsset]:
        """Get assets ready for mining submission"""
        return [asset for asset in self.assets.values() 
                if (asset.status == GenerationStatus.COMPLETED and 
                    asset.compressed_ply_data is not None and
                    asset.validation_metrics.local_validation_score > 0)]

    async def compress_asset_async(self, generation_id: str) -> bool:
        """Asynchronously compress an asset"""
        asset = self.get_asset(generation_id)
        if not asset:
            return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            asset.compress_ply_asset
        )

    def save_all_metadata(self):
        """Save metadata for all assets"""
        for asset in self.assets.values():
            try:
                asset.save_metadata()
            except Exception as e:
                logger.error(f"Failed to save metadata for {asset.generation_id}: {e}")

    def load_assets_from_directory(self, directory: str = None) -> int:
        """Load assets from metadata files in directory"""
        
        if directory is None:
            directory = self.base_output_dir
        
        directory = Path(directory)
        loaded_count = 0
        
        for metadata_file in directory.glob("*/metadata.json"):
            try:
                asset = GenerationAsset("", 0)  # Temporary
                if asset.load_metadata(str(metadata_file)):
                    with self._lock:
                        self.assets[asset.generation_id] = asset
                    loaded_count += 1
                    logger.info(f"Loaded asset {asset.generation_id}")
            except Exception as e:
                logger.error(f"Failed to load asset from {metadata_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} assets from {directory}")
        return loaded_count

    def cleanup_old_assets(self, max_age_hours: float = 24, keep_successful: bool = True):
        """Clean up old assets to save disk space"""
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        to_remove = []
        for generation_id, asset in self.assets.items():
            if asset.created_at < cutoff_time:
                if keep_successful and asset.status == GenerationStatus.COMPLETED:
                    continue
                to_remove.append(generation_id)
        
        for generation_id in to_remove:
            asset = self.assets[generation_id]
            asset.cleanup(keep_compressed=True, keep_metadata=True)
            del self.assets[generation_id]
            logger.info(f"Cleaned up old asset {generation_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about assets"""
        
        total_assets = len(self.assets)
        if total_assets == 0:
            return {"total_assets": 0}
        
        status_counts = {}
        total_size = 0
        total_generation_time = 0
        total_validation_score = 0
        completed_count = 0
        
        for asset in self.assets.values():
            # Status counts
            status = asset.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Size calculation
            for asset_file in asset.assets.values():
                total_size += asset_file.file_size
            
            # Performance metrics
            total_generation_time += asset.performance_metrics.total_generation_time
            
            # Validation scores
            if asset.status == GenerationStatus.COMPLETED:
                completed_count += 1
                total_validation_score += asset.validation_metrics.local_validation_score
        
        avg_validation_score = (total_validation_score / completed_count) if completed_count > 0 else 0
        avg_generation_time = total_generation_time / total_assets
        
        return {
            "total_assets": total_assets,
            "status_distribution": status_counts,
            "total_storage_mb": total_size / (1024 * 1024),
            "avg_generation_time": avg_generation_time,
            "avg_validation_score": avg_validation_score,
            "completed_assets": completed_count,
            "submission_ready": len(self.get_submission_ready_assets())
        }


# Global asset manager instance
global_asset_manager = AssetManager()


# Utility functions for integration with existing servers

def create_generation_asset_from_demo_outputs(prompt: str, 
                                            seed: int,
                                            demo_output_dir: str) -> GenerationAsset:
    """Create asset from flux_hunyuan_bpt_demo.py outputs"""
    
    asset = global_asset_manager.create_asset(prompt, seed)
    asset.update_status(GenerationStatus.IMAGE_GENERATING)
    
    demo_dir = Path(demo_output_dir)
    
    # Add all demo outputs
    file_mappings = {
        "t2i_original.png": AssetType.ORIGINAL_IMAGE,
        "t2i_no_bg.png": AssetType.BACKGROUND_REMOVED_IMAGE,
        "t2i_initial.glb": AssetType.INITIAL_MESH_GLB,
        "t2i_enhanced_bpt.glb": AssetType.BPT_ENHANCED_MESH_GLB,
        "t2i_textured.glb": AssetType.TEXTURED_MESH_GLB
    }
    
    for filename, asset_type in file_mappings.items():
        file_path = demo_dir / filename
        if file_path.exists():
            with open(file_path, 'rb') as f:
                data = f.read()
            asset.add_asset(asset_type, data, file_path.suffix)
    
    asset.update_status(GenerationStatus.COMPLETED)
    return asset


def integrate_with_robust_server(asset: GenerationAsset, 
                                server_metrics: Dict[str, Any]) -> None:
    """Integrate asset with robust_generation_server.py metrics"""
    
    # Update performance metrics from server
    if "generation_time" in server_metrics:
        asset.performance_metrics.total_generation_time = server_metrics["generation_time"]
    
    if "memory_peak" in server_metrics:
        asset.performance_metrics.memory_peak_usage = server_metrics["memory_peak"]
    
    if "gpu_memory_peak" in server_metrics:
        asset.performance_metrics.gpu_memory_peak = server_metrics["gpu_memory_peak"]


def prepare_for_mining_submission(asset: GenerationAsset,
                                task_id: str,
                                validator_hotkey: str,
                                validator_uid: int) -> Dict[str, Any]:
    """Prepare asset for mining submission"""
    
    # Update mining info
    asset.update_mining_info(
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
        submission_status="preparing"
    )
    
    # Ensure compressed PLY exists
    if not asset.compressed_ply_data:
        success = asset.compress_ply_asset()
        if not success:
            return {"error": "Failed to compress PLY"}
    
    # Prepare submission data
    submission_data = {
        "generation_id": asset.generation_id,
        "prompt": asset.parameters.prompt,
        "seed": asset.parameters.seed,
        "compressed_ply_b64": asset.get_compressed_ply_base64(),
        "local_validation_score": asset.validation_metrics.local_validation_score,
        "compression_ratio": asset.compression_ratio,
        "generation_time": asset.performance_metrics.total_generation_time,
        "face_count": asset.validation_metrics.face_count,
        "vertex_count": asset.validation_metrics.vertex_count
    }
    
    asset.update_mining_info(submission_status="ready")
    return submission_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create test asset
    asset = global_asset_manager.create_asset("a blue monkey", 42)
    
    # Simulate adding assets
    asset.update_status(GenerationStatus.IMAGE_GENERATING)
    
    # Add mock image
    test_image = Image.new('RGB', (1024, 1024), color='blue')
    asset.add_asset(AssetType.ORIGINAL_IMAGE, test_image)
    
    # Add mock mesh
    test_mesh = trimesh.creation.box()
    asset.add_asset(AssetType.INITIAL_MESH_PLY, test_mesh)
    
    # Compress for submission
    asset.compress_ply_asset()
    
    asset.update_status(GenerationStatus.COMPLETED)
    
    # Save metadata
    metadata_path = asset.save_metadata()
    print(f"Saved metadata to: {metadata_path}")
    
    # Get statistics
    stats = global_asset_manager.get_statistics()
    print(f"Asset statistics: {stats}") 