# Compatibility module for existing scripts
# This redirects imports to the correct validation.serve location

from validation.serve import decode_and_validate_txt as _decode_and_validate_txt
from validation.engine.data_structures import ValidationRequest, ValidationResponse, RequestData
from validation.engine.io.ply.loader import PlyLoader
from validation.engine.rendering.renderer import Renderer
from validation.engine.validation_engine import ValidationEngine
from validation.engine.data_structures import TimeStat
from typing import Optional, Dict, Any
import base64


def decode_and_validate_txt(
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    zstd_decompressor=None,  # Ignored for compatibility
    validator: Optional[ValidationEngine] = None,
    include_time_stat: bool = False
) -> Any:
    """
    Compatibility wrapper for existing scripts.
    Maps the old calling convention to the new function signature.
    """
    # Create a new ValidationRequest if needed
    if not isinstance(request, ValidationRequest):
        # Handle RequestData compatibility
        if isinstance(request, RequestData):
            request = ValidationRequest(
                prompt=request.prompt,
                data=request.data,
                compression=request.compression,
                generate_preview=request.generate_preview,
                preview_score_threshold=request.preview_score_threshold,
            )
        # Handle dict compatibility
        elif isinstance(request, dict):
            request = ValidationRequest(
                prompt=request.get('prompt'),
                data=request.get('data'),
                compression=request.get('compression', 0),
                generate_preview=request.get('generate_preview', False),
                preview_score_threshold=request.get('preview_score_threshold', 0.8),
            )
        else:
            raise ValueError(f"Invalid request type: {type(request)}")
    
    # Call the actual function with the correct signature
    response, time_stat = _decode_and_validate_txt(
        request=request,
        ply_data_loader=ply_data_loader,
        renderer=renderer,
        validator=validator
    )
    
    # Return in the format expected by old scripts
    if include_time_stat:
        # Create a compatibility response object
        class CompatibilityResponse:
            def __init__(self, response: ValidationResponse, time_stat: TimeStat):
                self.response_data = response
                self.time_stat = time_stat
        
        return CompatibilityResponse(response, time_stat)
    else:
        return response


__all__ = ['decode_and_validate_txt'] 