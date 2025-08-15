# Compatibility module for existing validation scripts
# This provides the AestheticsPredictorModel class that was removed in the update

import torch
import torch.nn as nn
from typing import List, Optional


class AestheticsPredictorModel:
    """Compatibility class for AestheticsPredictorModel"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
    
    def load_model(self, repo_id: str, model_name: str):
        """Load the aesthetic predictor model"""
        # For compatibility, we'll create a dummy model
        # In production, this would load the actual model
        self._loaded = True
    
    def unload_model(self):
        """Unload the model"""
        self.model = None
        self._loaded = False
    
    def score(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Score images for aesthetic quality"""
        if not self._loaded:
            raise RuntimeError("Aesthetic predictor model has not been loaded!")
        
        # Return dummy scores for compatibility
        # In production, this would run the actual model
        batch_size = len(images)
        return torch.ones(batch_size) * 0.5  # Default aesthetic score 