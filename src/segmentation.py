import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Any, List

class SegmentationProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.model_type = config['model_type']
        self.device = config['device']
        self.confidence_threshold = config['confidence_threshold']
        
        # Initialize SAM-B
        self.predictor = self._initialize_sam()

    def _initialize_sam(self) -> SamPredictor:
        """Initialize SAM-B model."""
        sam = sam_model_registry[self.model_type]()
        sam.to(device=self.device)
        return SamPredictor(sam)

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a single frame to segment objects.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            List of dictionaries containing segmentation masks and metadata
        """
        # Set image in predictor
        self.predictor.set_image(frame)
        
        # Generate automatic masks
        masks = self.predictor.generate()
        
        # Filter masks based on confidence
        filtered_masks = []
        for mask in masks:
            if mask['stability_score'] > self.confidence_threshold:
                filtered_masks.append({
                    'mask': mask['segmentation'],
                    'confidence': mask['stability_score'],
                    'bbox': mask['bbox']
                })
        
        return filtered_masks 