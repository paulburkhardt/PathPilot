import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Dict, Any, List
import numpy as np

class VLMProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config['model_name']
        self.device = config['device']
        self.max_length = config['max_length']
        
        # Initialize BLIP-2
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)

    def describe_objects(self, 
                        frame: np.ndarray, 
                        segmented_objects: List[Dict[str, Any]]) -> List[str]:
        """
        Generate descriptions for segmented objects using BLIP-2.
        
        Args:
            frame: Input RGB frame
            segmented_objects: List of segmented objects with masks
            
        Returns:
            List of object descriptions
        """
        descriptions = []
        
        for obj in segmented_objects:
            # Extract object region using mask
            mask = obj['mask']
            object_region = frame.copy()
            object_region[~mask] = 0
            
            # Process image with BLIP-2
            inputs = self.processor(
                object_region,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length
                )
            
            description = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            descriptions.append(description)
            
        return descriptions 