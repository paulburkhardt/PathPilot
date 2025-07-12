import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
import os

from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter



class ImageEmbeddingSegmenter(AbstractDataSegmenter):
    """
    Component for processing images with masks to generate embeddings and descriptions.
    
    Args:
        -
    Returns:
        -
    Raises:
        -
    """

    def __init__(self, 
                text= "A picture of"
                ) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        save_directory = "./blip-base"
        self.processor = AutoProcessor.from_pretrained(save_directory, use_safetensors=True, trust_remote_code=True)
        self.model = BlipForConditionalGeneration.from_pretrained(save_directory,  use_safetensors=True, trust_remote_code=True)
        self.model.to(self.device)
        self.text = text

    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires images and masks as input."""
        return ["image","image_segmentation_mask","key_frame_flag"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs embeddings and descriptions."""
        return ["embeddings", "descriptions"]
    


    def _run(self, image: Any, image_segmentation_mask, key_frame_flag, **kwargs: Any) -> Dict[str, Any]:
        """
        Process an image with image_segmentation_mask to generate embeddings and descriptions.
        
        Args:
            image: The input image
            image_segmentation_mask: Binary image_segmentation_mask for object segmentation
            key_frame_flag: Flag indicating if this is a key frame
            **kwargs: Additional unused arguments
        Returns:
            Dictionary containing embeddings and descriptions
        """
        embeddings = {}
        descriptions = {}

        for mask_key, mask in image_segmentation_mask.items():
            processed_image = self.apply_mask_and_crop(image, mask)
            processed_image_pil = Image.fromarray(processed_image)
            generated_caption, embedding = self.image_information_extraction(processed_image_pil)
            embeddings[mask_key] = embedding
            descriptions[mask_key] = generated_caption

        return {
            "embeddings": embeddings,
            "descriptions": descriptions
        }

    def apply_mask_and_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to image and crop to the mask bounds.
        
        Args:
            image: Input image as numpy array
            mask: Binary mask as numpy array (0s and 1s)
            
        Returns:
            Cropped and masked image as numpy array
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply mask to image (element-wise multiplication)
        masked_image = image * mask[..., np.newaxis] if len(image.shape) == 3 else image * mask
        
        # Find bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Crop to mask bounds
            cropped_image = masked_image[rmin:rmax+1, cmin:cmax+1]
        else:
            # If no mask found, return the original image
            cropped_image = image
        
        return cropped_image

    def image_information_extraction(self,processed_image_pil):
        # 5. Generate the Embedding Vector and Caption using BlipForConditionalGeneration
        with torch.no_grad():
            inputs = self.processor(images=processed_image_pil, text=self.text, return_tensors="pt").to(self.device)

            # Use the vision model to get image features
            vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values, return_dict=True)
            
            # Get the pooled image features and apply post-layernorm
            image_features = self.model.vision_model.post_layernorm(vision_outputs.pooler_output)
            
            normalized_embedding = F.normalize(image_features, p=2, dim=-1)
            print(f"Embedding vector generated. Shape: {normalized_embedding.shape}")

            # Generate a Caption for the Masked Object
            generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"Generated Caption: '{generated_caption}'")

        return generated_caption, normalized_embedding