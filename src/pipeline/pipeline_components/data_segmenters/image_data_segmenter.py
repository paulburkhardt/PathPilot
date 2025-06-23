from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter
from src.pipeline.data_entities.image_segmentation_mask_data_entity import ImageSegmentationMaskDataEntity
import torch

class ImageDataSegmenter(AbstractDataSegmenter):
    """
    Component for segmenting image data.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: As this is currently a placeholder
    """
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires RGB images as input."""
        return ["image"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs segmentation masks."""
        return ["image_segmentation_mask"]
    
    def _run(self, image: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Segment an RGB image.
        
        Args:
            rgb_image: The input RGB image
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """

        img = image.as_pytorch()


        # dummy mask with stripes.
        height, width = img.shape[0], img.shape[1]
        mask = torch.zeros((height, width), dtype=torch.long)

        stripe_height = height // 5
        for i in range(5):
            start = i * stripe_height
            end = (i + 1) * stripe_height if i < 4 else height
            mask[start:end, :] = i

        output= { 
            "image_segmentation_mask":ImageSegmentationMaskDataEntity(
                mask=mask
            )
        }
    
        return output

        