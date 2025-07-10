from typing import List, Dict, Any
import os
import numpy as np
import cv2
from .abstract_data_writer import AbstractDataWriter


class MaskDataWriter(AbstractDataWriter):
    """
    Writer component for saving point cloud data to PLY files.
    
    Args:
        output_dir: Directory where point cloud files will be written.
        only_save_last: Only saves the output if its the last frame
    Returns:
        -
    
    Raises:
        ValueError: If input point cloud data is invalid
    """

    def __init__(
            self, 
            output_dir: str, 
        ):
        super().__init__(output_dir)
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component reads point cloud data."""
        return ["step_nr","segmentation_masks", "image"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component doesn't add anything to the bucket."""
        return []
    
    def _run(
        self, 
        step_nr: int,
        segmentation_masks,
        image: Any
    ) -> Dict[str, Any]:
                # Visualize results
        vis_frame = self.get_visualization(image.as_numpy(), segmentation_masks)
                # Visualize results
        output_dir = os.path.join(self.output_dir,f"step_{step_nr}")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        cv2.imwrite(os.path.join(self.output_dir,f"now_frame_{step_nr}.jpg"), vis_frame)
        return {}
    
    def get_visualization(self, frame, masks):
        """Create visualization of masks"""
        vis = frame.copy() *255
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        
        for i, (obj_id, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            vis[mask] = vis[mask] * 0.7 + np.array(color) * 0.3
            
            # Add object ID text
            y, x = np.where(mask)
            if len(y) > 0:
                center_y, center_x = int(np.mean(y)), int(np.mean(x))
                cv2.putText(vis, f'ID:{obj_id}', (center_x-15, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis
    


















    