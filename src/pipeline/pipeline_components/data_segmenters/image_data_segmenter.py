from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter

import numpy as np
import torch
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../segment-anything-2/sam2')))
from build_sam import build_sam2_camera_predictor, build_sam2
from automatic_mask_generator import SAM2AutomaticMaskGenerator

from collections import defaultdict
import time
from scipy.optimize import linear_sum_assignment

import uuid
import argparse
import hydra

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

    def __init__(self,
                 model_cfg_path,
                 checkpoint_path,
                 automask_interval=30, 
                 number_of_objects=10,
                 min_mask_region_area=500,
                 device="cuda"
                 ) -> None:
        super().__init__()
        # DO NOT resolve model_cfg_path to absolute path! Pass as config name for Hydra
        # Only resolve checkpoint_path to absolute path if needed
        if not os.path.isabs(checkpoint_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
            checkpoint_path = os.path.join(project_root, checkpoint_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.automask_interval = automask_interval
        self.min_mask_region_area = min_mask_region_area
        # Build SAM2 predictor
        self.predictor = build_sam2_camera_predictor(model_cfg_path, checkpoint_path, device=self.device)
        # Auto mask generator for initialization
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.predictor,
            points_per_side=16,  # Reduced for speed
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=0,  # No cropping for speed
            min_mask_region_area=self.min_mask_region_area
        )
        # Tracking state
        self.last_automask_frame = -automask_interval
        self.track_history = defaultdict(int)  # Track lifetime of objects
        self.if_first_frame = False
        self.number_of_objects = number_of_objects
        self.masks = {}

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires RGB images as input."""
        return ["image","step_nr"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs segmentation masks."""
        return ["segmentation_masks"]
    
    def _run(self, image: Any, step_nr: int, **kwargs: Any) -> Dict[str, Any]:
        """
        Segment an RGB image.
        
        Args:
            image: The input RGB image
            **kwargs: Additional unused arguments
        """
        rgb_image = image.as_numpy()
        # Convert normalized float image [0,1] to 0-255 uint8 if needed
        if rgb_image.dtype in [np.float32, np.float64] and rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        elif rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        if step_nr - self.last_automask_frame >= self.automask_interval:
            # Generate new masks
            auto_masks, processed_bboxes = self._generate_auto_masks(rgb_image)
            if auto_masks:
                self._initialize_tracking(rgb_image, auto_masks, processed_bboxes)
                self.last_automask_frame = step_nr

        self.masks = self._propagate_masks(rgb_image)
        return {"segmentation_masks": self.masks}
    
    
    def _generate_auto_masks(self, frame):
        """Generate automatic masks using SAM2AutomaticMaskGenerator"""

        masks = self.mask_generator.generate(frame)
        
        processed_masks = []
        processed_bboxes = []
        for mask_data in masks[:self.number_of_objects]:  # Limit to top 10 masks
            if mask_data["area"]> self.min_mask_region_area:
                processed_masks.append(mask_data['segmentation'])
                processed_bboxes.append(mask_data['bbox'])
        
        return processed_masks, processed_bboxes
    

    def batch_iou(self, new_masks, old_masks):
        if not new_masks or not old_masks:
            return np.zeros((len(new_masks), len(old_masks)))
        new_stack = np.stack(new_masks).astype(bool)
        old_stack = np.stack(old_masks).astype(bool)
        inter = np.tensordot(new_stack, old_stack, axes=([1,2],[1,2]))
        union = new_stack.sum((1,2))[:, None] + old_stack.sum((1,2))[None, :] - inter
        return inter / (union + 1e-6)

    def match_new_masks(self, new_masks, iou_thresh=0.7):
        old_ids = list(self.masks.keys())
        old_masks = [self.masks[k] > 0 for k in old_ids]
        new_masks_bin = [m > 0 for m in new_masks]

        iou = self.batch_iou(new_masks_bin, old_masks)
        cost = -iou
        row, col = linear_sum_assignment(cost) if old_ids else ([], [])

        matches = {}
        matched = set()
        for i, j in zip(row, col):
            if iou[i, j] >= iou_thresh:
                matches[old_ids[j]] = new_masks[i]
                matched.add(i)

        for i, m in enumerate(new_masks):
            if i not in matched:
                new_id = str(uuid.uuid4())
                matches[new_id] = m

        self.masks = matches  # replace old with current matched masks
        return matches

    def _initialize_tracking(self, frame, masks,bboxes):
        """Initialize tracking with generated masks"""
        # Reset inference state
        if not self.if_first_frame:
            self.predictor.load_first_frame(frame)
            self.if_first_frame = True
            frame_idx = 0

        else: 
            self.predictor.reset_state()
            self.predictor.add_conditioning_frame(frame)
            self.predictor.condition_state["tracking_has_started"] = False
            frame_idx = self.predictor.condition_state["num_frames"] -1
            
        # for obj_id, mask in self.match_new_masks(masks).items():
        #     self.predictor.add_new_mask(
        #         frame_idx=frame_idx,
        #         obj_id=obj_id,
        #         mask=mask
        #     )
        for bbox in bboxes:
            print(bbox.type)
            self.predictor.add_new_prompt(
                frame_idx,
                str(uuid.uuid4()),
                points=None,
                bbox=bbox,
                clear_old_points=True,
                normalize_coords=True,
            )
    

    
    def _propagate_masks(self, frame):
        """Propagate masks to current frame"""

        # Propagate masks
        out_obj_ids, out_mask_logits = self.predictor.track(frame)
        
        # Process outputs
        current_masks = {}
        for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
            mask = (mask_logits > 0.0).cpu().numpy().squeeze()
            if np.sum(mask) > self.min_mask_region_area:  # Minimum area threshold
                current_masks[obj_id] = mask

        return current_masks
