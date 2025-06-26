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
# from omegaconf import OmegaConf
class SAM2OnlineSegmentation:
    def __init__(self, automask_interval=30, number_of_objects= 7):
        """
        Initialize SAM2 for online video segmentation
        
        Args:
            model_cfg: SAM2 model config
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to run on
            scale_factor: Downscale factor for processing
            automask_interval: Frames between auto-mask generation
        """


            # Dynamically determine project root (assume this script is always in src/pipeline/pipeline_components/data_segmenters/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../../../'))


        

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=os.path.join(project_root, 'segment-anything-2/sam2/configs'))
        model_cfg = 'sam2.1/sam2.1_hiera_t.yaml'
        checkpoint_path = os.path.join(project_root, 'segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt')

        self.device =  "cuda" if torch.cuda.is_available() else "cpu" 
        self.automask_interval = automask_interval
        self.min_mask_region_area = 500
        # Build SAM2 predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint_path, device=self.device)
        
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
        self.frame_idx = 0
        self.last_automask_frame = - automask_interval 
        self.track_history = defaultdict(int)  # Track lifetime of objects
        self.if_first_frame = False
        self.number_of_objects = number_of_objects
        self.masks = {}
        
    
    def _generate_auto_masks(self, frame):
        """Generate automatic masks using SAM2AutomaticMaskGenerator"""

        masks = self.mask_generator.generate(frame)
        
        processed_masks = []
        for mask_data in masks[:self.number_of_objects]:  # Limit to top 10 masks
            if mask_data["area"]> self.min_mask_region_area:
                processed_masks.append(mask_data['segmentation'])
        
        return processed_masks
    

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

    def _initialize_tracking(self, frame, masks):
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
            
        for obj_id, mask in self.match_new_masks(masks).items():
            self.predictor.add_new_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask
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
    
    def run(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame (H, W, 3) numpy array
            
        Returns:
            dict: {obj_id: mask} where mask is (H, W) boolean array
        """

        # Check if we need to generate new auto masks

        if self.frame_idx - self.last_automask_frame >= self.automask_interval:
            # Generate new masks
            auto_masks = self._generate_auto_masks(frame)
            if auto_masks:
                self._initialize_tracking(frame, auto_masks)
                self.last_automask_frame = self.frame_idx

        self.masks = self._propagate_masks(frame)
        self.frame_idx += 1
        return self.masks
    
    def get_visualization(self, frame, masks):
        """Create visualization of masks"""
        vis = frame.copy()
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




    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--detect_stride", type=int, default=30)
    parser.add_argument("--postnms", type=int, default=0)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.9)#0.7
    parser.add_argument("--box_nms_thresh", type=float, default=0.9)#0.7
    parser.add_argument("--stability_score_thresh", type=float, default=0.95) #0.85 default
    args = parser.parse_args()
    # logger.add(os.path.join(args.output_dir, 'sam2.log'), rotation="500 MB")
    # logger.info(args)



    video_dir = args.video_path
    base_dir = args.output_dir

    segmenter = SAM2OnlineSegmentation(
        scale_factor=0.5,  # Process at half resolution
        automask_interval=30  # Generate new masks every 30 frames
    )

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for frame_name in frame_names:
        image_path = os.path.join(video_dir, frame_name)
        image = cv2.imread(image_path)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Run segmentation
        start_time = time.time()
        masks = segmenter.run(frame)
        process_time = time.time() - start_time
        
        # Visualize results
        vis_frame = segmenter.get_visualization(frame, masks)
                # Visualize results
        cv2.imwrite(os.path.join(base_dir,f"now_frame_{frame_name}.jpg"), vis_frame)
        fps = 1.0 / process_time if process_time > 0 else 0
        print(fps)