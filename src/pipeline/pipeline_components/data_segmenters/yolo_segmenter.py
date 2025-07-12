from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter
import numpy as np
import os

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package is required for YOLOSegmenter. Install with: pip install ultralytics")


class YOLOSegmenter(AbstractDataSegmenter):
    """
    Component for segmenting image data using YOLO object detection.
    
    This component runs YOLO on every n-th frame and outputs bounding boxes
    and class labels for detected objects.
    
    Args:
        model_path: Path to the YOLO model file (e.g., "yolov8s.pt", "yolov8s-oiv7.pt")
        detection_interval: Run YOLO detection every N frames (default: 1)
        conf_threshold: Confidence threshold for detections (default: 0.25)
        iou_threshold: IoU threshold for NMS (default: 0.5)
        max_detections: Maximum number of detections per image (default: 100)
    
    Returns:
        Dictionary with detected object labels as keys and bounding boxes as values
    
    Raises:
        ImportError: If ultralytics package is not installed
        FileNotFoundError: If model file is not found
    """

    def __init__(self,
                 model_path: str = "models/yolov8s-oiv7.pt",
                 detection_interval: int = 1,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.5,
                 max_detections: int = 20) -> None:
        super().__init__()
        
        self.model_path = model_path
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = "cuda" if self._is_cuda_available() else "cpu"
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load YOLO model from {model_path}: {e}")
        
        # Track when last detection was run
        self.last_detection_frame = -detection_interval

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires RGB images and step number as input."""
        return ["image", "step_nr"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs object detection results."""
        return ["yolo_detections"]
    
    def _run(self, image: Any, step_nr: int, **kwargs: Any) -> Dict[str, Any]:
        """
        Run YOLO object detection on the input image.
        
        Args:
            image: The input RGB image (ImageDataEntity)
            step_nr: Current step/frame number
            **kwargs: Additional unused arguments
            
        Returns:
            Dictionary containing YOLO detection results
        """
        # Check if we should run detection on this frame
        if step_nr - self.last_detection_frame < self.detection_interval:
            # Return empty results if not time for detection
            return {"yolo_detections": {}}
        
        # Convert image to numpy array
        rgb_image = image.as_numpy()
        
        # Convert normalized float image [0,1] to 0-255 uint8 if needed
        if rgb_image.dtype in [np.float32, np.float64] and rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        elif rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        # Run YOLO inference
        try:
            results = self.model(
                rgb_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False  # Suppress verbose output
            )
            
            # Process detection results
            detections = self._process_yolo_results(results[0])
            
            # Update last detection frame
            self.last_detection_frame = step_nr
            
            return {"yolo_detections": detections}
            
        except Exception as e:
            print(f"Error during YOLO inference at step {step_nr}: {e}")
            return {"yolo_detections": {}}
    
    def _process_yolo_results(self, result) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process YOLO detection results into a structured format.
        
        Args:
            result: YOLO result object
            
        Returns:
            Dictionary with class names as keys and lists of detection info as values
        """
        detections = {}
        
        if len(result.boxes) == 0:
            return detections
        
        # Extract detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        class_names = result.names  # Class name mapping
        
        # Group detections by class name
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            class_name = class_names[int(cls_id)]
            
            # Create detection info
            detection_info = {
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(conf),
                "class_id": int(cls_id),
                "detection_id": i
            }
            
            # Add to detections dictionary
            if class_name not in detections:
                detections[class_name] = []
            detections[class_name].append(detection_info)
        
        return detections

