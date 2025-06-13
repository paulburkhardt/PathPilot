from typing import Dict, Any, Type
from .pipeline import Pipeline
from .pipeline_components.slam_components.mast3r_slam_component import MAST3RSLAMComponent
from .pipeline_components.data_writers.point_cloud_data_writer import PointCloudDataWriter
from .pipeline_components.data_loaders.mast3r_slam_video_data_loader import MAST3RSLAMVideoDataLoader

# Import new modular components
from .pipeline_components.data_loaders.trajectory_data_loader import TrajectoryDataLoader
from .pipeline_components.data_loaders.ply_point_cloud_loader import PLYPointCloudLoader
from .pipeline_components.data_segmenters.floor_detection_component import FloorDetectionComponent
from .pipeline_components.object_extractors.closest_point_finder_component import ClosestPointFinderComponent
from .pipeline_components.data_visualizer.point_cloud_data_vizualizer import PointCloudDataVisualizer
from .pipeline_components.data_visualizer.camera_trajectory_vizualizer import CameraTrajectoryVisualizer

class PipelineBuilder:
    """
    Builder class for constructing Pipeline instances from configuration.
    
    Args:
        -
    
    Returns:
        -
    
    Raises:
        ValueError: If component type is not found or configuration is invalid
    """
    
    COMPONENT_MAP = {
        # Original components
        "MAST3RSLAMComponent": MAST3RSLAMComponent,
        "PointCloudDataWriter": PointCloudDataWriter,
        "MAST3RSLAMVideoDataLoader": MAST3RSLAMVideoDataLoader,
        
        # New modular components from process_slam_output.py
        "TrajectoryDataLoader": TrajectoryDataLoader,
        "PLYPointCloudLoader": PLYPointCloudLoader,
        "FloorDetectionComponent": FloorDetectionComponent,
        "ClosestPointFinderComponent": ClosestPointFinderComponent,
        "PointCloudDataVisualizer": PointCloudDataVisualizer,
        "CameraTrajectoryVisualizer": CameraTrajectoryVisualizer,
        
        # Add other components here as they are implemented
    }
    
    @classmethod
    def build(cls, config: Dict[str, Any]) -> Pipeline:
        """
        Build a Pipeline instance from configuration.
        
        Args:
            config: Configuration dictionary defining the pipeline structure
        Returns:
            Configured Pipeline instance
        Raises:
            ValueError: If configuration is invalid or component type is not found
        """

        if "pipeline" not in config:
            raise ValueError("Configuration must contain a 'pipeline' section")
        
        pipeline_config = config["pipeline"]
        components = []
        
        for component_config in pipeline_config.get("components", []):
            component_type = component_config.get("type")
            if not component_type:
                raise ValueError("Each component must specify a 'type'")
                
            component_class = cls.COMPONENT_MAP.get(component_type)
            if not component_class:
                raise ValueError(f"Unknown component type: {component_type}")
                
            # Create component instance
            component_config_params = component_config.get("config", {})
            component = component_class(**component_config_params)
            
            # Set component configuration
            component.config = component_config
            components.append(component)
            
        return Pipeline(components)
