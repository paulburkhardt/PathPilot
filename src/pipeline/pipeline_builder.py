from typing import Dict, Any, Type
from .pipeline import Pipeline
from .pipeline_components.slam_components.mast3r_slam_component import MAST3RSLAMComponent
from .pipeline_components.data_writers.point_cloud_data_writer import PointCloudDataWriter
from .pipeline_components.data_loaders.mast3r_slam_video_data_loader import MAST3RSLAMVideoDataLoader

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
        "MAST3RSLAMComponent": MAST3RSLAMComponent,
        "PointCloudDataWriter": PointCloudDataWriter,
        "MAST3RSLAMVideoDataLoader": MAST3RSLAMVideoDataLoader
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
            component = component_class(**component_config.get("config"))
            
            # Set component configuration
            component.config = component_config
            components.append(component)
            
        return Pipeline(components)
