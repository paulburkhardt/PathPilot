from ..abstract_pipeline_component import AbstractPipelineComponent
from typing import Dict, Any
import rerun as rr
import uuid
from .abstract_data_vizualizer import AbstractDataVisualizer

class AbstractRerunDataVisualizer(AbstractDataVisualizer):
    """
    Abstract base class for data visualizer components.
    Made for Rerun in mind
    Args:
        
    Returns:
        -
    
    Raises:
        -
    """
    _rerun_initialized = False # static to allow to be called just ones

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self.setup_rerun()


    def setup_rerun(self) -> None:
        """Initialize Rerun with proper coordinate system.""" 

        if not self._rerun_initialized:
            rr.init("PathPilot", recording_id=uuid.uuid4(), spawn=True)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        self._rerun_initialized = True
        


