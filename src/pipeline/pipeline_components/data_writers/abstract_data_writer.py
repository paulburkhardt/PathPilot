from typing import Dict, Any
import os
from ..abstract_pipeline_component import AbstractPipelineComponent

class AbstractDataWriter(AbstractPipelineComponent):
    """
    Abstract base class for data writers.
    
    Args:
        output_dir: Directory where output files will be written.
    
    Returns:
        -
    
    Raises:
        -
    """
    
    def __init__(self, output_dir: str = None) -> None:
        super().__init__() 
        self._output_dir = output_dir

    @property
    def output_dir(self) -> str:
        """Get the output directory."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self,output_dir:str):
        """Set the ouptut directory."""
        self._output_dir = output_dir