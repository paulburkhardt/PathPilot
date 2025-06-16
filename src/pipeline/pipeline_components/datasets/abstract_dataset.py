from typing import Any, Iterator
from ..abstract_pipeline_component import AbstractPipelineComponent

class AbstractDataset(AbstractPipelineComponent):
    """
    Abstract base class for datasets.
    
    Args:
        -
    Returns:
        -
    Raises:
        -
    """
    
    def __init__(self) -> None:
        super().__init__()