from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..pipeline_data_bucket import PipelineDataBucket

class AbstractPipelineComponent(ABC):
    """
    Abstract base class for all pipeline components.
    
    Args:
        -

    Returns:
        -
    
    Raises:
        -
    """
    
    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        
    @property
    @abstractmethod
    def inputs_from_bucket(self) -> List[str]:
        """List of data entities this component needs as input."""
        pass
    
    @property
    @abstractmethod
    def outputs_to_bucket(self) -> List[str]:
        """List of data entities this component adds to the bucket."""
        pass
    
    @property
    def config(self) -> Dict[str, Any]:
        """Configuration dictionary for this component."""
        return self._config
    
    @config.setter
    def config(self, config: Dict[str, Any]) -> None:
        self._config = config
    
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the component's run method and handle data writing.
        
        Args:
            *args: Variable positional arguments passed to _run.
            **kwargs: Variable keyword arguments passed to _run.
        Returns:
            Dictionary containing the component's output data.
        """
        
        output = self._run(*args, **kwargs)
        assert type(output) is dict, "Output of _run method needs to be a dictionary."
        return output
    
    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Main execution method to be implemented by concrete components.
        
        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        Returns:
            Dictionary containing the component's output data.
        """
        pass