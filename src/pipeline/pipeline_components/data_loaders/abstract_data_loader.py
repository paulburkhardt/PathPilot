from typing import Any, Iterator
from ..abstract_pipeline_component import AbstractPipelineComponent

class AbstractDataLoader(AbstractPipelineComponent):
    """
    Abstract base class for data loaders.
    
    Args:
        -
    Returns:
        -
    Raises:
        -
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._dataloader: Any = None
    
    @property
    def dataloader(self) -> Any:
        """The underlying data loader instance."""
        return self._dataloader
    
    def __iter__(self) -> Iterator[Any]:
        """
        Make the data loader iterable.
        
        Returns:
            Iterator over the underlying data loader
        """
        return iter(self._dataloader)
