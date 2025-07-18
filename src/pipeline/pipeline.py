from typing import List, Dict, Any, Optional
from .pipeline_data_bucket import PipelineDataBucket
from .pipeline_components.abstract_pipeline_component import AbstractPipelineComponent

from tqdm import tqdm
import time

class Pipeline:
    """
    Main pipeline class that orchestrates the execution of pipeline components.
    
    Args:
        components: List of pipeline components to execute in sequence.
    
    Returns:
        -
    
    Raises:
        ValueError: If pipeline validation fails
    """
    
    def __init__(self, components: List[AbstractPipelineComponent]) -> None:
        self.components = components
        self.full_config: Optional[Dict[str, Any]] = None
        self.validate()
    
    def validate(self) -> None:
        """
        Validate that all required inputs will be available for each component.
        
        Args:
            -
        Returns:
            -
        Raises:
            ValueError: If a required input won't be available for any component
        """
        available_outputs = set(["step_nr","total_steps"])
        
        for i, component in enumerate(self.components):
            missing_inputs = set(component.inputs_from_bucket) - available_outputs
            if missing_inputs:
                raise ValueError(
                    f"Component {i} ({component.__class__.__name__}) requires inputs {missing_inputs} "
                    f"but they won't be available. Available outputs are {available_outputs}"
                )
            
            available_outputs.update(component.outputs_to_bucket)
    
    def run(self) -> None:
        """
        Execute all pipeline components in sequence.
        
        Args:
            -
        Returns:
            -
        """
        
        dataset_component = self.components[0]
        for i in tqdm(range(len(dataset_component)), desc="Processing dataset"):
            start_time = time.time()
            data_entity = dataset_component[i]

            bucket = PipelineDataBucket()
            bucket.put({
                "step_nr": i,
                "total_steps": len(dataset_component),
                **data_entity  
            })

            for component in self.components[1:]:
                step_start_time = time.time()
                inputs = bucket.get(*component.inputs_from_bucket)
                outputs = component(**inputs)
                bucket.put(outputs)
                print(f"[Timing] Component {i} - {component.__class__.__name__} took {time.time() - step_start_time:.2f} seconds")
            print(f"[Timing] Step run took {time.time() - start_time:.2f} seconds in total.")