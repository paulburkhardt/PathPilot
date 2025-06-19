from typing import List, Dict, Any
from .pipeline_data_bucket import PipelineDataBucket
from .pipeline_components.abstract_pipeline_component import AbstractPipelineComponent

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
        self.validate()
    
    def validate(self) -> None:
        """
        Validate that all required inputs will be available for each component.
        Optional inputs are not validated since they may become available later.
        
        Args:
            -
        Returns:
            -
        Raises:
            ValueError: If a required input won't be available for any component
        """
        available_outputs = set(["step_nr"])
        
        for i, component in enumerate(self.components):
            required_inputs = set(component.inputs_from_bucket)
            optional_inputs = set(component.optional_inputs_from_bucket)
            
            # Only check required inputs, not optional ones
            missing_inputs = required_inputs - available_outputs
            if missing_inputs:
                raise ValueError(
                    f"Component {i} ({component.__class__.__name__}) requires inputs {missing_inputs} "
                    f"but they won't be available. Available outputs are {available_outputs}. "
                    f"Optional inputs {optional_inputs} are not checked during validation."
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
        
        #pipeline assumes the first component within the pipeline to be iterable 
        iterator_component = self.components[0]
        for i,data_entity in enumerate(iterator_component):

            bucket = PipelineDataBucket()
            bucket.put({
                "step_nr": i,
                **data_entity  
            })

            for component in self.components[1:]:

                # Get required inputs
                required_inputs = component.inputs_from_bucket
                optional_inputs = component.optional_inputs_from_bucket
                
                if optional_inputs:
                    inputs = bucket.get_with_optional(required_inputs, optional_inputs)
                else:
                    inputs = bucket.get(*required_inputs)
                    
                outputs = component(**inputs)
                bucket.put(outputs)
