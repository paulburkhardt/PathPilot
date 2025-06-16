from typing import List, Dict, Any
from .pipeline_data_bucket import PipelineDataBucket
from .pipeline_components.abstract_pipeline_component import AbstractPipelineComponent

from tqdm import tqdm

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
        
        #pipeline assumes the first component within the pipeline to be iterable 
        dataset_component = self.components[0]
        #for i,data_entity in enumerate(iterator_component):
#        for i,data_entity in tqdm(enumerate(iterator_component),desc="Running pipeline",total=len(iterator_component)):

        for i in tqdm(range(len(dataset_component)), desc="Processing dataset"):

            data_entity = dataset_component[i]

            bucket = PipelineDataBucket()
            bucket.put({
                "step_nr": i,
                "total_steps": len(dataset_component),
                **data_entity  
            })

            for component in self.components[1:]:

                inputs = bucket.get(*component.inputs_from_bucket)
                outputs = component(**inputs)
                bucket.put(outputs)