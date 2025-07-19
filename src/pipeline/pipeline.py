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
        
        # Start timing the entire pipeline
        pipeline_start_time = time.time()
        print(f"🚀 Starting pipeline execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pipeline_start_time))}")
        
        dataset_component = self.components[0]
        total_frames = len(dataset_component)
        
        print(f"📊 Processing {total_frames} frames through {len(self.components)} components")
        
        for i in tqdm(range(total_frames), desc="Processing dataset"):

            data_entity = dataset_component[i]

            bucket = PipelineDataBucket()
            bucket.put({
                "step_nr": i,
                "total_steps": total_frames,
                **data_entity  
            })

            for component in self.components[1:]:                    

                inputs = bucket.get(*component.inputs_from_bucket)
                outputs = component(**inputs)
                bucket.put(outputs)

                    
        # Calculate and log total pipeline execution time
                        
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time                
        
        print(f"✅ Pipeline execution completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pipeline_end_time))}")
        print(f"⏱️  Total pipeline execution time: {self._format_duration(total_duration)}")
        print(f"📈 Average time per frame: {total_duration / total_frames:.2f} seconds")
        print(f"🎯 Processing rate: {total_frames / total_duration:.2f} frames/second")
    
    def _format_duration(self, duration_seconds: float) -> str:
        """
        Format duration in a human-readable format.
        
        Args:
            duration_seconds: Duration in seconds
        Returns:
            Formatted duration string
        """
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds:.2f}s"
        elif minutes > 0:
            return f"{minutes}m {seconds:.2f}s"
        else:
            return f"{seconds:.2f}s"