from typing import Dict, Any
import os
from ..abstract_pipeline_component import AbstractPipelineComponent
import re

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

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        run_nr = self._get_run_nr(output_dir)
        self._output_dir = os.path.join(output_dir,f"run_{run_nr}")
        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)

    @staticmethod
    def _get_run_nr(output_dir):
        """
        Checks the output_dir for folders of the form run_<run_number>.
        If there exist any, then it evaluates the max run_number and increments it by 1 and returns it.
        """

        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"There exists no {output_dir} directory.")

        run_pattern = re.compile(r"^run_(\d+)$")
        max_run = -1
        for name in os.listdir(output_dir):
            match = run_pattern.match(name)
            if match:
                run_nr = int(match.group(1))
                if run_nr > max_run:
                    max_run = run_nr
        return max_run + 1

    @property
    def output_dir(self) -> str:
        """Get the output directory."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self,output_dir:str):
        """Set the ouptut directory."""
        self._output_dir = output_dir