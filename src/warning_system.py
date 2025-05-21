import time
from typing import Dict, Any, List

class WarningSystem:
    def __init__(self, config: Dict[str, Any]):
        self.message_template = config['message_template']
        self.min_warning_interval = config['min_warning_interval']
        self.last_warning_time = 0

    def generate_warnings(self, 
                         distances: List[Dict[str, Any]], 
                         object_descriptions: List[str]) -> List[str]:
        """
        Generate warnings based on object distances and descriptions.
        
        Args:
            distances: List of object distances
            object_descriptions: List of object descriptions from BLIP-2
            
        Returns:
            List of warning messages
        """
        current_time = time.time()
        warnings = []
        
        # Check if enough time has passed since last warning
        if current_time - self.last_warning_time < self.min_warning_interval:
            return warnings
            
        # Find closest object
        if not distances:
            return warnings
            
        closest_obj = min(distances, key=lambda x: x['distance'])
        
        # Generate warning if object is too close
        if closest_obj['distance'] < 0.3:  # 30cm threshold
            warning = self.message_template.format(
                distance=closest_obj['distance']
            )
            warnings.append(warning)
            self.last_warning_time = current_time
            
        return warnings 