from typing import List, Dict, Any
import numpy as np
from .abstract_object_database import AbstractObjectDatabase
from rtree import index




class PointCloudObjectDatabase(AbstractObjectDatabase):
    """
    Database component for storing and managing point cloud objects.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: As this is currently a placeholder
    """
    
    def __init__(self, downsample: float = 1.0) -> None:
        super().__init__()

        self.downsample = downsample
        #self._object_database: Dict[str, Any] = {}
        p = index.Property()
        p.dimension = 3
        self.database = index.Index('3d_index',properties=p)
        self.subsampled_points_dir  = {}



    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires object data as input."""
        return ["point_cloud"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs database information."""
        return ["database"]
    
    def _run(self, point_cloud: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Store and manage point cloud objects in the database.
        
        Args:
            point_cloud: The input object data to store
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """
        if point_cloud.new_keyframe:

            self.delete_point_cloud(point_cloud)
            self.insert_pointcloud(point_cloud)

        return self.database

        
    
    def delete_point_cloud(self,point_cloud):
        for frame_key in point_cloud.current_relevant_frames():
            for point in self.subsampled_points_dir[frame_key]:
                self.database.delete(frame_key, (point[0], point[1], point[2], point[0], point[1], point[2]))
            del self.subsampled_points_dir[frame_key]


    def insert_pointcloud(self, point_cloud):
        for frame_key in point_cloud.current_relevant_frames():
            raw_points_for_frame = point_cloud.get_points(frame_key)
            
            subsampled_points_list = []

            if self.downsample < 1.0:
                for point in raw_points_for_frame:
                    if np.random.rand() < self.downsample:
                        self.database.insert(frame_key, (point[0], point[1], point[2], point[0], point[1], point[2]))
                        subsampled_points_list.append(point)
            else:
                for point in raw_points_for_frame:
                    self.database.insert(frame_key, (point[0], point[1], point[2], point[0], point[1], point[2]))
                    subsampled_points_list.append(point)
            
            self.subsampled_points_dir[frame_key] = np.array(subsampled_points_list)


#     retrieval_inds = retrieval_database.update(
#     frame,
#     add_after_query=False,  # Set to False to query existing keyframes without adding the current frame yet
#     k=config["retrieval"]["k"],  # Number of nearest neighbors to retrieve
#     min_thresh=config["retrieval"]["min_thresh"],  # Minimum similarity threshold
# )