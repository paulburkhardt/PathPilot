from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter
import numpy as np
from sklearn.cluster import DBSCAN

class PointCloudFilterSegmenter(AbstractDataSegmenter):
    """
    Component for segmenting point cloud data.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: As this is currently a placeholder
    """

    def __init__(self, 
                downsample_rate : int = 4,
                min_cluster_size: int = 100,
                neighbor_distance:  float = 0.05
                ) -> None:
        super().__init__()
        self.downsample_rate = downsample_rate # needs to be int
        self.min_cluster_size = min_cluster_size
        self.neighbor_distance = neighbor_distance
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point clouds as input."""
        return ["point_cloud"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs segmented point clouds."""
        return ["object_point_cloud"]
    
    def _run(self, point_cloud: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Segment a point cloud.
        
        Args:
            point_cloud: The input point cloud
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """

        pointcloud = point_cloud.as_numpy(with_segmentation_mask=True)[::self.downsample_rate]
        ids = pointcloud[:, -1]
        sorted_pc = pointcloud[np.argsort(ids)]
        unique_ids, counts = np.unique(ids, return_counts=True)
        splits = np.cumsum(counts)[:-1]

        object_pointclouds = {}
        for obj_id, arr in zip(unique_ids, np.split(sorted_pc, splits)):
            points = arr[:, :-1]
            if len(points) > 0:
                db = DBSCAN(eps= self.neighbor_distance, min_samples=5).fit(points)
                labels = db.labels_
                # Count points in each cluster (excluding outliers)
                unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
                # Find clusters that are large enough
                valid_labels = unique_labels[label_counts >= self.min_cluster_size]
                # Create a mask for points in valid clusters
                mask = np.isin(labels, valid_labels)
                filtered_points = points[mask]
                if len(filtered_points) > 0:
                    object_pointclouds[int(obj_id)] = filtered_points
        return {"object_point_cloud":object_pointclouds}
    
    
    
    
    
     