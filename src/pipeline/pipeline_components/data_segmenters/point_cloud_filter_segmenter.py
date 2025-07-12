from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter
import numpy as np
from sklearn.cluster import DBSCAN
from src.pipeline.data_entities.point_cloud_data_entity import PointCloudDataEntity


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
        return ["point_cloud","key_frame_flag"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs segmented point clouds."""
        return ["object_point_cloud", "point_cloud", ]
    
    def _run(self, point_cloud: Any, key_frame_flag, **kwargs: Any) -> Dict[str, Any]:
        """
        Segment a point cloud.
        
        Args:
            point_cloud: The input point cloud
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """
        # if key_frame_flag:
        #     return {"object_point_cloud": None, 
        #             "point_cloud": point_cloud}
        
        pointcloud = point_cloud.as_numpy(with_rgb = True, 
                                          with_confidence_score = True ,
                                          with_segmentation_mask=True)[::self.downsample_rate] #when accumalation when 
        ids = pointcloud[:, -1]
        sorted_pc = pointcloud[np.argsort(ids)]
        unique_ids, counts = np.unique(ids, return_counts=True)
        # splits = np.cumsum(counts)[:-1]
        splits = np.cumsum(counts)[:-1]
        

        object_pointclouds = {}
        all_points = []
        for obj_id, arr in zip(unique_ids, np.split(sorted_pc, splits)):
            points = arr[:, :-1]
            # Remove all-zero points
            if len(points) > 0:
                db = DBSCAN(eps= self.neighbor_distance, min_samples=5).fit(points[:,:3])
                labels = db.labels_
                unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
                valid_labels = unique_labels[label_counts >= self.min_cluster_size]
                mask = np.isin(labels, valid_labels)
                filtered_points = points[mask]


                if len(filtered_points) > 0:
                    object_pointclouds[int(obj_id)] = filtered_points[:,:3]

                    # Re-attach the object id as last column for each point
                    obj_ids_col = np.full((filtered_points.shape[0], 1), obj_id)
                    all_points.append(np.hstack([filtered_points, obj_ids_col]))

        combined_pointcloud = None
        if all_points:
            combined_points = np.vstack(all_points)
            combined_pointcloud = PointCloudDataEntity(combined_points[:,:3],
                                                       combined_points[:,3:6],
                                                       combined_points[:,6].reshape(-1, 1),
                                                       combined_points[:,-1].astype(np.int32).reshape(-1, 1))

        return {"object_point_cloud": object_pointclouds, 
                "point_cloud": combined_pointcloud}
    
    
    
    
    
    
     