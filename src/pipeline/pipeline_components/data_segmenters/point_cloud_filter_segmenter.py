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
                neighbor_distance:  float = 0.05,
                ignore_backround : bool = True,
                rewrite_pointcloud: bool = False 
                ) -> None:
        super().__init__()
        self.downsample_rate = downsample_rate # needs to be int
        self.min_cluster_size = min_cluster_size
        self.neighbor_distance = neighbor_distance
        self.ignore_backround = ignore_backround
        self.rewrite_pointcloud = rewrite_pointcloud
        self.backround = None
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point clouds as input."""
        return ["point_cloud","key_frame_flag"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs segmented point clouds."""
        output = ["object_point_cloud" ]
        if self.rewrite_pointcloud:
            output.append("point_cloud" ) 
        if not self.ignore_backround:
            output.append("backround" ) 
        return output
    
    def _run(self, point_cloud: Any, key_frame_flag, **kwargs: Any) -> Dict[str, Any]:
        """
        Segment a point cloud.
        
        Args:
            point_cloud: The input point cloud
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """
        if not key_frame_flag:
            output = {"object_point_cloud": None }
            if self.rewrite_pointcloud:
                output["point_cloud"]  = point_cloud
            if not self.ignore_backround:
                output["backround"]  = self.backround
            return output
 
        
        pointcloud = point_cloud.as_numpy(with_rgb = True, 
                                          with_confidence_score = True ,
                                          with_segmentation_mask=True)[::self.downsample_rate] #when accumalation when 
        ids = pointcloud[:, -1]
        sorted_pc = pointcloud[np.argsort(ids)]
        unique_ids, counts = np.unique(ids, return_counts=True)
        print("unique ids of the pointcloud")
        print(unique_ids)
        # splits = np.cumsum(counts)[:-1]
        splits = np.cumsum(counts)[:-1]
        object_pointclouds = {}
        all_points = []


        for obj_id, arr in zip(unique_ids, np.split(sorted_pc, splits)):

            points = arr[:, :-1]

            if obj_id == 0 and not self.ignore_backround: #NOTE its good to do this since we also dont have a mask for it
                self.backround = self.create_pointcloud_entity(arr)  
                continue

            # Remove all-zero points
            if len(points) > 0:
                db = DBSCAN(eps= self.neighbor_distance, min_samples=5).fit(points[:,:3])
                labels = db.labels_
                unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
                valid_labels = unique_labels[label_counts >= self.min_cluster_size]
                mask = np.isin(labels, valid_labels)
                filtered_points = points[mask]


                if len(filtered_points) > 0:
                    obj_ids_col = np.full((filtered_points.shape[0], 1), obj_id)
                    filtered_points_together = np.hstack([filtered_points, obj_ids_col])
                    object_pointclouds[int(obj_id)]= self.create_pointcloud_entity(filtered_points)
                    # Re-attach the object id as last column for each point
                    all_points.append(filtered_points_together)

        combined_pointcloud = None
        if self.rewrite_pointcloud and all_points:
            combined_points = np.vstack(all_points)
            combined_pointcloud = self.create_pointcloud_entity(combined_points)


        output = {"object_point_cloud": object_pointclouds }
        if self.rewrite_pointcloud:
            output["point_cloud"]  = combined_pointcloud
        if not self.ignore_backround:
            output["backround"]  = self.backround
        return output
    
    
    
    def create_pointcloud_entity(self,points):
        return   PointCloudDataEntity(  points[:,:3],
                                        points[:,3:6],
                                        points[:,6].reshape(-1, 1),
                                        points[:,-1].astype(np.int32).reshape(-1, 1))

     