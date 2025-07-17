from typing import List, Dict, Any
from .abstract_object_database import AbstractObjectDatabase

import numpy as np
import torch
from rtree import index
import lietorch
from bidict import bidict
import torch.nn.functional as F
from src.pipeline.data_entities.object_data_entity import ObjectDataEntity

class BBoxObjectDatabase(AbstractObjectDatabase):
    """
    Database component for storing and managing Gaussian objects.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: As this is currently a placeholder
    """
    
    def __init__(self, 
                 match_threshold=1.0, 
                 embedding_weight=0.7,
                 running_embedding_weight = 0.8, 
                 knn_count = 4,
                 use_blip = False, 
                 use_fusion = False,
                 enable_same_mask_tracking = True,
                 track_points = False) -> None:
        super().__init__()

        p = index.Property()
        p.dimension = 3
        # p.dat_extension = 'data'
        # p.idx_extension = 'index'
        self.rtree = index.Index(properties=p, interleaved = False)

        self.match_threshold = match_threshold
        self.embedding_weight = embedding_weight
        self.objects_map = {}  # object_id -> {'points', 'aabb', 'centroid', 'embedding'}
        self.id_map = {}  # mask_id -> object_id
        self.use_blip = use_blip
        self.knn_count = knn_count
        self.use_fusion = use_fusion
        self.running_embedding_weight = running_embedding_weight
        self.obj_id_counter = 1
        self.enable_same_mask_tracking =enable_same_mask_tracking
        self.track_points = track_points

    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires Gaussian object data as input."""
        inputs= ["object_point_cloud","camera_pose","key_frame_flag"]

        if self.use_blip:
            inputs.append("embeddings")
            inputs.append("descriptions")
            inputs.append("segmentation_labels")
        return inputs 
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs database information."""
        return ["objects","object_dict"]
    
    def _run(self, object_point_cloud: Any, camera_pose, key_frame_flag: bool ,embeddings: Dict = None, descriptions: Dict = None, segmentation_labels: Dict = None ,**kwargs: Any) -> Dict[str, Any]:
        """
        Store and manage Gaussian objects in the database.
        
        Args:
            segmented_pointcloud: The input Gaussian object data to store
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """

        if key_frame_flag:
            self.add_frame(object_point_cloud, embeddings, descriptions, segmentation_labels )

        translation = tuple(camera_pose.translation().flatten().cpu().numpy())[:3]
        object_ids = self.query_objects(translation, "KNN")
        objects = [self.objects_map[obj_id] for obj_id in object_ids] #TODO Change to an intersection before the camera
        return {"objects": objects ,
                "object_dict": self.objects_map} 
    
    


    def add_frame(self, segmented_pointclouds, embeddings: Dict = None, descriptions: Dict = None, segmentation_labels: Dict = None):
        if segmented_pointclouds is None: 
            return        
        if embeddings is None:
            embeddings = {}
            descriptions = {}
            segmentation_labels = {}
        updated_objects = []
        for mask_key, segmented_pointcloud in segmented_pointclouds.items():
            embeddings_vector = embeddings.get(mask_key, None)
            description = descriptions.get(mask_key, None)
            label = segmentation_labels.get(mask_key, None)
            if label != None and description != None:
                description = label + ": " + description
            if embeddings_vector is not None:
                embeddings_vector = F.normalize(embeddings_vector, p=2, dim=-1)
            if mask_key in self.id_map:
                if self.enable_same_mask_tracking:
                    self._update_object(mask_key, self.id_map[mask_key],segmented_pointcloud, embeddings_vector, description)
                continue
            
            pointcloud = segmented_pointcloud.as_numpy()
            centroid = pointcloud.mean(axis=0)
            aabb = self._get_aabb(pointcloud)
            obj_id = self._match_object(centroid, aabb, embeddings_vector)

            if obj_id is None:
                self._create_object(mask_key, segmented_pointcloud, centroid, embeddings_vector, description)
            else:
                self._update_object(mask_key, obj_id,segmented_pointcloud, embeddings_vector, description)
                updated_objects.append(obj_id)

        # check if the object if the updated object can be fused with another object
        if not self.use_fusion:
            return
        
        while updated_objects: 
            updated_obj_id = updated_objects.pop(0)
            object = self.objects_map[updated_obj_id]
            matching_obj_id = self._match_object(object.centroid, object.aabb, object.running_embedding, ignore_ID = updated_obj_id)
            if matching_obj_id is not None:
                self.fuse_objects(updated_obj_id, matching_obj_id)
                updated_objects.append(updated_obj_id)

        
    def _match_object(self, centroid, aabb, embeddings_vector, ignore_ID=None):
        object_ids = self.query_objects(centroid, mode="KNN")
        if ignore_ID is not None and object_ids:
            object_ids = [oid for oid in object_ids if oid != ignore_ID]

        if object_ids:
            # Create a bidirectional mapping between object_ids and their indices
            oid_idx_map = bidict({oid: idx for idx, oid in enumerate(object_ids)})
            centroids = np.stack([self.objects_map[oid].centroid for oid in object_ids])
            geo_dists = np.linalg.norm(centroids - centroid, axis=1)
            overlap_dists = np.array([self.overlap_3d(self.objects_map[oid].aabb, aabb) for oid in object_ids])

            if self.use_blip and embeddings_vector is not None:
                embeddings_matrix = torch.stack([self.objects_map[oid].running_embedding for oid in object_ids])
                emb_dists = (1 - (embeddings_matrix @ embeddings_vector.T)).cpu().numpy().flatten()
            else:
                emb_dists = np.ones(len(object_ids))

            scores = (1 - self.embedding_weight) * geo_dists * overlap_dists + self.embedding_weight * emb_dists
            
            scores = scores.flatten()
            
            best_idx = np.argmin(scores)
            best_obj_id = oid_idx_map.inverse[best_idx]
            best_score = float(np.array(scores[best_idx]).flatten()[0])
            if best_score < self.match_threshold:
                return best_obj_id
        return None
            
    def overlap_3d(self,boxA, boxB):
        ax1, ax2, ay1, ay2, az1, az2 = boxA
        bx1, bx2, by1, by2, bz1, bz2 = boxB

        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        iz = max(0, min(az2, bz2) - max(az1, bz1))
        inter_vol = ix * iy * iz

        volA = (ax2 - ax1) * (ay2 - ay1) * (az2 - az1)
        volB = (bx2 - bx1) * (by2 - by1) * (bz2 - bz1)
        
        checkA = inter_vol / volA if volA else 0
        checkB = inter_vol / volB if volB else 0

        return 1 - max(checkA ,checkB )
           


    def _create_object(self, mask_id, segmented_pointcloud, centroid, embeddings_vector = None, description = None):
        obj_id = self.obj_id_counter
        self.obj_id_counter += 1

        self.id_map[mask_id] = obj_id
        obj  = ObjectDataEntity(obj_id= obj_id, 
                                mask_id = mask_id,
                                centroid= centroid,
                                points = segmented_pointcloud,
                                embedding = embeddings_vector,
                                description= description,
                                running_embedding_weight = self.running_embedding_weight,
                                track_points = self.track_points
                                )
        self.objects_map[obj_id] = obj
        self.rtree.insert(obj_id, obj.aabb)


    def _update_object(self, mask_id, obj_id, new_points, new_embedding, description = None):
        self.id_map[mask_id] = obj_id
        obj = self.objects_map[obj_id]

        self.rtree.delete(obj_id, obj.aabb)
        obj.update(mask_id,new_points,new_embedding, description)
        self.rtree.insert(obj_id, obj.aabb)

    def fuse_objects(self, obj_id_1, obj_id_2):
        self.rtree.delete(obj_id_1, self.objects_map[obj_id_1].aabb)
        self.rtree.delete(obj_id_2, self.objects_map[obj_id_2].aabb)

        for mask_id in self.objects_map[obj_id_2].mask_id:
            self.id_map[mask_id] = obj_id_1
        self.objects_map[obj_id_1].fuse(self.objects_map[obj_id_2])
        del self.objects_map[obj_id_2]
        self.rtree.insert(obj_id_1, self.objects_map[obj_id_1].aabb)

    def region_pre_step(self, region):
        match len(region):
            case 3:

                return (region[0],region[0],region[1],region[1],region[2],region[2])
            case 6:
                return tuple(region)
            case _:
                raise ValueError("Wrong dimensions",region.shape)

    def query_objects(self, region, mode = "KNN"):

        bbox = self.region_pre_step(region)
        match mode:
            case "KNN":
                return list(self.rtree.nearest(bbox, self.knn_count))
            case "intersection":
                return list(self.rtree.intersection(bbox))
            case _:
                raise TypeError("Mode not detected")
            
    def _get_aabb(self, points):
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        
        # Validate that min <= max for all dimensions
        if np.any(min_bound > max_bound):
            print(f"Warning: min_bound > max_bound detected!")
            print(f"min_bound: {min_bound}")
            print(f"max_bound: {max_bound}")
            print(f"points shape: {points.shape}")
            print(f"points sample: {points[:5] if len(points) > 5 else points}")
            raise ValueError("Invalid AABB: min_bound > max_bound")
        
        aabb = (min_bound[0], max_bound[0], min_bound[1], max_bound[1], min_bound[2], max_bound[2])
        
        # Additional validation for the final AABB
        if aabb[0] > aabb[1] or aabb[2] > aabb[3] or aabb[4] > aabb[5]:
            print(f"Warning: Invalid AABB coordinates: {aabb}")
            raise ValueError("Invalid AABB coordinates")
        
        return aabb

