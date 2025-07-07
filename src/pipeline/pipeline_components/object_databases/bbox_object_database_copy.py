from typing import List, Dict, Any
from .abstract_object_database import AbstractObjectDatabase

import numpy as np
from rtree import index
import uuid
import torch
from PIL import Image
from transformers import BlipProcessor, BlipModel
import lietorch


class Object3D:
    def __init__(self, 
                 obj_id,
                 points, 
                 centroid, 
                 description = None,
                 embedding=None
                 ):
        self.id = obj_id
        self.centroid = centroid
        self.description = description
        self.aabb = self.get_aabb(points)
        self.embeddings = np.expand_dims(embedding, axis=0)

        self.cum_sum = np.sum(points,axis= 0).astype(float) 
        self.cum_len = points.shape[0]


    def update(self,new_points, new_embedding):
        self.cum_sum += np.sum(new_points, axis= 0).astype(float) 
        self.cum_len += new_points.shape[0]

        self.centroid = self.cum_sum / self.cum_len
        aabb = self.get_aabb(new_points)
        self.aabb = self.update_aabb(self.aabb,aabb)
        self.embeddings = np.vstack([self.embeddings,new_embedding])


    def fuse(self, obj):
        self.cum_sum += obj.cum_sum
        self.cum_len += obj.cum_len

        self.centroid = self.cum_sum / self.cum_len
        self.aabb = self.update_aabb(self.aabb, obj.aabb) 
        self.embeddings = np.vstack([self.embeddings,obj.embeddings])

    def get_aabb(self, points):
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        return (min_bound[0], max_bound[0],min_bound[1], max_bound[1],min_bound[2], max_bound[2])

    def update_aabb(self, aabb1,aabb2): 
        return (
            min(aabb1[0], aabb2[0]), max(aabb1[1], aabb2[1]),
            min(aabb1[2], aabb2[2]), max(aabb1[3], aabb2[3]),
            min(aabb1[4], aabb2[4]), max(aabb1[5], aabb2[5])
        )   

    
class GaussianObjectDatabase(AbstractObjectDatabase):
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
                 use_blip = False, 
                 knn_count = 5) -> None:
        super().__init__()

        p = index.Property()
        p.dimension = 3
        # p.dat_extension = 'data'
        # p.idx_extension = 'index'
        self.rtree = index.Index(properties=p)

        self.match_threshold = match_threshold
        self.embedding_weight = embedding_weight
        self.overlap_weight = 0.3
        self.objects_map = {}  # object_id -> {'points', 'aabb', 'centroid', 'embedding'}
        self.id_map = {}  # mask_id -> object_id
        self.use_blip = use_blip
        self.knn_count = knn_count

        self.obj_id_counter = 1
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires Gaussian object data as input."""
        return ["segmented_pointcloud","camera_pose"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs database information."""
        return ["database_info"]
    
    def _run(self, segmented_pointcloud: Any, camera_pose, **kwargs: Any) -> Dict[str, Any]:
        """
        Store and manage Gaussian objects in the database.
        
        Args:
            segmented_pointcloud: The input Gaussian object data to store
            **kwargs: Additional unused arguments
        Raises:
            NotImplementedError: As this is currently a placeholder
        """
        self.add_frame(segmented_pointcloud)
        translation = lietorch.SE3(camera_pose).translation()
        return self.query_objects(translation, self.knn_count)


    def add_frame(self, segmented_pointclouds, embeddings = {}):
 
        for mask_key, segmented_pointcloud in segmented_pointclouds.items():

            embeddings_vector = embeddings.get(mask_key, None)
            if mask_key in self.id_map:
                self._update_object(mask_key, self.id_map[mask_key],segmented_pointcloud, embeddings_vector)
                continue

            centroid = segmented_pointcloud.mean(axis=0)
            object_ids = self.query_objects(centroid, mode = "KNN")
            
            if object_ids:
                centroids = np.stack([self.objects_map[oid].centroid for oid in object_ids])
                embeddings_matrix = np.stack([self.objects_map[oid].embeddings for oid in object_ids])


                geo_dists = np.linalg.norm(centroids - centroid, axis=1)

                if self.use_blip  and embeddings_vector is not None:
                    emb_sims = embeddings_matrix @ embeddings_vector
                    emb_dists = 1 - emb_sims
                else:
                    emb_dists = np.ones(len(object_ids))

                 # Calculate overlap distances
                overlap_dists = np.array([
                    self.overlap_3d(self.objects_map[oid].aabb, self._get_aabb(segmented_pointcloud))
                    for oid in object_ids
                ])

                # Combine all distances into a score
                scores = (1 - self.embedding_weight) * geo_dists * overlap_dists + self.embedding_weight * emb_dists

                best_idx = np.argmin(scores)
                best_score = float(np.array(scores[best_idx]).flatten()[0])
                if best_score < self.match_threshold:
                    self._update_object(mask_key, object_ids[best_idx],segmented_pointcloud, embeddings_vector)
                    continue

            self._create_object(mask_key, segmented_pointcloud, centroid, embeddings_vector)
            
            
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
           


    def _create_object(self, mask_id, segmented_pointcloud, centroid, embeddings_vector = None):
        obj_id = self.obj_id_counter
        self.obj_id_counter += 1

        self.id_map[mask_id] = obj_id
        obj  = Object3D(obj_id= obj_id, 
                        centroid= centroid, 
                        points = segmented_pointcloud,
                        embedding = embeddings_vector)
        self.objects_map[obj_id] = obj
        self.rtree.insert(obj_id, obj.aabb)


    def _update_object(self, mask_id, obj_id, new_points, new_embedding):
        self.id_map[mask_id] = obj_id
        obj = self.objects_map[obj_id]

        self.rtree.delete(obj_id, obj.aabb)
        obj.update(new_points,new_embedding)
        self.rtree.insert(obj_id, obj.aabb)

    def region_pre_step(self, region):
        match len(region):
            case 3:
                return (region[0],region[0],region[1],region[1],region[2],region[2])
            case 6:
                return tuple(region)
            case _:
                raise ValueError("Wrong dimensions")

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
        return (min_bound[0], max_bound[0],min_bound[1], max_bound[1],min_bound[2], max_bound[2])

# def test_object_tracker3d_basic():
#     # Create tracker
#     tracker = ObjectTracker3D(match_threshold=0.5, embedding_weight=0.5, use_blip=True, knn_count=1)

#     # Create two simple point clouds
#     pc1 = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
#     pc2 = np.array([[0.1, 0, 0], [1.1, 1, 1], [0, 1.1, 0]])
#     emb1 = np.array([1.0, 0.0, 0.0])
#     emb2 = np.array([0.9, 0.1, 0.0])

#     # Add first frame
#     tracker.add_frame({'mask1': pc1}, {'mask1': emb1})
#     assert len(tracker.objects_map) == 1, "Should have 1 object after first frame"

#     # Add second frame (should match the first object)
#     tracker.add_frame({'mask1': pc2}, {'mask1': emb2})
#     assert len(tracker.objects_map) == 1, "Should still have 1 object after matching frame"

#     # Add a very different point cloud (should create a new object)
#     pc3 = np.array([[10, 10, 10], [11, 11, 11], [10, 11, 10]])
#     emb3 = np.array([0.0, 1.0, 0.0])
#     tracker.add_frame({'mask2': pc3}, {'mask2': emb3})
#     assert len(tracker.objects_map) == 2, "Should have 2 objects after adding a distant point cloud"

#     print("All basic ObjectTracker3D tests passed.")

# def test_object_tracker3d_edge_cases():
#     tracker = ObjectTracker3D(match_threshold=0.5, embedding_weight=0.5, use_blip=True, knn_count=1)

#     # Initial point cloud and embedding
#     pc1 = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
#     emb1 = np.array([1.0, 0.0, 0.0])
#     tracker.add_frame({'mask1': pc1}, {'mask1': emb1})
#     print("After first add:", tracker.objects_map.keys())

#     # New mask ID, similar point cloud (should create a new object)
#     pc2 = np.array([[0.1, 0, 0], [1.1, 1, 1], [0, 1.1, 0]])
#     emb2 = np.array([0.9, 0.1, 0.0])
#     tracker.add_frame({'mask2': pc2}, {'mask2': emb2})
#     print("After similar but new mask:", tracker.objects_map.keys())

#     # Existing mask ID, very different point cloud (should update the object)
#     pc3 = np.array([[10, 10, 10], [11, 11, 11], [10, 11, 10]])
#     emb3 = np.array([0.0, 1.0, 0.0])
#     tracker.add_frame({'mask1': pc3}, {'mask1': emb3})
#     print("After very different pc with same mask:", tracker.objects_map.keys())

#     # New mask ID, but point cloud close to first object (should match if logic allows)
#     pc4 = np.array([[0.05, 0, 0], [1.05, 1, 1], [0, 1.05, 0]])
#     emb4 = np.array([0.95, 0.05, 0.0])
#     tracker.add_frame({'mask3': pc4}, {'mask3': emb4})
#     print("After new mask, close pc:", tracker.objects_map.keys())

#     # Print all objects in Rtree
#     print("\nFinal objects in Rtree:")
#     for obj_id, obj in tracker.objects_map.items():
#         print(f"Object ID: {obj_id}, AABB: {obj.aabb}, Centroid: {obj.centroid}")

# if __name__ == "__main__":
#     # test_object_tracker3d_basic()
#     test_object_tracker3d_edge_cases()
