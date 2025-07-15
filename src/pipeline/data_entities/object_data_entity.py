import numpy as np
import torch
import torch.nn.functional as F
from .abstract_data_entity import AbstractDataEntity
 
class ObjectDataEntity(AbstractDataEntity):
    def __init__(self, 
                 obj_id,
                 mask_id,
                 points, 
                 centroid, 
                 description = None,
                 embedding=None,
                 running_embedding_weight = 0.8,
                 track_points =False
                 ):
        self.id = obj_id
        self.mask_id = {mask_id}
        self.centroid = centroid
        self.description = description
        self.aabb = self.get_aabb(points) # 6 Tupel - min_x,max_x,min_y,max_y,min_z,max_z

        self.cum_sum = np.sum(points,axis= 0).astype(float) 
        self.cum_len = points.shape[0]
        self.original_len = points.shape[0]

        self.track_points = track_points
        if self.track_points:
            self.points = points

        if embedding is not None: 
            self.embeddings = embedding
            self.running_embedding = F.normalize(embedding, p=2, dim=-1)
        else: 
            self.embeddings = None
            self.running_embedding = None
        self.running_embedding_weight = running_embedding_weight


    def update(self,mask_id,new_points, new_embedding, description):
        self.mask_id.add(mask_id)
        
        new_len = new_points.shape[0] 
        if new_len  > self.original_len:
            self.description = description
            self.original_len = new_len

        self.cum_sum += np.sum(new_points, axis= 0).astype(float) 
        self.cum_len += new_len

        if self.track_points:
            self.points = np.vstack([self.points,new_points])

        self.centroid = self.cum_sum / self.cum_len
        self.aabb = self.update_aabb(self.aabb,self.get_aabb(new_points))

        if self.embeddings is not None: 
            self.running_embedding =  self.running_embedding_weight * self.running_embedding + (1 - self.running_embedding_weight) * new_embedding
            self.running_embedding = F.normalize(self.running_embedding, p=2, dim=-1) 
            self.embeddings = torch.vstack([self.embeddings,new_embedding])

    def fuse(self, obj):
        self.mask_id.update(obj.mask_id)
        
        if obj.original_len > self.original_len:
            self.description = obj.description
            self.original_len = obj.original_len

        self.cum_sum += obj.cum_sum
        self.cum_len += obj.cum_len
        
        if self.track_points:
            self.points = np.vstack([self.points,obj.points])

        self.centroid = self.cum_sum / self.cum_len
        self.aabb = self.update_aabb(self.aabb, obj.aabb) 
        
        if self.embeddings is not None:
            self.running_embedding = self.running_embedding * self.embeddings.shape[0] + obj.embeddings.shape[0] * obj.running_embedding
            self.running_embedding = F.normalize(self.running_embedding, p=2, dim=-1) 
            self.embeddings = torch.vstack([self.embeddings,obj.embeddings])

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