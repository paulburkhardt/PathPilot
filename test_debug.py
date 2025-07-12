#!/usr/bin/env python3

import numpy as np
import torch
from src.pipeline.pipeline_components.object_databases.bbox_object_database import BBoxObjectDatabase

def test_match_object():
    # Create database with BLIP enabled
    db = BBoxObjectDatabase(use_blip=True, match_threshold=1.0, knn_count=10)
    
    # Create some test objects with embeddings
    test_objects = {
        'mask1': np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]]),
        'mask2': np.array([[2, 2, 2], [3, 3, 3], [2, 3, 2]]),
        'mask3': np.array([[4, 4, 4], [5, 5, 5], [4, 5, 4]])
    }
    
    test_embeddings = {
        'mask1': torch.tensor([1.0, 0.0, 0.0]),
        'mask2': torch.tensor([0.0, 1.0, 0.0]),
        'mask3': torch.tensor([0.0, 0.0, 1.0])
    }
    
    # Add objects to database
    print("Adding objects to database...")
    db.add_frame(test_objects, test_embeddings)
    
    print(f"Database now has {len(db.objects_map)} objects")
    print(f"Object IDs: {list(db.objects_map.keys())}")
    
    # Try to match a new object
    print("\nTrying to match a new object...")
    new_points = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    new_embedding = torch.tensor([0.8, 0.2, 0.0])
    centroid = new_points.mean(axis=0)
    aabb = db._get_aabb(new_points)
    
    # This should trigger the print statement in _match_object
    result = db._match_object(centroid, aabb, new_embedding)
    print(f"Match result: {result}")

if __name__ == "__main__":
    test_match_object() 