import numpy as np
import torch

from .abstract_data_entity import AbstractDataEntity


class PointCloudDataEntity(AbstractDataEntity):
    """
    Data entity for a pointcloud
    
    Args:
        point_cloud: Pytorch tensor or numpy array of shape (<batch_size>,<Number of points>,3).
    """

    def __init__(
        self,
        point_cloud: np.ndarray | torch.Tensor, 
    ) -> None:
        """
        Initializes the PointCloudDataEntity.

        Args:
            point_cloud: Pytorch tensor or numpy array of shape (<batch_size>, <Number of points>, 3).
        Returns:
            -
        Raises:
            TypeError: If point_cloud is not a numpy array or torch tensor.
            AssertionError: If point_cloud does not have shape (batch_size, num_points, 3).
        """
        if isinstance(point_cloud, torch.Tensor):
            self.point_cloud_pytorch = point_cloud
            self.point_cloud_numpy = point_cloud.numpy()
        elif isinstance(point_cloud, np.ndarray):
            self.point_cloud_pytorch = torch.from_numpy(point_cloud)
            self.point_cloud_numpy = point_cloud
        else:
            raise TypeError(f"Invalid type {type(point_cloud)}. Has to be Numpy array or Pytorch tensor.")

        if self.point_cloud_pytorch.ndim != 3 or self.point_cloud_pytorch.shape[2] != 3:
            raise AssertionError(
                f"point_cloud must have shape (batch_size, num_points, 3), got {self.point_cloud_pytorch.shape}"
            )

    def as_numpy(self):
        return self.point_cloud_numpy

    def as_pytorch(self):
        return self.point_cloud_pytorch    

