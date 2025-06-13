import numpy as np
import torch

from .abstract_data_entity import AbstractDataEntity


class PointCloudDataEntity(AbstractDataEntity):
    """
    Data entity for a pointcloud.
    A pointcloud consists of the following structure:
    (<number of points>, <x,y,z, (r,g,b), (Confidence score)>), where the inner () means optional
    

    Args:
        point_cloud: Pytorch tensor or numpy array of shape (<Number of points>,3).
        rgb: Pytorch tensor or numpy array of shape (<Number of points>, 3).
        confidence_scores: Pytorch tensor or numpy array of shape (<Number of points>, 1) 
    """

    def __init__(
        self,
        point_cloud: np.ndarray | torch.Tensor, 
        rgb: np.ndarray | torch.Tensor = None,
        confidence_scores: np.ndarray | torch.Tensor = None
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
            self.point_cloud_numpy = point_cloud.cpu().numpy()
        elif isinstance(point_cloud, np.ndarray):
            self.point_cloud_pytorch = torch.from_numpy(point_cloud)
            self.point_cloud_numpy = point_cloud
        else:
            raise TypeError(f"Invalid type {type(point_cloud)}. Has to be Numpy array or Pytorch tensor.")

        if rgb is not None:
            if isinstance(rgb, torch.Tensor):
                self.rgb_pytorch = rgb
                self.rgb_numpy = rgb.cpu().numpy()
            elif isinstance(rgb, np.ndarray):
                self.rgb_pytorch = torch.from_numpy(rgb)
                self.rgb_numpy = rgb
            else:
                raise TypeError(f"Invalid type {type(rgb)} for rgb. Has to be Numpy array or Pytorch tensor.")
        else:
            self.rgb_pytorch = None
            self.rgb_numpy = None

        if confidence_scores is not None:
            if isinstance(confidence_scores, torch.Tensor):
                self.confidence_scores_pytorch = confidence_scores
                self.confidence_scores_numpy = confidence_scores.cpu().numpy()
            elif isinstance(confidence_scores, np.ndarray):
                self.confidence_scores_pytorch = torch.from_numpy(confidence_scores)
                self.confidence_scores_numpy = confidence_scores
            else:
                raise TypeError(f"Invalid type {type(confidence_scores)} for confidence_scores. Has to be Numpy array or Pytorch tensor.")
        else:
            self.confidence_scores_pytorch = None
            self.confidence_scores_numpy = None

        if self.point_cloud_pytorch.ndim != 2 or self.point_cloud_pytorch.shape[1] != 3:
            raise AssertionError(
                f"point_cloud must have shape (num_points, 3), got {self.point_cloud_pytorch.shape}"
            )

    def as_numpy(self,with_rgb:bool=False,with_confidence_score:bool=False):
        """
        Returns the point cloud data as a NumPy array.

        Args:
            with_rgb (bool, optional): If True, includes RGB color information in the returned array.
                Raises a TypeError if RGB data is not available. Default is False.
            with_confidence_score (bool, optional): If True, includes confidence scores in the returned array.
                Raises a TypeError if confidence score data is not available. Default is False.
        
        Returns:
            np.ndarray: A NumPy array containing the point cloud data. The array will have additional
                columns for RGB and/or confidence scores if requested and available.
        
        Raises:
            TypeError: If `with_rgb` is True but RGB data is not present.
            TypeError: If `with_confidence_score` is True but confidence score data is not present.
        """
        
        components = [self.point_cloud_numpy]
        if with_rgb: 
            if self.rgb_numpy is not None:
                components.append(self.rgb_numpy)
            else:
                raise TypeError("There is no rgb info in this pointcloud.")
        if with_confidence_score:
            if self.confidence_scores_numpy is not None:
                components.append(self.confidence_scores_numpy)
            else:
                raise TypeError("There is no confidence score in this pointcloud.")
        
        if len(components)>1:
            return np.concatenate(components, axis=-1)
        else:
            return self.point_cloud_numpy

    def as_pytorch(self, with_rgb:bool=False, with_confidence_score:bool=False):
        """
        Converts the point cloud data to a PyTorch tensor.
        This method returns the point cloud as a PyTorch tensor, optionally including RGB information and/or confidence scores as additional channels.
        
        Args:
            with_rgb (bool, optional): If True, includes RGB information in the output tensor. Raises a TypeError if RGB data is not available. Default is False.
            with_confidence_score (bool, optional): If True, includes confidence scores in the output tensor. Raises a TypeError if confidence scores are not available. Default is False.
        
        Returns:
            torch.Tensor: A tensor containing the point cloud data, and optionally RGB and confidence score information concatenated along the last dimension.
        
        Raises:
            TypeError: If `with_rgb` is True but RGB data is not present.
            TypeError: If `with_confidence_score` is True but confidence scores are not present.
        """

        components = [self.point_cloud_pytorch]
        if with_rgb:
            if self.rgb_pytorch is not None:
                components.append(self.rgb_pytorch)
            else:
                raise TypeError("There is no rgb info in this pointcloud.")
        if with_confidence_score:
            if self.confidence_scores_pytorch is not None:
                components.append(self.confidence_scores_pytorch)
            else:
                raise TypeError("There is no confidence score in this pointcloud.")

        if len(components) > 1:
            return torch.cat(components, dim=-1)
        else:
            return self.point_cloud_pytorch    

