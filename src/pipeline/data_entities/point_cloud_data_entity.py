import numpy as np
import torch

from .abstract_data_entity import AbstractDataEntity


class PointCloudDataEntity(AbstractDataEntity):
    """
    Data entity for a pointcloud.
    A pointcloud consists of the following structure:
    (<number of points>, <x,y,z, (r,g,b), (Confidence score), (Segmentation mask)>), where the inner () means optional

    Args:
        point_cloud: Pytorch tensor or numpy array of shape (<Number of points>,3).
        rgb: Pytorch tensor or numpy array of shape (<Number of points>, 3).
        confidence_scores: Pytorch tensor or numpy array of shape (<Number of points>, 1) 
        segmentation_mask: Pytorch tensor or numpy array of shape (<Number of points>, 1), dtype uint
    """

    def __init__(
        self,
        point_cloud: np.ndarray | torch.Tensor, 
        rgb: np.ndarray | torch.Tensor = None,
        confidence_scores: np.ndarray | torch.Tensor = None,
        segmentation_mask: np.ndarray | torch.Tensor = None
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

        if segmentation_mask is not None:
            if isinstance(segmentation_mask, torch.Tensor):
                self.segmentation_mask_pytorch = segmentation_mask
                self.segmentation_mask_numpy = segmentation_mask.cpu().numpy()
            elif isinstance(segmentation_mask, np.ndarray):
                self.segmentation_mask_pytorch = torch.from_numpy(segmentation_mask)
                self.segmentation_mask_numpy = segmentation_mask
            else:
                raise TypeError(f"Invalid type {type(segmentation_mask)} for segmentation_mask. Has to be Numpy array or Pytorch tensor.")
            # Check dtype
            if self.segmentation_mask_numpy.dtype.kind not in {'u', 'i'}:
                raise TypeError("segmentation_mask must be of unsigned/signed integer dtype.")
        else:
            self.segmentation_mask_pytorch = None
            self.segmentation_mask_numpy = None

        if self.point_cloud_pytorch.ndim != 2 or self.point_cloud_pytorch.shape[1] != 3:
            raise AssertionError(
                f"point_cloud must have shape (num_points, 3), got {self.point_cloud_pytorch.shape}"
            )

    def as_numpy(self, with_rgb: bool = False, with_confidence_score: bool = False, with_segmentation_mask: bool = False):
        """
        Returns the point cloud data as a NumPy array.

        Args:
            with_rgb (bool, optional): If True, includes RGB color information in the returned array.
            with_confidence_score (bool, optional): If True, includes confidence scores in the returned array.
            with_segmentation_mask (bool, optional): If True, includes segmentation mask in the returned array.
        Returns:
            np.ndarray: A NumPy array containing the point cloud data.
        Raises:
            TypeError: If requested data is not present.
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
        if with_segmentation_mask:
            if self.segmentation_mask_numpy is not None:
                components.append(self.segmentation_mask_numpy)
            else:
                raise TypeError("There is no segmentation mask in this pointcloud.")

        if len(components) > 1:
            components = [c.reshape(-1, 1) if c.ndim == 1 else c for c in components]
            return np.concatenate(components, axis=-1)
        else:
            return self.point_cloud_numpy

    def as_pytorch(self, with_rgb: bool = False, with_confidence_score: bool = False, with_segmentation_mask: bool = False):
        """
        Converts the point cloud data to a PyTorch tensor.
        This method returns the point cloud as a PyTorch tensor, optionally including RGB information,
        confidence scores, and/or segmentation mask as additional channels.
        Args:
            with_rgb (bool, optional): If True, includes RGB information in the output tensor.
            with_confidence_score (bool, optional): If True, includes confidence scores in the output tensor.
            with_segmentation_mask (bool, optional): If True, includes segmentation mask in the output tensor.
        Returns:
            torch.Tensor: A tensor containing the point cloud data.
        Raises:
            TypeError: If requested data is not present.
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
        if with_segmentation_mask:
            if self.segmentation_mask_pytorch is not None:
                components.append(self.segmentation_mask_pytorch)
            else:
                raise TypeError("There is no segmentation mask in this pointcloud.")

        if len(components) > 1:
            return torch.cat(components, dim=-1)
        else:
            return self.point_cloud_pytorch
