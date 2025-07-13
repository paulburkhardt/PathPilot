import numpy as np
import torch

from .abstract_data_entity import AbstractDataEntity


class ImageSegmentationMaskDataEntity(AbstractDataEntity):
    """
    Data entity for a single image segmentation mask.

    Args:
        mask: Pytorch tensor or numpy array of shape (<H>, <W>), where each value is a class label.
    """

    def __init__(
        self,
        mask: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Initializes the ImageSegmentationMaskDataEntity.

        Args:
            mask: Pytorch tensor or numpy array of shape (<H>, <W>).
        Returns:
            -
        Raises:
            TypeError: If mask is not a numpy array or torch tensor.
            AssertionError: If mask does not have shape (H, W).
        """
        if isinstance(mask, torch.Tensor):
            self.mask_pytorch = mask
            self.mask_numpy = mask.cpu().numpy()
        elif isinstance(mask, np.ndarray):
            self.mask_pytorch = torch.from_numpy(mask)
            self.mask_numpy = mask
        else:
            raise TypeError(f"Invalid type {type(mask)}. Has to be Numpy array or Pytorch tensor.")

        if self.mask_pytorch.ndim != 2:
            raise AssertionError(
                f"mask must have shape (H, W), got {self.mask_pytorch.shape}"
            )

    def as_numpy(self):
        return self.mask_numpy

    def as_pytorch(self):
        return self.mask_pytorch
