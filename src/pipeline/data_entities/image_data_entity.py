import numpy as np
import torch

from .abstract_data_entity import AbstractDataEntity


class ImageDataEntity(AbstractDataEntity):
    """
    Data entity for a single image.

    Args:
        image: Pytorch tensor or numpy array of shape (<H>, <W>, 3).
    """

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Initializes the ImageDataEntity.

        Args:
            image: Pytorch tensor or numpy array of shape (<H>, <W>, 3).
        Returns:
            -
        Raises:
            TypeError: If image is not a numpy array or torch tensor.
            AssertionError: If image does not have shape (H, W, 3).
        """
        if isinstance(image, torch.Tensor):
            self.image_pytorch = image
            self.image_numpy = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            self.image_pytorch = torch.from_numpy(image)
            self.image_numpy = image
        else:
            raise TypeError(f"Invalid type {type(image)}. Has to be Numpy array or Pytorch tensor.")

        if self.image_pytorch.ndim != 3 or self.image_pytorch.shape[2] != 3:
            raise AssertionError(
                f"image must have shape (H, W, 3), got {self.image_pytorch.shape}"
            )

    def as_numpy(self):
        return self.image_numpy

    def as_pytorch(self):
        return self.image_pytorch
