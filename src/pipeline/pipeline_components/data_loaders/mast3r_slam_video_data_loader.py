from typing import List,Any,Dict
from torch.utils.data import Dataset, DataLoader

import yaml
import torch

from .abstract_data_loader import AbstractDataLoader
from src.pipeline.data_entities.image_data_entity import ImageDataEntity

# Add the MASt3r-SLAM root directory to sys.path if not already present
import sys
import os
mast3r_slam_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mast3r_slam"))
if mast3r_slam_root not in sys.path:
    sys.path.insert(0, mast3r_slam_root)

from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.config import load_config, config, set_global_config






class MAST3RSLAMVideoDataSet(Dataset):
    """
    Dataset wrapper for the MasterSlam dataset.

    Args:
        video_path: Full path to the video.

    Returns:
        - 

    Raises:
        -
    """

    def __init__(self, video_path: str,calibration_config_path:str = None,device: str = "cuda:0") -> None:
        
        self.device = device
        self._dataset = load_dataset(video_path)
        self._dataset.subsample(config["dataset"]["subsample"])
        
        if calibration_config_path is not None:
            with open(calibration_config_path, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            self._dataset.use_calibration = True
            self._dataset.camera_intrinsics = Intrinsics.from_calib(
                self._dataset.img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )
        
        has_calib = self._dataset.has_calib()
        use_calib = config["use_calib"]

        if use_calib and not has_calib:
            print("[Warning] No calibration provided for this dataset!")
            sys.exit(0)

        if use_calib:
            self.K = torch.from_numpy(self._dataset.camera_intrinsics.K_frame).to(
                self.device,dtype=torch.float32
            )
        else:
            self.K = None
        
        self._dataset_iter = iter(self._dataset)
    



    def __len__(self) -> int:
        """
        Return the length of the dataset
        """

        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return the item from the internal dataset.

        Args:
            idx: Index of the item to return

        Returns:
            dict{
                "image": ImageDataEntity,
                "timestamp": Timestamp of the frame,
                "image_size": Size of the image as a single int,
                "image_height": Height of the images,
                "image_width": Width of the images
            }
        """
        
        timestamp,frame = self._dataset.__getitem__(idx)
        h,w = self._dataset.get_img_shape()[0]
        img_size = self._dataset.img_size

        #convert the information to the corresponding data entities
        image = ImageDataEntity(frame)

        return {
            "image": image,
            "timestamp": timestamp,
            "image_size": img_size,
            "image_height": h,
            "image_width": w,
            "calibration_K": self.K
            } 

class MAST3RSLAMVideoDataLoader(AbstractDataLoader):
    """
    Data loader component for loading video frames from a master slam dataset
    Not a real pytorch dataloader tho.
    
    Args:
        video_path: Full path to the video to load
        mast3r_slam_config_path: Full path to the config to use for MasterSlam incl. the dataset
        calibration_conig_path: Full path to the config for the calibration of the camera
        device: Device on which the data is loaded
    Returns:
        -
    Raises:
        NotImplementedError: When _run method is called
    """
    
    def __init__(self, 
                 video_path:str,
                 mast3r_slam_config_path:str,
                 calibration_config_path: str = None,
                 device: str = "cuda:0"
        ) -> None:

        super().__init__()
        self.video_path = video_path
        self.mast3r_slam_config_path = mast3r_slam_config_path
        self.calibration_config_path = calibration_config_path
        self.device = device

        
        #load the masterslam config -> sets parameters globally
        load_config(self.mast3r_slam_config_path)


        dataset = MAST3RSLAMVideoDataSet(
            self.video_path,
            self.calibration_config_path,
            device=self.device
        )
        
        



        self._dataloader = iter(dataset)
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component has no inputs as it's a data source."""
        return []
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs images."""
        return ["image","timestamp","image_size","image_width","image_height","calibration_K"]
    
    def _run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Not used for data loaders as they are meant to be used as iterators.
        
        Args:
            *args: Unused positional arguments
            **kwargs: Unused keyword arguments
        Raises:
            NotImplementedError: Always, as this method should not be used
        """
        raise NotImplementedError(
            "VideoDataLoader should not be called directly. Use it as an iterator instead."
        )
