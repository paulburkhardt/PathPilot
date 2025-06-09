from typing import List,Any,Dict

from .abstract_data_loader import AbstractDataLoader

from torch.utils.data import Dataset, DataLoader

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
    Dataset wrapper for the MasterSlam dataset
    """

    def __init__(self, video_path: str) -> None:
        
        self._dataset = load_dataset(video_path)
        self._dataset_iter = iter(self._dataset)
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        timestamp,frame = self._dataset.__getitem__(idx)
        h,w = self._dataset.get_img_shape()[0]

        return {
            "rgb_image": frame,
            "timestamp": timestamp,
            "img_size": self._dataset.img_size,
            "img_height": h,
            "img_width": w
            } 
    
    def __del__(self) -> None:
        pass

class MAST3RSLAMVideoDataLoader(AbstractDataLoader):
    """
    Data loader component for loading video frames from a master slam dataset

    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: When _run method is called
    """
    
    def __init__(self, 
                 video_path:str,
                 mast3r_slam_config_path:str
        ) -> None:
        super().__init__()
        self.video_path = video_path
        self.mast3r_slam_config_path = mast3r_slam_config_path
        load_config(self.mast3r_slam_config_path)

        self._dataloader = None
        self._initialize_dataloader()
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component has no inputs as it's a data source."""
        return []
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs RGB images."""
        return ["rgb_image","timestamp","img_size","img_width","img_height"]
    
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
    
    def _initialize_dataloader(self) -> None:
        """
        Initialize the video data loader with configuration parameters.
        
        Args:
            -
        Returns:
            -
        """
        if not self._dataloader:
            if not self.video_path:
                raise ValueError("video_path must be specified in config")
            
            dataset = MAST3RSLAMVideoDataSet(self.video_path)
            self._dataloader = iter(dataset)
            
            #DataLoader(
            #    dataset,
            #    batch_size=1,
            #    shuffle=False,
            #    num_workers=0  # No multiprocessing for video loading
            #)
