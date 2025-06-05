from typing import List, Dict, Any
import cv2
from torch.utils.data import Dataset, DataLoader
from .abstract_data_loader import AbstractDataLoader

class VideoDataset(Dataset):
    """
    Dataset class for loading video frames.
    
    Args:
        video_path: Path to the video file to load.
    
    Returns:
        -
    
    Raises:
        RuntimeError: If video file cannot be opened
    """
    
    def __init__(self, video_path: str) -> None:
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __len__(self) -> int:
        return self.num_frames
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {"rgb_image": frame}
    
    def __del__(self) -> None:
        if hasattr(self, 'cap'):
            self.cap.release()

class VideoDataLoader(AbstractDataLoader):
    """
    Data loader component for loading video frames.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: When _run method is called
    """
    
    def __init__(self, video_path:str) -> None:
        super().__init__()
        self.video_path = video_path
        self._dataloader = None
        self._initialize_dataloader()
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component has no inputs as it's a data source."""
        return []
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs RGB images."""
        return ["rgb_image"]
    
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
            
            dataset = VideoDataset(self.video_path)
            self._dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0  # No multiprocessing for video loading
            )
