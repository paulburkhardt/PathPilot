# PathPilot

PathPilot is a real-time obstacle detection and warning system that uses computer vision and SLAM to help users navigate through spaces safely.

## Features

- Video input processing
- 3D point cloud generation using Mast3r_slam
- Object segmentation using SAM-B
- Distance calculation to nearest objects
- Real-time warning system
- Visual Language Model integration using BLIP-2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pathpilot.git
cd pathpilot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py --video_path path/to/your/video.mp4
```

2. The system will process the video and provide real-time warnings about nearby objects.

## Configuration

Edit `config/config.yaml` to modify:
- Warning distance thresholds
- Model parameters
- Processing settings

## Project Structure

```
pathpilot/
├── src/                    # Source code
├── config/                 # Configuration files
├── requirements.txt        # Project dependencies
└── main.py                # Main entry point
```

## Dependencies

- PyTorch
- OpenCV
- Transformers (Hugging Face)
- Segment Anything Model (SAM-B)
- Mast3r_slam
- BLIP-2

## License

MIT License 





# Sam 2
## Installation:
1. Clone the official sam2 repository:
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    ```
2. Install sam2 in editable mode (No need to reinstall PyTorch if you have it installed from your masterslam setup):
    ```bash
    cd sam2
    pip install -e .
    ```
    Follow the instructions for downloading sam2 checkpoints from the [sam2 GitHub setup](https://github.com/facebookresearch/sam2#getting-started).
3. Rename the top level directory of the sam2 project (Otherwise the repo is not executable outside of their directory due to a sam2/sam2/ within the path.):
    ```bash
    cd ..
    mv sam2 segment_anything_2
    ```
    The structure should now be `PathPilot/segment_anything_2/<sam2 files>`.
4. Copy the custom `build_sam.py` into the sam2 module:
    ```bash
    cp src/sam2_additional_components/build_sam.py PathPilot/segment_anything_2/sam2/
    ```
    This file should already exist in their implementation, so you need to replace it. 
5. Copy the `sam2_camera_predictor.py` file:
    ```bash
    cp src/sam2_additional_components/sam2_camera_predictor.py PathPilot/segment_anything_2/sam2/
    ```
6. Remove or comment out all code in `PathPilot/segment_anything_2/sam2/__init__.py` to avoid hydra issues.