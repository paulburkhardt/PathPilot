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