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
├── config/                 # Configuration files
├── data/                   # Data files for the project
├── models/                 # Foundation models 
├── plots/                  # Plots and visualizations stored here
│   ├── __init__.py         # Package initialization
│   ├── model_visualization.py    # Visualization of the model
│   └── plots_files/              # Plots files stored here
├── references/             # explanatory materials to understand the project
├── results/                # Results stored here
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── data_base.py        # Data base for storing and retrieving the objects
│   ├── pipelines.py         # All Pipelines of the incremental stages
│   ├── utils.py                    # Utility functions
│   ├── data_extraction/            # Data extraction
│   │   ├── __init__.py                 # Package initialization
│   │   ├── video_to_frames.py          # Video processing
│   │   ├── mast3r_processor.py         # Mast3r_slam process
│   │   ├── sam_processor.py            # SAM processing
│   │   ├── vlm_processor.py            # VLM processing
│   │   ├── object_segmentation.py      # 2D to 3D Projection of the objects
│   │   └── object_compression.py       # Object compression
│   │
│   ├── data_retrieval/             # Data retrieval
│   │   ├── __init__.py                 # Package initialization
│   │   ├── detection_mechanism.py      # Detection of the object based on user pose
│   │   ├── warning_generator.py        # Warning generator  
│   │   └── text_to_speech.py           # Text to speech
│   │
│   
├── tests/                  # Test files and notebooks for checking functionality
├── evaluation/             # Evaluate the pipelines
├── README.md               # Project overview
├── requirements.txt        # Project dependencies
└── main.py                 # Main entry point
└── .gitignore              # Git ignore file
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