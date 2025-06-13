# PathPilot Source Code

This directory contains the core source code for the PathPilot project - a computer vision and AI-powered navigation assistance system. The system is designed to process video data, extract 3D scene information, detect objects, and provide navigation guidance through warnings and audio feedback.

## Project Overview

PathPilot is a multi-stage project that processes camera feeds to provide real-time navigation assistance. The system uses advanced computer vision techniques including:

- 3D scene reconstruction (MaSt3R)
- Object detection and segmentation (SAM)
- Visual Language Model processing (VLM)
- Spatial database management
- Audio feedback generation

## Directory Structure

```
src/
├── pipeline/                   # Core pipeline architecture
│   ├── pipeline_components/    # Individual pipeline components
│   ├── data_entities/         # Data structure definitions
│   ├── pipeline.py            # Main pipeline class
│   ├── pipeline_builder.py    # Pipeline construction utilities
│   └── pipeline_data_bucket.py # Data flow management
├── data_extraction/           # Computer vision and data processing
│   ├── vlm_processor.py       # Visual Language Model processing
│   ├── video_to_frame.py      # Video frame extraction
│   ├── sam_processor.py       # Segment Anything Model processor
│   ├── mast3r_processor.py    # MaSt3R 3D reconstruction
│   ├── object_segmentation.py # Object segmentation utilities
│   └── object_compression.py  # Object data compression
├── data_retrieval/            # Object detection and user interaction
│   ├── detection_mechanism.py # Object detection logic
│   ├── warning_generator.py   # Warning system
│   └── text_to_speech.py      # Audio feedback generation
├── inputs/                    # Input data directory
├── outputs/                   # Output data directory
│   └── individual_masks/      # Segmentation mask outputs
├── pipelines.py              # Main pipeline orchestrator
├── data_base.py              # 3D spatial database management
├── utils.py                  # Utility functions
└── __init__.py               # Package initialization
```

## Core Components

### 1. Pipeline System (`pipeline/`)
The pipeline system provides a modular architecture for processing data through multiple stages. It includes:
- **Pipeline Builder**: Constructs processing pipelines
- **Data Bucket**: Manages data flow between components
- **Pipeline Components**: Modular processing units

### 2. Data Extraction (`data_extraction/`)
Handles computer vision and data processing tasks:
- **Video Processing**: Converts video streams to individual frames
- **3D Reconstruction**: Uses MaSt3R for camera pose estimation and 3D scene understanding
- **Object Segmentation**: Employs SAM (Segment Anything Model) for precise object boundaries
- **VLM Processing**: Integrates Visual Language Models for object understanding
- **Data Compression**: Optimizes object data for storage and retrieval

### 3. Data Retrieval (`data_retrieval/`)
Manages object detection and user interaction:
- **Detection Mechanism**: Identifies objects of interest in the scene
- **Warning Generator**: Creates appropriate warnings based on detected objects
- **Text-to-Speech**: Converts warnings to audio feedback for navigation assistance

### 4. Spatial Database (`data_base.py`)
Implements a 3D spatial database system:
- **Object3D Class**: Represents 3D objects with position, scale, and descriptions
- **Spatial Indexing**: Uses cKDTree for efficient spatial queries
- **Camera-relative Queries**: Finds objects within camera view and interaction radius
- **Position Description**: Generates relative position information

### 5. Main Pipeline Orchestrator (`pipelines.py`)
The central coordination system that:
- Manages different project stages (1, 2, 3)
- Orchestrates data extraction and retrieval pipelines
- Configures components based on project requirements
- Coordinates real-time processing workflow

## Project Stages

The system supports three development stages:

### Stage 1: Basic Infrastructure
- Video processing and frame extraction
- 3D camera pose estimation
- Basic spatial database

### Stage 2: Object Detection
- Adds object segmentation capabilities
- Implements object compression
- Enhanced spatial queries

### Stage 3: Advanced AI Integration
- Full VLM processing for object understanding
- Advanced warning generation
- Complete navigation assistance pipeline

## Key Features

- **Real-time Processing**: Efficient pipeline architecture for live video processing
- **3D Spatial Awareness**: Uses camera pose estimation and 3D reconstruction
- **Modular Design**: Component-based architecture for easy extension and testing
- **Multi-stage Development**: Progressive feature implementation across project phases
- **Audio Feedback**: Text-to-speech integration for accessibility
- **Efficient Storage**: Spatial indexing and data compression for performance

## Usage

The main entry point is through the `Pipelines` class in `pipelines.py`:

```python
from pipelines import Pipelines

# Initialize with configuration
config = {
    'project_stage': 3,
    'video_processor': {...},
    'mast3r': {...},
    'detection_mechanism': {...},
    # ... other component configs
}

pipeline = Pipelines(config)

# Run extraction pipeline
pipeline.pipeline_extraction()

# Run retrieval pipeline
pipeline.pipeline_retrieval()
```

## Dependencies

The system relies on several key libraries:
- **NumPy**: Numerical computations and array operations
- **SciPy**: Spatial data structures (cKDTree)
- **LieTorch**: SE3 transformations for camera poses
- **Computer Vision Models**: SAM, MaSt3R, VLM integrations

## Data Flow

1. **Input**: Video stream from camera
2. **Frame Extraction**: Convert video to individual frames
3. **3D Processing**: Estimate camera pose and reconstruct scene
4. **Object Detection**: Identify and segment objects of interest
5. **Spatial Query**: Find relevant objects based on camera position
6. **Warning Generation**: Create appropriate navigation warnings
7. **Audio Output**: Convert warnings to speech for user feedback

This architecture enables real-time navigation assistance through advanced computer vision and AI processing. 