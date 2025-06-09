# Pipeline Module

A flexible and modular pipeline framework for processing video data through various computer vision and SLAM (Simultaneous Localization and Mapping) components.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Extending the Pipeline](#extending-the-pipeline)
- [Data Flow](#data-flow)
- [Error Handling](#error-handling)

## 🎯 Overview

The Pipeline module provides a robust framework for chaining together various data processing components in a sequential manner. It's designed specifically for computer vision workflows involving video processing, SLAM, object detection, and data output operations.

### Key Features

- **Modular Design**: Components can be easily added, removed, or reordered
- **Type Safety**: Built-in validation ensures data compatibility between components
- **Flexible Configuration**: JSON-based configuration for easy pipeline setup
- **Data Validation**: Automatic validation of input/output requirements
- **Iterator Support**: First component acts as data iterator for batch processing

## 🏗️ Architecture

The pipeline follows a producer-consumer pattern where:

1. **Data Loader** (Iterator): Provides input data (e.g., video frames)
2. **Processing Components**: Transform data (SLAM, segmentation, object extraction)
3. **Data Writers**: Output results (point clouds, files)

```
[Data Loader] → [Component 1] → [Component 2] → ... → [Data Writer]
       ↓               ↓              ↓                      ↓
   [Data Bucket] → [Data Bucket] → [Data Bucket] → [Data Bucket]
```

## 🚀 Quick Start

### Basic Usage

```python
from pipeline import Pipeline
from pipeline.pipeline_builder import PipelineBuilder

# Method 1: Using PipelineBuilder with configuration
config = {
    "pipeline": {
        "components": [
            {
                "type": "VideoDataLoader",
                "config": {"video_path": "path/to/video.mp4"}
            },
            {
                "type": "MAST3RSLAMComponent",
                "config": {}
            },
            {
                "type": "PointCloudDataWriter", 
                "config": {"output_dir": "results/"}
            }
        ]
    }
}

pipeline = PipelineBuilder.build(config)
pipeline.run()
```

### Manual Pipeline Creation

```python
from pipeline import Pipeline
from pipeline.pipeline_components.data_loaders.video_data_loader import VideoDataLoader
from pipeline.pipeline_components.slam_components.mast3r_slam_component import MAST3RSLAMComponent
from pipeline.pipeline_components.data_writers.point_cloud_data_writer import PointCloudDataWriter

# Create components
video_loader = VideoDataLoader(video_path="path/to/video.mp4")
slam_component = MAST3RSLAMComponent()
point_cloud_writer = PointCloudDataWriter(output_dir="results/")

# Build pipeline
pipeline = Pipeline([video_loader, slam_component, point_cloud_writer])
pipeline.run()
```

## 🔧 Core Components

### Pipeline (`pipeline.py`)

The main orchestrator that manages component execution and data flow.

**Key Methods:**
- `__init__(components)`: Initialize with list of components
- `validate()`: Validate input/output compatibility
- `run()`: Execute all components sequentially

### PipelineDataBucket (`pipeline_data_bucket.py`)

Thread-safe data container for passing information between components.

**Available Data Entities:**
- `step_nr`: Current processing step number
- `rgb_image`: RGB image data
- `depth_image`: Depth map data
- `point_cloud`: 3D point cloud data
- `segmentation_mask`: Segmentation masks
- `object_data`: Extracted object information
- `slam_data`: SLAM processing results
- `camera_pose`: Camera pose estimation

**Key Methods:**
- `put(data)`: Store data dictionary
- `get(*keys)`: Retrieve specific data entries

### PipelineBuilder (`pipeline_builder.py`)

Factory class for building pipelines from configuration files.

**Supported Component Types:**
- `VideoDataLoader`: Load video frames
- `MAST3RSLAMComponent`: SLAM processing
- `PointCloudDataWriter`: Output point clouds

### AbstractPipelineComponent (`abstract_pipeline_component.py`)

Base class for all pipeline components defining the interface contract.

**Required Properties:**
- `inputs_from_bucket`: List of required input data types
- `outputs_to_bucket`: List of output data types

**Required Methods:**
- `_run(*args, **kwargs)`: Main processing logic

## 🧩 Pipeline Components

### Data Loaders (`pipeline_components/data_loaders/`)

**Purpose**: Provide input data to the pipeline
- `VideoDataLoader`: Loads frames from video files
- `AbstractDataLoader`: Base class for data loaders

### SLAM Components (`pipeline_components/slam_components/`)

**Purpose**: Perform Simultaneous Localization and Mapping
- `MAST3RSLAMComponent`: Implementation using MAST3R algorithm
- `AbstractSLAMComponent`: Base class for SLAM components

### Data Writers (`pipeline_components/data_writers/`)

**Purpose**: Output processed results
- `PointCloudDataWriter`: Saves 3D point clouds
- `AbstractDataWriter`: Base class for data writers

### Additional Component Categories

- **Object Extractors** (`object_extractors/`): Extract objects from scenes
- **Object Databases** (`object_databases/`): Manage object information
- **Data Segmenters** (`data_segmenters/`): Segment images/point clouds

## ⚙️ Configuration

### Configuration Format

```json
{
  "pipeline": {
    "components": [
      {
        "type": "ComponentType",
        "config": {
          "parameter1": "value1",
          "parameter2": "value2"
        }
      }
    ]
  }
}
```

### Example Configuration

```json
{
  "pipeline": {
    "components": [
      {
        "type": "VideoDataLoader",
        "config": {
          "video_path": "/path/to/input/video.mp4",
          "frame_skip": 1,
          "max_frames": 1000
        }
      },
      {
        "type": "MAST3RSLAMComponent", 
        "config": {
          "model_path": "/path/to/mast3r/model",
          "processing_resolution": [512, 384]
        }
      },
      {
        "type": "PointCloudDataWriter",
        "config": {
          "output_dir": "/path/to/output/",
          "file_format": "ply",
          "coordinate_system": "world"
        }
      }
    ]
  }
}
```

## 💡 Usage Examples

### Video Processing Pipeline

```python
# Configuration for video → SLAM → point cloud pipeline
config = {
    "pipeline": {
        "components": [
            {
                "type": "VideoDataLoader",
                "config": {"video_path": "input.mp4"}
            },
            {
                "type": "MAST3RSLAMComponent",
                "config": {}
            },
            {
                "type": "PointCloudDataWriter",
                "config": {"output_dir": "output/"}
            }
        ]
    }
}

pipeline = PipelineBuilder.build(config)
pipeline.run()
```

### Error Handling

```python
try:
    pipeline = PipelineBuilder.build(config)
    pipeline.run()
except ValueError as e:
    print(f"Pipeline validation failed: {e}")
except KeyError as e:
    print(f"Data access error: {e}")
```

## 🔨 Extending the Pipeline

### Creating a Custom Component

```python
from pipeline.pipeline_components.abstract_pipeline_component import AbstractPipelineComponent
from typing import List, Dict, Any

class CustomProcessor(AbstractPipelineComponent):
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        return ["rgb_image"]  # Required inputs
    
    @property 
    def outputs_to_bucket(self) -> List[str]:
        return ["processed_image"]  # Generated outputs
    
    def _run(self, rgb_image: Any) -> Dict[str, Any]:
        # Your processing logic here
        processed_image = self.process_image(rgb_image)
        
        return {
            "processed_image": processed_image
        }
    
    def process_image(self, image):
        # Custom processing implementation
        return image
```

### Registering New Component

Add your component to `PipelineBuilder.COMPONENT_MAP`:

```python
# In pipeline_builder.py
COMPONENT_MAP = {
    "VideoDataLoader": VideoDataLoader,
    "MAST3RSLAMComponent": MAST3RSLAMComponent,
    "PointCloudDataWriter": PointCloudDataWriter,
    "CustomProcessor": CustomProcessor,  # Add your component
}
```

## 📊 Data Flow

```
Step 1: VideoDataLoader
├── Input: video file
└── Output: rgb_image, depth_image → Data Bucket

Step 2: MAST3RSLAMComponent  
├── Input: rgb_image, depth_image ← Data Bucket
└── Output: point_cloud, camera_pose → Data Bucket

Step 3: PointCloudDataWriter
├── Input: point_cloud ← Data Bucket
└── Output: saved files
```

## ⚠️ Error Handling

### Common Errors

1. **Validation Error**: Component requires unavailable inputs
   ```
   ValueError: Component 1 (ComponentName) requires inputs {'missing_input'} 
   but they won't be available.
   ```

2. **Data Entity Error**: Invalid key in data bucket
   ```
   KeyError: The following keys are not valid data entities: {'invalid_key'}
   ```

3. **Configuration Error**: Missing or invalid component configuration
   ```
   ValueError: Unknown component type: InvalidComponent
   ```

### Best Practices

- Always validate pipeline before running
- Check component input/output compatibility
- Use descriptive error messages in custom components
- Test components individually before pipeline integration

## 🔍 Troubleshooting

### Pipeline Validation Issues
- Ensure each component's required inputs are provided by previous components
- Check that the first component is an iterator (data loader)
- Verify component types are registered in `COMPONENT_MAP`

### Data Bucket Issues  
- Only use predefined data entity keys
- Ensure data types match component expectations
- Check for missing required inputs

### Configuration Issues
- Validate JSON syntax
- Ensure all required configuration parameters are provided
- Check file paths and permissions

---

For more detailed information about specific components, refer to their individual documentation within each component directory. 