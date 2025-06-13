# PathPilot Pipeline Architecture Reference

## Overview

PathPilot is a modular computer vision pipeline designed for SLAM (Simultaneous Localization and Mapping) and 3D scene reconstruction. The pipeline follows a component-based architecture where data flows through a sequence of specialized components, each performing specific tasks like data loading, SLAM processing, object extraction, visualization, and data writing.

## Architecture Overview

```
[Data Loader] → [SLAM Component] → [Object Extractor] → [Visualizer] → [Data Writer]
                         ↓
                [Pipeline Data Bucket]
                   (Shared State)
```

The pipeline consists of:

1. **Pipeline Core**: Orchestrates component execution and manages data flow
2. **Pipeline Components**: Modular processing units with specific responsibilities
3. **Data Entities**: Structured data representations for different types of information
4. **Configuration System**: YAML-based configuration for pipeline setup

## Core Classes

### 1. Pipeline (`src/pipeline/pipeline.py`)

The main orchestrator that executes components in sequence.

**Key Features:**
- Validates component dependencies before execution
- Manages iterative execution over data sequences
- Ensures data availability between components

**Key Methods:**
- `validate()`: Checks that all component inputs will be available
- `run()`: Executes the pipeline, iterating over the first component (data loader)

**Execution Flow:**
1. First component must be iterable (typically a data loader)
2. For each data item, creates a new `PipelineDataBucket`
3. Executes remaining components sequentially
4. Each component receives inputs from the bucket and adds outputs to it

### 2. PipelineBuilder (`src/pipeline/pipeline_builder.py`)

Factory class for constructing pipelines from YAML configuration.

**Key Features:**
- Maps component type strings to component classes
- Instantiates components with their configurations
- Returns configured `Pipeline` instance

**Component Registry:**
Currently supports:
- `MAST3RSLAMComponent`: SLAM processing
- `PointCloudDataWriter`: Point cloud output
- `MAST3RSLAMVideoDataLoader`: Video data input

### 3. PipelineDataBucket (`src/pipeline/pipeline_data_bucket.py`)

Shared data container for passing information between components.

**Available Data Entities:**
- `step_nr`: Current processing step number
- `image`, `image_height`, `image_width`, `image_size`: Image data and metadata
- `point_cloud`: 3D point cloud data
- `camera_pose`: Camera position and orientation
- `timestamp`: Temporal information
- `calibration_K`: Camera calibration matrix

**Key Methods:**
- `put(data)`: Store data (validates against allowed entities)
- `get(*keys)`: Retrieve specific data by keys

## Component Architecture

### Abstract Base Classes

#### AbstractPipelineComponent (`src/pipeline/pipeline_components/abstract_pipeline_component.py`)

Base class for all pipeline components.

**Required Properties:**
- `inputs_from_bucket`: List of required input data entities
- `outputs_to_bucket`: List of data entities this component produces

**Key Methods:**
- `__call__()`: Entry point that calls `_run()` and validates output
- `_run()`: Abstract method implemented by concrete components

#### Component Categories

1. **Data Loaders** (`data_loaders/`)
   - `AbstractDataLoader`: Base for data input components
   - Must be iterable (used as first component in pipeline)
   - Example: `MAST3RSLAMVideoDataLoader`

2. **SLAM Components** (`slam_components/`)
   - `AbstractSLAMComponent`: Base for SLAM processing
   - Example: `MAST3RSLAMComponent`

3. **Data Writers** (`data_writers/`)
   - `AbstractDataWriter`: Base for data output components
   - Examples: `PointCloudDataWriter`, `VizualizerDataWriter`

4. **Object Extractors** (`object_extractors/`)
   - `AbstractObjectExtractor`: Base for object detection/extraction
   - Examples: `PointCloudObjectExtractor`, `GaussianObjectExtractor`

5. **Data Visualizers** (`data_visualizer/`)
   - `AbstractDataVizualizer`: Base for data visualization
   - `AbstractRerunDataVizualizer`: Specialized for Rerun visualization
   - Examples: `PointCloudDataVizualizer`, `CameraDataVizualizer`

6. **Data Segmenters** (`data_segmenters/`)
   - Components for data segmentation tasks

7. **Object Databases** (`object_databases/`)
   - Components for object storage and retrieval

## Data Entities

Located in `src/pipeline/data_entities/`, these classes represent structured data:

### AbstractDataEntity (`abstract_data_entity.py`)
Base class for all data entities.

### ImageDataEntity (`image_data_entity.py`)
Represents image data with support for both NumPy arrays and PyTorch tensors.

**Features:**
- Automatic conversion between NumPy and PyTorch formats
- Shape validation (H, W, 3)
- Methods: `as_numpy()`, `as_pytorch()`

### PointCloudDataEntity (`point_cloud_data_entity.py`)
Represents 3D point cloud data with additional metadata.

## Configuration System

### Configuration Structure (`configs/default_config.yaml`)

```yaml
pipeline:
  components:
    - type: MAST3RSLAMVideoDataLoader
      config:
        video_path: path/to/video/data
        mast3r_slam_config_path: path/to/slam/config
    - type: MAST3RSLAMComponent
      config:
        point_cloud_method: accumulating
        c_confidence_threshold: 1.0
        mast3r_slam_config_path: path/to/slam/config
    - type: PointCloudDataWriter
      config:
        output_dir: outputs/
```

### Configuration Properties

- **type**: Component class name (must be in `PipelineBuilder.COMPONENT_MAP`)
- **config**: Component-specific configuration parameters

## Usage Patterns

### 1. Basic Pipeline Setup

```python
import hydra
from omegaconf import DictConfig
from src.pipeline.pipeline_builder import PipelineBuilder

@hydra.main(config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    pipeline = PipelineBuilder.build(cfg)
    pipeline.run()
```

### 2. Component Dependencies

Components declare their dependencies through properties:

```python
class MyComponent(AbstractPipelineComponent):
    @property
    def inputs_from_bucket(self) -> List[str]:
        return ["image", "camera_pose"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        return ["processed_data"]
```

### 3. Data Flow Validation

The pipeline automatically validates that:
- All required inputs are available before component execution
- Components are ordered correctly based on dependencies
- Data entity keys are valid (defined in `PipelineDataBucket`)

## Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Composability**: Components can be combined in different configurations
3. **Type Safety**: Strong typing and validation throughout the pipeline
4. **Configuration-Driven**: Pipeline structure defined through YAML files
5. **Iterative Processing**: Designed for processing sequences of data (video frames, image sets)
6. **Validation**: Extensive validation of component dependencies and data entities

## Entry Point

The main entry point is `main.py`, which uses Hydra for configuration management:

```python
@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(cfg: DictConfig) -> None:
    pipeline = PipelineBuilder.build(cfg)
    pipeline.run()
```

This allows for flexible configuration management and easy switching between different pipeline configurations.

## Extension Points

To add new components:

1. Create a new component class inheriting from the appropriate abstract base
2. Implement required properties (`inputs_from_bucket`, `outputs_to_bucket`)
3. Implement the `_run()` method
4. Add the component to `PipelineBuilder.COMPONENT_MAP`
5. Update `PipelineDataBucket.__available_data_entities` if new data types are needed

This architecture provides a robust foundation for computer vision pipelines with clear separation of concerns and easy extensibility. 