# PathPilot Setup Guide

## Project Overview
PathPilot is a real-time obstacle detection system that combines computer vision, SLAM, and AI to help users navigate safely through spaces. The system processes video input to detect and warn about nearby obstacles.

## Current Implementation Status

### ✅ Completed Components
1. **Project Structure**
   - Basic project layout with modular components
   - Configuration system
   - Dependency management

2. **Core Modules**
   - Video processing pipeline
   - Segmentation using SAM-B
   - Distance calculation logic
   - Warning system
   - BLIP-2 integration for object description

### 🚧 Pending Implementation
1. **Mast3r_slam Integration**
   - The `slam_processor.py` currently contains placeholder code
   - Need to implement actual SLAM integration
   - Replace dummy point cloud generation with real SLAM output

## Next Steps

### 1. Environment Setup with Miniconda

1. **Install Miniconda**
   ```bash
   # Download Miniconda installer
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   
   # Install Miniconda
   bash Miniconda3-latest-Linux-x86_64.sh
   
   # Initialize conda (after installation)
   conda init bash
   ```

2. **Create and Activate Conda Environment**
   ```bash
   # Create new environment with Python 3.11
   conda create -n pathpilot python=3.11
   
   # Activate environment
   conda activate pathpilot
   
   # Install CUDA toolkit and PyTorch
   conda install -c nvidia cuda-toolkit
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   
   # Set CUDA environment variables
   export CUDA_HOME=$CONDA_PREFIX
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   # Check CUDA availability
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   
   # Check PyTorch version
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   
   # Check CUDA version
   nvcc --version
   ```

### 2. Model Downloads
1. **SAM-B Model**
   - Download the SAM-B model weights
   - Place in appropriate directory (typically `models/`)
   - Update model path in configuration if needed

2. **BLIP-2 Model**
   - The model will be downloaded automatically on first run
   - Ensure sufficient disk space (~10GB)
   - GPU recommended for optimal performance

### 3. Mast3r_slam Integration
1. Replace the placeholder code in `src/slam_processor.py`:
   ```python
   # TODO: Replace with actual implementation
   self.slam = Mast3r_slam(...)
   ```
2. Implement proper point cloud generation
3. Add error handling for SLAM failures

### 4. Testing
1. **Unit Tests**
   - Create test cases for each component
   - Test edge cases and error conditions
   - Verify warning system thresholds

2. **Integration Tests**
   - Test full pipeline with sample videos
   - Verify real-time performance
   - Check warning accuracy

### 5. Performance Optimization
1. **GPU Utilization**
   - Ensure proper CUDA setup
   - Optimize batch processing
   - Consider model quantization

2. **Real-time Processing**
   - Profile each component
   - Optimize bottlenecks
   - Consider multi-threading if needed

## Configuration

The system is configured through `config/config.yaml`. Key parameters to adjust:

1. **Video Processing**
   - Resolution and FPS
   - Processing pipeline settings

2. **SLAM Settings**
   - Point cloud parameters
   - Confidence thresholds

3. **Warning System**
   - Distance thresholds
   - Warning intervals
   - Message templates

## Usage Example

```bash
# Activate environment (if not already activated)
conda activate pathpilot

# Basic usage
python main.py --video_path path/to/video.mp4

# With custom config
python main.py --video_path path/to/video.mp4 --config custom_config.yaml
```

## Troubleshooting

### Common Issues
1. **CUDA Installation Issues**
   - If CUDA_HOME error occurs:
     ```bash
     export CUDA_HOME=$CONDA_PREFIX
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
     ```
   - If CUDA not found:
     ```bash
     conda install -c nvidia cuda-toolkit
     ```

2. **CUDA Out of Memory**
   - Reduce batch sizes
   - Use smaller model variants
   - Enable gradient checkpointing

3. **Slow Processing**
   - Check GPU utilization
   - Optimize video resolution
   - Consider frame skipping

4. **SLAM Failures**
   - Ensure good lighting
   - Check camera calibration
   - Verify sufficient features

5. **Conda Environment Issues**
   - If environment becomes corrupted:
     ```bash
     conda deactivate
     conda remove -n pathpilot --all
     conda create -n pathpilot python=3.11
     ```
   - If CUDA issues occur:
     ```bash
     conda install -c nvidia cuda-toolkit
     conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --force-reinstall
     ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
MIT License - See LICENSE file for details 