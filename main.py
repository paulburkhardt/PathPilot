import argparse
import yaml
from pathlib import Path
from src.video_processor import VideoProcessor
from src.slam_processor import SLAMProcessor
from src.segmentation import SegmentationProcessor
from src.distance_calculator import DistanceCalculator
from src.warning_system import WarningSystem
from src.vlm_processor import VLMProcessor

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='PathPilot - Obstacle Detection System')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize components
    video_processor = VideoProcessor(config['video'])
    slam_processor = SLAMProcessor(config['slam'])
    segmentation_processor = SegmentationProcessor(config['segmentation'])
    distance_calculator = DistanceCalculator(config['distance'])
    warning_system = WarningSystem(config['warning'])
    vlm_processor = VLMProcessor(config['vlm'])

    # Process video
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Main processing loop
    for frame in video_processor.process_video(video_path):
        # Get 3D point cloud from SLAM
        point_cloud = slam_processor.process_frame(frame)
        
        # Segment objects in the frame
        segmented_objects = segmentation_processor.process_frame(frame)
        
        # Calculate distances to objects
        distances = distance_calculator.calculate_distances(point_cloud, segmented_objects)
        
        # Get object descriptions using BLIP-2
        object_descriptions = vlm_processor.describe_objects(frame, segmented_objects)
        
        # Generate warnings if needed
        warnings = warning_system.generate_warnings(distances, object_descriptions)
        
        # Display warnings
        for warning in warnings:
            print(warning)

if __name__ == "__main__":
    main() 