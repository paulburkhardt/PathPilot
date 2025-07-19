#!/usr/bin/env python3
"""
Video Converter for Rerun Compatibility

Converts videos in a directory to rerun-compatible format.
Handles HDR, 10-bit, Dolby Vision, and other incompatible video formats.

Rerun video limitations addressed:
- #7354: Only MP4 container format supported
- #7755: No AV1 support on Linux ARM  
- #5181: No audio support (preserved but ignored)
- #7594: HDR video not supported

Usage:
    python convert_videos_for_rerun.py /path/to/video/folder
    python convert_videos_for_rerun.py /path/to/video/folder --output-suffix "_rerun"
    python convert_videos_for_rerun.py /path/to/video/folder --force-reconvert
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os


class VideoConverter:
    """Convert videos to rerun-compatible format."""
    
    def __init__(self, input_dir: str, force_reconvert: bool = False):
        self.input_dir = Path(input_dir)
        self.force_reconvert = force_reconvert
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def find_video_files(self) -> List[Path]:
        """Find all video files in the input directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f"*{ext}"))
            video_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        return sorted(video_files)
    
    def analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """Analyze video format to determine if conversion is needed."""
        try:
            result = subprocess.run([
                '/usr/bin/ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ], capture_output=True, text=True, check=True)
            
            video_info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return {'needs_conversion': False, 'reasons': [], 'error': 'No video stream found'}
            
            # Check compatibility
            needs_conversion = False
            reasons = []
            
            # Check container format
            format_name = video_info.get('format', {}).get('format_name', '')
            if 'mp4' not in format_name.lower():
                needs_conversion = True
                reasons.append(f"Container format: {format_name} (rerun needs MP4)")
            
            # Check for 10-bit color (HDR indicator)
            pix_fmt = video_stream.get('pix_fmt', '')
            if any(fmt in pix_fmt for fmt in ['10le', '10be', 'p10']):
                needs_conversion = True
                reasons.append(f"10-bit color depth: {pix_fmt}")
            
            # Check for High 10 profile (HDR-capable)
            profile = video_stream.get('profile', '')
            if 'High 10' in profile:
                needs_conversion = True
                reasons.append(f"High 10 profile: {profile}")
            
            # Check for HDR color spaces
            colorspace = video_stream.get('color_space', '')
            if colorspace in ['bt2020nc', 'bt2020c']:
                needs_conversion = True
                reasons.append(f"HDR color space: {colorspace}")
            
            # Check for Dolby Vision (HDR metadata)
            if 'side_data_list' in video_stream:
                for side_data in video_stream['side_data_list']:
                    if 'DOVI' in side_data.get('side_data_type', ''):
                        needs_conversion = True
                        reasons.append("Dolby Vision metadata")
                        break
            
            # Check codec compatibility
            codec_name = video_stream.get('codec_name', '')
            if codec_name == 'av1':
                needs_conversion = True
                reasons.append("AV1 codec (not supported on Linux ARM)")
            
            return {
                'needs_conversion': needs_conversion,
                'reasons': reasons,
                'video_info': video_stream,
                'format_info': video_info.get('format', {})
            }
            
        except subprocess.CalledProcessError as e:
            return {'needs_conversion': False, 'reasons': [], 'error': f'ffprobe failed: {e}'}
        except json.JSONDecodeError as e:
            return {'needs_conversion': False, 'reasons': [], 'error': f'JSON decode failed: {e}'}
        except Exception as e:
            return {'needs_conversion': False, 'reasons': [], 'error': f'Analysis failed: {e}'}
    
    def convert_video(self, input_path: Path, output_path: Path) -> bool:
        """Convert video to rerun-compatible format."""
        print(f"Converting: {input_path.name} -> {output_path.name}")
        # Use temporary file during conversion
        temp_output = input_path.parent / f".{input_path.stem}_temp_convert.mp4"
        
        # FFmpeg command for maximum rerun compatibility
        ffmpeg_cmd = [
            '/usr/bin/ffmpeg', '-y',  # Use system ffmpeg, overwrite output
            '-i', str(input_path),
            
            # Video encoding - H.264 Constrained Baseline for maximum compatibility
            '-c:v', 'libx264',
            '-profile:v', 'baseline',  # Most compatible profile
            '-level', '3.0',           # Compatible with most devices
            '-pix_fmt', 'yuv420p',     # 8-bit 4:2:0 (no 10-bit)
            
            # Color space - Standard Rec.709 (no HDR)
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            
            # Container optimization
            '-movflags', '+faststart',  # Optimize for web streaming
            
            # Audio - preserve but rerun will ignore it
            '-c:a', 'aac',
            '-b:a', '128k',
            
            # Output
            str(temp_output)
        ]
        
        try:
            print("  Running ffmpeg conversion...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ❌ Conversion failed: {result.stderr}")
                if temp_output.exists():
                    temp_output.unlink()  # Clean up temp file
                return False
            
            # Move temp file to final output location
            if output_path.exists():
                output_path.unlink()  # Remove existing output if it exists
            temp_output.rename(output_path)  # Rename temp to final output
            
            # Remove original file if it's different from output
            if input_path != output_path and input_path.exists():
                input_path.unlink()
            
            print(f"  ✅ Conversion successful -> {output_path.name}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Conversion failed: {e}")
            if temp_output.exists():
                temp_output.unlink()  # Clean up temp file
            return False
    
    def get_output_path(self, input_path: Path) -> Path:
        """Generate output path with .mp4 extension."""
        return input_path.with_suffix('.mp4')
    
    def should_convert(self, input_path: Path, output_path: Path, analysis: Dict[str, Any]) -> bool:
        """Determine if conversion should be performed."""
        # Force reconvert if requested
        if self.force_reconvert:
            return True
        
        # Only convert if analysis says it's needed
        return analysis.get('needs_conversion', False)
    
    def process_videos(self) -> None:
        """Process all videos in the input directory."""
        video_files = self.find_video_files()
        
        if not video_files:
            print(f"No video files found in: {self.input_dir}")
            return
        
        print(f"Found {len(video_files)} video files in: {self.input_dir}")
        print("=" * 60)
        
        converted_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            

            
            # Analyze video
            analysis = self.analyze_video(video_file)
            
            if 'error' in analysis:
                print(f"  ❌ Analysis error: {analysis['error']}")
                error_count += 1
                continue
            
            # Generate output path
            output_path = self.get_output_path(video_file)
            
            # Check if conversion is needed
            if not self.should_convert(video_file, output_path, analysis):
                print(f"  ⏭️  Skipping (no conversion needed): {video_file.name}")
                skipped_count += 1
                continue
            
            # Show analysis results
            print(f"  🔄 Conversion needed:")
            for reason in analysis['reasons']:
                print(f"     - {reason}")
            
            # Convert video
            if self.convert_video(video_file, output_path):
                converted_count += 1
            else:
                error_count += 1
        
        # Summary
        print("\n" + "=" * 60)
        print(f"SUMMARY:")
        print(f"  ✅ Converted: {converted_count}")
        print(f"  ⏭️  Skipped: {skipped_count}")
        print(f"  ❌ Errors: {error_count}")
        print(f"  📁 Total processed: {len(video_files)}")
        
        if converted_count > 0:
            print(f"\nVideos have been converted to rerun-compatible MP4 format.")
            print("Original non-MP4 files have been replaced with MP4 versions.")
            print("You can now use these videos with the SLAM visualizer.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert videos to rerun-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_videos_for_rerun.py Data/Videos/
    python convert_videos_for_rerun.py Data/Videos/ --force-reconvert

Rerun Video Requirements:
    - Container: MP4 only
    - Codec: H.264 (no AV1 on Linux ARM)
    - Profile: Baseline/Main (no High 10)
    - Color: 8-bit SDR (no 10-bit HDR)
    - No Dolby Vision metadata
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory containing video files to convert'
    )
    
    parser.add_argument(
        '--force-reconvert', 
        action='store_true',
        help='Force reconversion of all videos, even if they are already compatible'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Check if system ffmpeg is available
        subprocess.run(['/usr/bin/ffmpeg', '-version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: System ffmpeg not found at /usr/bin/ffmpeg")
        print("Please install ffmpeg: sudo apt install ffmpeg")
        sys.exit(1)
    
    try:
        converter = VideoConverter(
            input_dir=args.input_dir,
            force_reconvert=args.force_reconvert
        )
        converter.process_videos()
        
    except KeyboardInterrupt:
        print("\nConversion interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 