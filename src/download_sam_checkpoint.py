#!/usr/bin/env python3
"""
Script to download SAM model checkpoints from Meta.
"""

import os
import urllib.request
from pathlib import Path

# SAM model checkpoints
CHECKPOINTS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "375MB"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.25GB"
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth", 
        "size": "2.56GB"
    }
}

def download_with_progress(url, filename):
    """Download file with progress bar."""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100.0)
            print(f"\rDownloading {filename}: {percent:.1f}% "
                  f"({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", 
                  end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False

def main():
    """Download SAM checkpoints."""
    print("SAM Checkpoint Downloader")
    print("=" * 40)
    
    # Create checkpoints directory
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    print("\nAvailable models:")
    for model_type, info in CHECKPOINTS.items():
        print(f"  {model_type}: {info['filename']} ({info['size']})")
    
    # Ask user which model to download
    print("\nWhich model would you like to download?")
    print("1. vit_b (recommended, fastest)")
    print("2. vit_l (better quality, slower)")
    print("3. vit_h (best quality, slowest)")
    print("4. all models")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    models_to_download = []
    if choice == "1":
        models_to_download = ["vit_b"]
    elif choice == "2":
        models_to_download = ["vit_l"]
    elif choice == "3":
        models_to_download = ["vit_h"]
    elif choice == "4":
        models_to_download = ["vit_b", "vit_l", "vit_h"]
    else:
        print("Invalid choice. Downloading vit_b by default.")
        models_to_download = ["vit_b"]
    
    # Download selected models
    for model_type in models_to_download:
        info = CHECKPOINTS[model_type]
        filepath = checkpoints_dir / info["filename"]
        
        if filepath.exists():
            print(f"\n{info['filename']} already exists. Skipping.")
            continue
            
        print(f"\nDownloading {model_type} model ({info['size']})...")
        success = download_with_progress(info["url"], str(filepath))
        
        if success:
            print(f"✓ Successfully downloaded {info['filename']}")
        else:
            print(f"✗ Failed to download {info['filename']}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
    
    print(f"\nDownload complete! Checkpoints saved in: {checkpoints_dir.absolute()}")
    print("\nTo use in your script, update the checkpoint path:")
    for model_type in models_to_download:
        info = CHECKPOINTS[model_type]
        filepath = checkpoints_dir / info["filename"]
        if filepath.exists():
            print(f"  {model_type}: sam_checkpoint = '{filepath}'")

if __name__ == "__main__":
    main() 