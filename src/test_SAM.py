from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
import os

# Memory management helpers
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()

def print_gpu_mem(step=""):
    if torch.cuda.is_available():
        print(f"[{step}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
              f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Check for CUDA availability and print memory info
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Using device: {device}")


def show_mask(mask, ax, random_color=False):
    """
    Display a mask on the given axes.
    
    Args:
        mask: A tensor with shape (H, W) or (C, H, W) representing a binary mask
        ax: Matplotlib axes for displaying
        random_color: Whether to use a random color for the mask
    """
    # Generate a distinct color for this mask
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)  # Lower alpha for better layering
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
    
    # Convert to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Print the mask shape for debugging
    print(f"In show_mask: mask shape = {mask.shape}")
    
    # Handle mask with multiple channels
    if len(mask.shape) == 3:
        print(f"Mask has multiple channels: {mask.shape}")
        # Take channel with highest mean value (most likely to be the foreground)
        channel_means = [np.mean(mask[i]) for i in range(mask.shape[0])]
        best_channel = np.argmax(channel_means)
        print(f"Selected channel {best_channel} with mean {channel_means[best_channel]:.4f}")
        mask = mask[best_channel]
    
    # Ensure mask is normalized to 0-1 range
    if mask.max() > 1.0:
        mask = mask / mask.max()
        
    if mask.min() < 0.0:
        mask = (mask - mask.min()) / (mask.max() - mask.min())

    h, w = mask.shape
    
    # Use matplotlib's built-in alpha compositing for overlays
    masked_image = np.ones((h, w, 4))
    masked_image[:, :, 0] = color[0]  # R
    masked_image[:, :, 1] = color[1]  # G
    masked_image[:, :, 2] = color[2]  # B
    masked_image[:, :, 3] = mask * color[3]  # Alpha channel
    
    # This will properly overlay the mask on existing content
    ax.imshow(masked_image)

# Clear any existing GPU memory
clear_gpu_memory()
print_gpu_mem("start")

try:
    print("Loading model...")
    # Load model with higher quality settings
    model = SamModel.from_pretrained(
        "facebook/sam-vit-base",
        # Remove half precision to maintain full quality
        # torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # Keep this as it doesn't affect quality
        device_map="auto"
        # Remove offload folder to prevent quality loss
        # offload_folder="temp_offload"
    )
    print("Model loaded successfully")
    print_gpu_mem("after model load")
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print_gpu_mem("after processor load")

    # Download and process image
    print("Processing image...")
    input_image_path = "inputs/yoga_lady.png"
    raw_image = Image.open(input_image_path)
    print(f"Processing image of size: {raw_image.size}")
    print_gpu_mem("after image load")
    
    def create_point_grid(image_width, image_height, grid_size=50):
        """
        Create a grid of points across the image.
        
        Args:
            image_width: Width of the image
            image_height: Height of the image
            grid_size: Spacing between points in pixels
            
        Returns:
            List of points in format [[x1, y1], [x2, y2], ...]
        """
        points = []
        for y in range(grid_size, image_height - grid_size, grid_size):
            for x in range(grid_size, image_width - grid_size, grid_size):
                points.append([x, y])
        return points

    # Create grid of points
    width, height = raw_image.size
    grid_points = create_point_grid(width, height, grid_size=100)  # Adjust grid_size as needed
    
    # Process points in batches to avoid memory issues
    BATCH_SIZE = 16  # Adjust this value based on your GPU memory
    all_outputs = []
    
    for i in range(0, len(grid_points), BATCH_SIZE):
        batch_points = grid_points[i:i + BATCH_SIZE]
        # Format for SAM: shape (1, num_points, 2) - all points processed for same image
        input_points = [[[x, y] for x, y in batch_points]]
        input_labels = [[1 for _ in batch_points]]  # 1 for foreground point
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(grid_points) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        # Process inputs with full precision
        with torch.no_grad():
            inputs = processor(raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")
            print_gpu_mem("after processor call")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print_gpu_mem("after inputs to device")
            outputs = model(**inputs)
            print_gpu_mem("after model inference")
            
            # Store or process the outputs
            all_outputs.append(outputs)
            
            # Clear memory after each batch
            del inputs
            clear_gpu_memory()

    # Combine all outputs
    combined_masks = []
    combined_scores = []
    
    for outputs in all_outputs:
        # Process the outputs at full quality
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            [[height, width]],  # Original image size as list of lists
            [outputs.pred_masks.shape[-2:]]  # Input size as list of tuples
        )
        scores = outputs.iou_scores.cpu()
        
        if isinstance(masks, list):
            combined_masks.extend(masks)
        else:
            combined_masks.append(masks)
            
        if isinstance(scores, list):
            combined_scores.extend(scores)
        else:
            combined_scores.append(scores.squeeze())
        
        del masks
        del scores
    
    # Print mask info for debugging
    if combined_masks:
        print(f"Total number of masks: {len(combined_masks)}")
        if isinstance(combined_masks[0], torch.Tensor):
            print(f"First mask shape: {combined_masks[0].shape}")

    print_gpu_mem("before visualization")

    # Create high quality visualization
    plt.figure(figsize=(20, 20), dpi=300)  # Higher resolution figure
    
    # First display the original image
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    
    # Sort masks by score
    mask_score_pairs = []
    for idx, (mask, score) in enumerate(zip(combined_masks, combined_scores)):
        if isinstance(score, torch.Tensor):
            score = score.mean().item()
        mask_score_pairs.append((idx, mask, score))
    
    # Sort by score in descending order
    mask_score_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Display only top N masks
    MAX_MASKS = 50  # Limit number of displayed masks
    displayed_count = 0
    
    print(f"\nDisplaying top {MAX_MASKS} masks out of {len(mask_score_pairs)} total masks...")
    
    # Create output directory for individual masks if it doesn't exist
    masks_dir = "outputs/individual_masks"
    os.makedirs(masks_dir, exist_ok=True)
    
    for idx, mask, score in mask_score_pairs:
        if score > 0.6 and displayed_count < MAX_MASKS:  # Lower threshold to include more potential masks
            print(f"Displaying mask {idx} with score {score:.4f} ({displayed_count + 1}/{MAX_MASKS})")
            
            # Ensure mask is in correct format (H, W)
            if isinstance(mask, torch.Tensor):
                if len(mask.shape) == 4:  # (1, num_masks, H, W)
                    mask = mask[0, 0]
                elif len(mask.shape) == 3:  # (num_masks, H, W)
                    mask = mask[0]
            
            # Save individual mask
            plt.figure(figsize=(20, 20))
            plt.imshow(np.array(raw_image))  # Show original image
            show_mask(mask, plt.gca(), random_color=False)  # Show mask in blue
            plt.axis('off')
            mask_file = os.path.join(masks_dir, f"mask_{idx:03d}_score_{score:.3f}.png")
            plt.savefig(mask_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Also save binary mask (just the mask without the image)
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            binary_mask = (mask > 0).astype(np.uint8) * 255
            binary_mask_file = os.path.join(masks_dir, f"mask_{idx:03d}_score_{score:.3f}_binary.png")
            Image.fromarray(binary_mask).save(binary_mask_file)
            
            # Add to combined visualization
            show_mask(mask, ax=ax, random_color=True)
            displayed_count += 1
            
            # Save intermediate combined results every 10 masks
            if displayed_count % 10 == 0:
                plt.savefig(f"outputs/sam_segmentation_intermediate_{displayed_count}.png", 
                          bbox_inches='tight', dpi=300)
                print(f"Saved intermediate result with {displayed_count} masks")
    
    plt.axis("off")
    
    # Save final output
    output_file = f"outputs/sam_segmentation_result_{input_image_path.split('/')[-1]}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)  # Higher DPI for better quality
    print(f"\nFinal visualization saved to {output_file}")
    plt.close()
    
    # Clean up memory after visualization is complete
    del combined_masks
    del combined_scores
    clear_gpu_memory()
    print_gpu_mem("after visualization and cleanup")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise

finally:
    # Always clear memory when done
    if 'model' in locals():
        del model
    if 'processor' in locals():
        del processor
    clear_gpu_memory()
    print_gpu_mem("final cleanup")