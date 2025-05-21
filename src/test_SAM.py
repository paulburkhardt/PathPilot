from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

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

# Set smaller image size for processing
MAX_IMAGE_SIZE = 384  # Reduced from 512 to 384 for even less memory usage

def resize_image_if_needed(image):
    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def show_mask(mask, ax, random_color=False):
    # Check mask shape - if we have multiple masks (dim > 2)
    if len(mask.shape) > 2 and mask.shape[0] > 1:
        # We have multiple masks, handle each one
        for i in range(mask.shape[0]):
            single_mask = mask[i]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            h, w = single_mask.shape[-2:]
            mask_image = single_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
    else:
        # Original code for single mask
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

# Clear any existing GPU memory
clear_gpu_memory()
print_gpu_mem("start")

try:
    print("Loading model...")
    # Load model with more aggressive memory efficient settings
    model = SamModel.from_pretrained(
        "facebook/sam-vit-base",
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True,
        device_map="auto",  # Let the model decide optimal device mapping
        offload_folder="temp_offload"  # Enable CPU offloading if needed
    )
    print("Model loaded successfully")
    print_gpu_mem("after model load")
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print_gpu_mem("after processor load")

    # Download and process image
    print("Processing image...")
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    
    # Resize image if needed
    raw_image = resize_image_if_needed(raw_image)
    print(f"Processing image of size: {raw_image.size}")
    print_gpu_mem("after image load and resize")
    
    input_points = [[[450 * (raw_image.size[0] / MAX_IMAGE_SIZE), 600 * (raw_image.size[1] / MAX_IMAGE_SIZE)]]]

    # Process inputs in half precision with gradient disabled
    print("Running inference...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        inputs = processor(raw_image, input_points=input_points, return_tensors="pt")
        print_gpu_mem("after processor call")
        # Move inputs to device in chunks
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print_gpu_mem("after inputs to device")
        outputs = model(**inputs)
        print_gpu_mem("after model inference")

        # Immediately move outputs to CPU and clear GPU memory
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu()
        print_gpu_mem("after moving outputs to CPU")

    # Clear GPU memory as soon as possible
    del outputs
    del inputs
    clear_gpu_memory()
    print_gpu_mem("after clearing outputs and inputs")

    print("Masks shape:", masks[0].shape if masks else "No masks generated")
    print("Scores:", scores)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    
    # Display each mask with its score
    if masks:
        for i, mask in enumerate(masks):
            if i < len(scores[0][0]):
                score = scores[0][0][i].item()
                print(f"Displaying mask {i} with score {score:.4f}")
                if score > 0.5:  # Only show higher confidence masks
                    show_mask(mask[0], ax=ax, random_color=True)  # Pass first mask in the batch
    
    plt.axis("off")
    plt.show()

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