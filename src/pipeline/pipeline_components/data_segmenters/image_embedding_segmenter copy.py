import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
import os


# 1. Download and Load the BLIP Model and Processor
# Using a base model for efficiency.
# This will download the model weights and configuration if not already cached.
# print("Loading model and processor...")
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # Move model to GPU if available for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# print(f"Model loaded on {device}.")

# # Define a directory to save the model
save_directory = "./blip-base"

# # 2. Save the Model and Processor Locally
# print(f"Saving model and processor to {save_directory}...")
# os.makedirs(save_directory, exist_ok=True) # Create the directory if it doesn't exist
# processor.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
# print("Model and processor saved successfully.")


processor = AutoProcessor.from_pretrained(save_directory, use_safetensors=True, trust_remote_code=True)
model = BlipForConditionalGeneration.from_pretrained(save_directory,  use_safetensors=True, trust_remote_code=True)
model.to(device)


# 3. Prepare your Image, Binary Mask, and Bounding Box
dummy_image_np = np.ones((256, 256, 3), dtype=np.uint8) * 255
dummy_image_np[50:150, 50:150] = 0
original_image = Image.fromarray(dummy_image_np).convert("RGB")

binary_mask = np.zeros((256, 256), dtype=np.uint8)
binary_mask[40:160, 40:160] = 1

bbox = [40, 40, 160, 160]  # Example bounding box around the object [x1, y1, x2, y2]

def apply_mask_and_bbox(image: Image.Image, mask: np.ndarray, bbox: list) -> Image.Image:
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")
    black_background = Image.new("RGB", image.size, (0, 0, 0))
    masked_image = Image.composite(image, black_background, mask_pil)
    cropped_image = masked_image.crop(bbox)
    return cropped_image

processed_image = apply_mask_and_bbox(original_image, binary_mask, bbox)
print("Image masked and cropped.")

# 5. Generate the Embedding Vector using BlipForConditionalGeneration's forward pass
text = "A picture of"
with torch.no_grad():
    inputs_for_embedding = processor(images=processed_image, text= text, return_tensors="pt").to(device)

    # Use the vision model to get image features
    vision_outputs = model.vision_model(pixel_values=inputs_for_embedding.pixel_values, return_dict=True)
    
    # Get the pooled image features and apply post-layernorm
    image_features = model.vision_model.post_layernorm(vision_outputs.pooler_output)
    
    normalized_embedding = F.normalize(image_features, p=2, dim=-1)
    print(f"Embedding vector generated. Shape: {normalized_embedding.shape}")

# 6. Generate a Caption for the Masked Object
with torch.no_grad():
    inputs_for_captioning = processor(images=processed_image, text= text, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs_for_captioning.pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"Generated Caption: '{generated_caption}'")