import torch
from PIL import Image
import requests
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForConditionalGeneration
from huggingface_hub import snapshot_download

# 1. Load BLIP Model and Processor
def load_blip_model():
    """
    Loads the BLIP model and its corresponding processor from a local directory.
    """
    local_dir = "/usr/prakt/s0120/PathPilot/blip-base"
    processor = AutoProcessor.from_pretrained(local_dir)
    model = BlipForConditionalGeneration.from_pretrained(local_dir)
    model.eval()  # Set model to evaluation mode
    return processor, model

# 2. Apply Mask to Image (Pre-processing Step)
def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Applies a binary mask to a PIL Image. Pixels outside the mask are set to black.

    Args:
        image (PIL.Image.Image): The original input image.
        mask (np.ndarray): A binary mask (2D numpy array, 0s and 1s) of the same
                          dimensions as the image, where 1 indicates the object.

    Returns:
        PIL.Image.Image: The masked image with background pixels set to black.
    """
    if image.size[::-1] != mask.shape:
        raise ValueError(f"Image dimensions {image.size[::-1]} and mask dimensions {mask.shape} must match.")

    # Convert mask to a PIL Image (L mode for 1-bit pixels)
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255, 'L')

    # Create a black background image
    black_background = Image.new("RGB", image.size, (0, 0, 0))

    # Composite the original image onto the black background using the mask
    # The mask_pil.convert("L") ensures it's a single channel mask for Image.composite
    masked_image = Image.composite(image, black_background, mask_pil)
    return masked_image

# 3. Get Image Embedding
def get_image_embedding(processor, model, image: Image.Image) -> torch.Tensor:
    """
    Generates a normalized embedding vector for an image using the BLIP model.
    """
    # Process the image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # Get image features (embeddings)
        # BlipForImageTextRetrieval has get_image_features
        image_features = model.get_image_features(**inputs)

    # Normalize the embedding for cosine similarity
    normalized_embedding = F.normalize(image_features, p=2, dim=-1)
    return normalized_embedding

# 4. Get Text Embedding
def get_text_embedding(processor, model, text: str) -> torch.Tensor:
    """
    Generates a normalized embedding vector for text using the BLIP model.
    """
    # Process the text for the model
    inputs = processor(text=text, return_tensors="pt")

    with torch.no_grad():
        # Get text features (embeddings)
        # BlipForImageTextRetrieval has get_text_features
        text_features = model.get_text_features(**inputs)

    # Normalize the embedding for cosine similarity
    normalized_embedding = F.normalize(text_features, p=2, dim=-1)
    return normalized_embedding

# 5. Calculate Image-Text Similarity
def calculate_image_text_similarity(image_embedding: torch.Tensor, text_embedding: torch.Tensor, model) -> float:
    """
    Calculates the cosine similarity between an image embedding and a text embedding.
    Applies the logit_scale from the model for CLIP/BLIP-compatible similarity scores.
    """
    # Ensure embeddings are 2D (batch_size, embedding_dim)
    image_embedding = image_embedding.squeeze(0)
    text_embedding = text_embedding.squeeze(0)

    # Calculate dot product (cosine similarity for normalized vectors)
    similarity = torch.dot(image_embedding, text_embedding)

    # Apply the logit_scale from the model for CLIP/BLIP-compatible scores
    # This is typically exp(model.logit_scale)
    logit_scale = model.logit_scale.exp()
    scaled_similarity = similarity * logit_scale

    return scaled_similarity.item()

# --- Example Usage ---
if __name__ == "__main__":
    # Load model and processor
    print("Loading BLIP model...")
    blip_processor, blip_model = load_blip_model()
    print("Model loaded.")

    # --- Step 1: Prepare an example image and a dummy mask ---
    # In a real scenario, you would load your image and mask from files
    # For demonstration, we'll fetch an image and create a dummy mask.
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg" # A cat image
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"Original image size: {image.size}")

    # Create a dummy mask (e.g., a central square mask)
    # In your application, this 'mask_array' would come from your segmentation model
    mask_array = np.zeros(image.size[::-1], dtype=bool) # (height, width)
    h, w = mask_array.shape
    # Simulate a mask for the central part of the image (e.g., where the cat is)
    mask_array[h//4:3*h//4, w//4:3*w//4] = True
    print(f"Dummy mask created with shape: {mask_array.shape}")

    # Apply the mask to the image
    masked_image = apply_mask_to_image(image, mask_array)
    print("Mask applied to image.")
    # You can save or display the masked_image to verify
    # masked_image.save("masked_cat_image.jpg")
    # masked_image.show()

    # --- Step 2: Get the embedding vector of the masked image ---
    print("Generating image embedding...")
    image_embedding = get_image_embedding(blip_processor, blip_model, masked_image)
    print(f"Image embedding shape: {image_embedding.shape}") # Should be (1, embedding_dim)

    # --- Step 3: Use this embedding vector for image-to-text comparison ---
    text_queries = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a masked object",
        "a photo of a car"
    ]

    print("\nCalculating image-to-text similarities:")
    for query_text in text_queries:
        text_embedding = get_text_embedding(blip_processor, blip_model, query_text)
        similarity_score = calculate_image_text_similarity(image_embedding, text_embedding, blip_model)
        print(f"  Image vs '{query_text}': {similarity_score:.4f}")

    # --- Optional: Image-to-Image comparison (for completeness) ---
    print("\nOptional: Demonstrating Image-to-Image comparison:")
    image_url_dog = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Golden_Retriever_with_a_ball.jpg/800px-Golden_Retriever_with_a_ball.jpg"
    dog_image = Image.open(requests.get(image_url_dog, stream=True).raw).convert("RGB")

    # Create a dummy mask for the dog image (e.g., full image for simplicity)
    dog_mask_array = np.ones(dog_image.size[::-1], dtype=bool)
    masked_dog_image = apply_mask_to_image(dog_image, dog_mask_array)

    dog_image_embedding = get_image_embedding(blip_processor, blip_model, masked_dog_image)

    # Cosine similarity between two normalized image embeddings
    # Note: For image-to-image, you typically don't apply logit_scale unless you're
    # comparing it in the same way CLIP/BLIP calculates image-text logits.
    # For pure cosine similarity, just use F.cosine_similarity or dot product of normalized vectors.
    image_to_image_similarity = F.cosine_similarity(image_embedding, dog_image_embedding, dim=-1).item()
    print(f"  Masked Cat Image vs Masked Dog Image (cosine similarity): {image_to_image_similarity:.4f}")

    # --- Option 1: Use huggingface_hub to download the model ---
    print("\nOption 1: Downloading BLIP model using huggingface_hub...")
    try:
        snapshot_download(
            repo_id="Salesforce/blip-base",
            local_dir="/usr/prakt/s0120/PathPilot/blip-base",
            local_dir_use_symlinks=False
        )
        print("BLIP model downloaded successfully using huggingface_hub.")
    except Exception as e:
        print(f"Error downloading BLIP model using huggingface_hub: {e}")