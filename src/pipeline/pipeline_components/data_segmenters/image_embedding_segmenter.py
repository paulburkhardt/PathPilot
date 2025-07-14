import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
import os
import cv2
from typing import List, Dict, Any
from .abstract_data_segmenter import AbstractDataSegmenter



class ImageEmbeddingSegmenter(AbstractDataSegmenter):
    """
    Component for processing images with masks to generate embeddings and descriptions.
    
    Args:
        -
    Returns:
        -
    Raises:
        -
    """

    def __init__(self, 
                text= "A picture of"
                ) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        save_directory = "./blip-base"
        self.processor = AutoProcessor.from_pretrained(save_directory, use_safetensors=True, trust_remote_code=True)
        self.model = BlipForConditionalGeneration.from_pretrained(save_directory,  use_safetensors=True, trust_remote_code=True)
        self.model.to(self.device)
        self.text = text

    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires images and masks as input."""
        return ["image","image_segmentation_mask","key_frame_flag"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs embeddings and descriptions."""
        return ["embeddings", "descriptions"]
    


    def _run(self, image: Any, image_segmentation_mask, key_frame_flag, **kwargs: Any) -> Dict[str, Any]:
        """
        Process an image with image_segmentation_mask to generate embeddings and descriptions.
        
        Args:
            image: The input image
            image_segmentation_mask: Binary image_segmentation_mask for object segmentation
            key_frame_flag: Flag indicating if this is a key frame
            **kwargs: Additional unused arguments
        Returns:
            Dictionary containing embeddings and descriptions
        """
        if not key_frame_flag:
            return {
            "embeddings": None,
            "descriptions": None
            }

        embeddings = {}
        descriptions = {}

        for mask_key, mask in image_segmentation_mask.as_binary_dict().items():
            processed_image = self.apply_mask_and_blur(image, mask)
            processed_image_pil = Image.fromarray(processed_image)
            generated_caption, embedding = self.image_information_extraction(processed_image_pil, mask)
            embeddings[mask_key] = embedding
            descriptions[mask_key] = generated_caption

        return {
            "embeddings": embeddings,
            "descriptions": descriptions
        }
    
    def apply_mask_and_blur(self, image: np.ndarray, mask: np.ndarray):
        # Convert image to numpy array if it's not already
        rgb_image = image.as_numpy()
        # Convert normalized float image [0,1] to 0-255 uint8 if needed
        if rgb_image.dtype in [np.float32, np.float64] and rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        elif rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        blur = cv2.GaussianBlur(rgb_image, (51, 51), 0)
        focused = np.where(mask[..., None], rgb_image, blur)
        return focused
    

    def apply_mask_and_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: #NOTE Cropping is shit apparantly for BLIP. its better to blur
        """
        Apply binary mask to image and crop to the mask bounds.
        
        Args:
            image: Input image as numpy array
            mask: Binary mask as numpy array (0s and 1s)
            
        Returns:
            Cropped and masked image as numpy array
        """

        rgb_image = image.as_numpy()
        # Convert normalized float image [0,1] to 0-255 uint8 if needed
        if rgb_image.dtype in [np.float32, np.float64] and rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        elif rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        # Apply mask to image (element-wise multiplication)
        masked_image = rgb_image * mask[..., np.newaxis] if len(rgb_image.shape) == 3 else rgb_image * mask
        
        # Find bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Crop to mask bounds
            cropped_image = masked_image[rmin:rmax+1, cmin:cmax+1]
        else:
            # If no mask found, return the original image
            cropped_image = rgb_image
        
        return cropped_image

    def _create_adaptive_patch_mask(self, mask, num_patches):
        """Create an adaptive patch mask for non-standard patch counts."""
        # Convert mask to binary and get object center
        mask_binary = mask.astype(bool)
        if not np.any(mask_binary):
            # No object in mask, return all zeros
            return torch.zeros((1, num_patches), device=self.device)
        
        # Find object center
        y_coords, x_coords = np.where(mask_binary)
        center_y, center_x = np.mean(y_coords), np.mean(x_coords)
        
        # Normalize center to [0, 1] range
        center_y_norm = center_y / mask.shape[0]
        center_x_norm = center_x / mask.shape[1]
        
        # Create patch mask based on center position
        patch_mask = torch.zeros((1, num_patches), device=self.device)
        
        # Estimate patch grid dimensions (try to find factors of num_patches)
        factors = []
        for i in range(1, int(np.sqrt(num_patches)) + 1):
            if num_patches % i == 0:
                factors.append((i, num_patches // i))
        
        if factors:
            # Use the most square-like factor pair
            h_patches, w_patches = max(factors, key=lambda x: min(x[0], x[1]) / max(x[0], x[1]))
            
            # Calculate which patches should be active based on object center
            center_patch_y = int(center_y_norm * h_patches)
            center_patch_x = int(center_x_norm * w_patches)
            
            # Activate patches around the center
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    patch_y = center_patch_y + dy
                    patch_x = center_patch_x + dx
                    if 0 <= patch_y < h_patches and 0 <= patch_x < w_patches:
                        patch_idx = patch_y * w_patches + patch_x
                        if patch_idx < num_patches:
                            patch_mask[0, patch_idx] = 1
        else:
            # Fallback: activate center patches
            center_patches = max(1, num_patches // 16)
            start_idx = (num_patches - center_patches) // 2
            patch_mask[0, start_idx:start_idx + center_patches] = 1
        
        return patch_mask

    def image_information_extraction(self,processed_image_pil,mask):
        # 5. Generate the Embedding Vector and Caption using BlipForConditionalGeneration
        with torch.no_grad():
            # inputs = self.processor(images=processed_image_pil, text=self.text, return_tensors="pt").to(self.device)

            # Use the vision model to get image features
            # vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values, return_dict=True)
            # image_features = self.model.vision_model.post_layernorm(vision_outputs.pooler_output)
            



            # 4. Encode image
            inputs = self.processor(images=processed_image_pil, return_tensors="pt").to(self.device)
            vision_out = self.model.vision_model(pixel_values=inputs.pixel_values, output_hidden_states=False)
            all_embeds = vision_out.last_hidden_state  # [1, N+1, dim]; N = patch count
            cls_embed, patch_embeds = all_embeds[:, :1, :], all_embeds[:, 1:, :]
            num_patches = patch_embeds.shape[1]

            # 5. Build patch_mask aligned to patch embeddings
            H, W = inputs.pixel_values.shape[2:]
            ps = self.model.config.vision_config.patch_size
            hp, wp = H // ps, W // ps
            mask_small = cv2.resize(mask.astype(np.uint8), (wp, hp), interpolation=cv2.INTER_NEAREST)
            flat_mask = torch.tensor(mask_small.reshape(-1), device=self.device).unsqueeze(0)  # [1, hp*wp]
            if flat_mask.shape[1] != num_patches:
                flat_mask = flat_mask[:, :num_patches]  # or handle adaptive logic

            # 6. Apply mask to patch embeddings
            patch_embeds_masked = patch_embeds * flat_mask.unsqueeze(-1)

            # 7. Reconstruct embeddings + attention mask (include CLS)
            encoder_hidden_states = torch.cat([cls_embed, patch_embeds_masked], dim=1)
            encoder_attention_mask = torch.cat([
                torch.ones((1,1), device=self.device),
                flat_mask
            ], dim=1)  # 1 for CLS + patches

            # 8. Extract embedding vector (mean‑pool object patches + normalize)
            obj_emb = patch_embeds_masked[flat_mask.bool()].mean(dim=0)
            normalized_embedding = F.normalize(obj_emb, p=2, dim=-1).unsqueeze(0)  # Add batch dimension

            # 9. Generate caption using masked encoder with better prompting
            # Use a simple prompt for adjective + object format
            dec_input = self.processor.tokenizer(self.text, return_tensors="pt").to(self.device)
            
            generated = self.model.text_decoder.generate(
                input_ids=dec_input.input_ids,
                attention_mask=dec_input.attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=4,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.5,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            generated_caption = self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
            # Clean up the caption to get just the adjective + object
            if generated_caption.startswith("a "):
                generated_caption = generated_caption[2:].strip()
            elif generated_caption.startswith("an "):
                generated_caption = generated_caption[3:].strip()
            elif generated_caption.startswith(self.text):
                generated_caption = generated_caption[len(self.text):].strip()


            # # Generate a Caption for the Masked Object
            # generated_ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=50)
            # generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            print(f"Generated Caption: '{generated_caption}'")

        return generated_caption, normalized_embedding