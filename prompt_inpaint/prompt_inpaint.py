import cv2
import numpy as np
import tempfile
import torch
from .prompt_based_detection import GroundedSAMPipeline
import requests
import base64
import os
import io
from PIL import Image
import json
from datetime import datetime

# Set paths to your model configuration and weights
GROUNDING_DINO_CONFIG_PATH = "models/grounding_dino/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "models/grounding_dino/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"
# SAM_CHECKPOINT_PATH = "sam_hq_vit_h.pth"

class ImageProcessor:
    def __init__(self):
        # Initialize the pipeline
        self.pipeline = GroundedSAMPipeline(
            grounding_dino_config_path=GROUNDING_DINO_CONFIG_PATH,
            grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            sam_checkpoint_path=SAM_CHECKPOINT_PATH,
            sam_encoder_version="vit_h",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def call_inpainting_api(self, image_path: str, mask_path: str, prompt: str, negative_prompt: str) -> Image.Image:
        """Call the SDXL inpainting API with retry logic"""
        try:
            print("API called")
            files = {
                'image': ('image.png', open(image_path, 'rb'), 'image/png'),
                'mask': ('mask.png', open(mask_path, 'rb'), 'image/png')
            }
            data = {'prompt': prompt, 'negative_prompt': negative_prompt }

            session = requests.Session()
            session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
            print("API called")
            response = session.post(
                "http://localhost:1505/change-background",
                files=files,
                data=data,
                timeout=(30, 300)
            )
            response.raise_for_status()
            print("Received response")
            base64_image = response.json()['base64_image']
            image_bytes = base64.b64decode(base64_image)
            return Image.open(io.BytesIO(image_bytes))
        
        except requests.exceptions.Timeout:
            raise Exception("API request timed out. Please check if the API server is running at localhost:1501")
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to API. Please ensure the API server is running at localhost:1501")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def save_image(self, image_array: np.ndarray, path: str) -> str:
        """Convert numpy array to PIL Image and save it"""
        Image.fromarray(image_array).save(path)
        return path

    def process_image(self, image, prompt, bg_prompt):
        """
        Process the input image and object detection prompt using GroundedSAMPipeline
        """
        os.makedirs("outputs", exist_ok=True)
        print(f"\nProcessing image with:")
        print(f"- Detection prompt: {prompt}")
        print(f"- Background prompt: {bg_prompt}")
        
        # Write the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_filename = tmp.name
            cv2.imwrite(tmp_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            # Run the pipeline with the provided prompt
            _, merged_mask, detections = self.pipeline.run(
                source_image_path=tmp_filename,
                classes=[prompt],
                box_threshold=0.3,
                text_threshold=0.25,
                nms_threshold=0.8
            )
            
            print("\nPipeline results:")
            print(f"- Detections found: {detections is not None}")
            if detections is not None:
                print(f"- Number of detections: {len(detections.xyxy)}")
                print(f"- Class IDs: {detections.class_id}")
            print(f"- Merged mask shape: {merged_mask.shape if merged_mask is not None else None}")
            
            # If no mask was found, return the original image
            if merged_mask is None:
                return image, None
            
            input_path = os.path.join("outputs", f"input.png")
            cv2.imwrite(input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            mask = cv2.bitwise_not(merged_mask)
            mask_path = 'outputs/background_mask.png'
            Image.fromarray(mask).save(mask_path)
            background_prompt = f"{bg_prompt}, photorealistic, high detail 8k uhd, professional photo, natural lighting, clean, ultra realistic"
            negative_prompt = f"Tin can, text"
            result_image = self.call_inpainting_api(input_path, mask_path, background_prompt, negative_prompt)

            output_path = os.path.join("outputs", f"output.png")
            result_image_array = np.array(result_image)
            self.save_image(result_image_array, output_path)
            
            return result_image_array, merged_mask
        
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e