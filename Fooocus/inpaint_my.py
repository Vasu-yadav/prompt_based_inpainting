import gradio as gr
import numpy as np
from PIL import Image
import modules.config
import modules.flags as flags
import args_manager
import modules.async_worker as worker
from typing import Optional, Union, Dict, List, Tuple
import time
import os
import cv2
from pathlib import Path
import random

class InpaintingPipeline:
    def __init__(self):
        """Initialize the inpainting pipeline"""
        self.current_task = None
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path and convert to RGB numpy array"""
        image_path = Path(image_path) if isinstance(image_path, str) else image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        empty_mask = np.zeros_like(image)
        return image, empty_mask
    
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Load mask from path and ensure correct format"""
        mask_path = Path(mask_path) if isinstance(mask_path, str) else mask_path
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
            
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
            
        return mask

    def prepare_inputs(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
        empty_image: Union[np.ndarray, Image.Image],
        prompt: str,
        base_model: str = "juggernautXL_v8Rundiffusion.safetensors",
        refiner_model: str = "None",
        negative_prompt: str = "",
        inpaint_engine: str = "v1",
        inpaint_strength: float = 1.0,
        inpaint_respective_field: float = 0.618,
    ) -> List:
        """Prepare inputs for the inpainting pipeline"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        # if mask.ndim == 3:
        #     mask = mask[:, :, 0]

        # LoRA settings - for each of 4 possible LoRAs:
        # (enabled (bool), model name (str), weight (float))
        
        task_args = [
            True,  # generate_image_grid
            "",  # prompt
            negative_prompt,  # negative_prompt
            ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp'],  # style_selections
            'Speed',  # performance_selection
            '1152×896',  # aspect_ratios_selection  
            1,  # image_number
            'png',  # output_format
            str(random.randint(1, 1e10)),  # seed
            False,  # read_wildcards_in_order
            2,  # sharpness
            4,  # cfg_scale
            base_model,  # base_model_name
            refiner_model,  # refiner_model_name - disabled to avoid loading issues
            0.5,  # refiner_switch
            True,                 # Unknown boolean flag
            'sd_xl_offset_example-lora_1.0.safetensors',  # LoRA model
            0.1,                  # LoRA weight
            True,                 # LoRA enabled
            'None',               # Additional LoRA 1
            1,                    # Additional LoRA 1 weight
            True,                 # Additional LoRA 1 enabled
            'None',               # Additional LoRA 2
            1,                    # Additional LoRA 2 weight
            True,                 # Additional LoRA 2 enabled
            'None',               # Additional LoRA 3
            1,                    # Additional LoRA 3 weight
            True,                 # Additional LoRA 3 enabled
            'None',               # Additional LoRA 4
            1,                    # Additional LoRA 4 weight # All LoRA arguments
            True,  # input_image_checkbox  
            'inpaint',  # current_tab
            'Disabled',  # uov_method
            None,  # uov_input_image
            [],  # outpaint_selections
            {'image': image, 'mask': mask},  # inpaint_input_image 
            prompt,  # inpaint_additional_prompt
            {'image': image, 'mask': mask},  # inpaint_mask_image_upload
            False,  # disable_preview
            False,  # disable_intermediate_results
            False,  # disable_seed_increment
            False,  # black_out_nsfw
            1.5,  # adm_scaler_positive
            0.8,  # adm_scaler_negative 
            0.3,  # adm_scaler_end
            7,  # adaptive_cfg
            2,  # clip_skip
            'dpmpp_2m_sde_gpu',  # sampler_name
            'karras',  # scheduler_name
            'ponyDiffusionV6XL_vae.safetensors',  # vae_name
            -1,  # overwrite_step
            -1,  # overwrite_switch
            -1,  # overwrite_width
            -1,  # overwrite_height
            -1,  # overwrite_vary_strength
            -1,  # overwrite_upscale_strength
            False,  # mixing_image_prompt_and_vary_upscale
            False,  # mixing_image_prompt_and_inpaint
            False,  # debugging_cn_preprocessor
            False,  # skipping_cn_preprocessor
            64,  # canny_low_threshold
            128,  # canny_high_threshold
            'joint',  # refiner_swap_method
            0.25,  # controlnet_softness
            False,  # freeu_enabled
            1.01,  # freeu_b1
            1.02,  # freeu_b2
            0.99,  # freeu_s1
            0.95,  # freeu_s2
            False,  # debugging_inpaint_preprocessor
            True,  # inpaint_disable_initial_latent  
            'v2.6',  # inpaint_engine
            1,  # inpaint_strength
            0,  # inpaint_respective_field
            False,  # inpaint_advanced_masking_checkbox
            False,  # invert_mask_checkbox
            0,  # inpaint_erode_or_dilate
            False,  # save_final_enhanced_image_only
            False,  # save_metadata_to_images
            'fooocus',  # metadata_scheme
            None,  # cn_ip1
            0.5,  # cn_ip1_stop
            0.6,  # cn_ip1_weight  
            'ImagePrompt',  # cn_ip1_type
            None,  # cn_ip2
            0.5,  # cn_ip2_stop
            0.5,  # cn_ip2_weight
            'ImagePrompt',  # cn_ip2_type
            None,  # cn_ip3  
            0.5,  # cn_ip3_stop
            0.5,  # cn_ip3_weight
            'ImagePrompt',  # cn_ip3_type
            None,  # cn_ip4
            0.5,  # cn_ip4_stop
            0.5,  # cn_ip4_weight
            'ImagePrompt',  # cn_ip4_type
            False,  # debugging_dino
            0,  # dino_erode_or_dilate  
            False,  # debugging_enhance_masks_checkbox
            None,  # enhance_input_image
            False,  # enhance_checkbox
            'Disabled',  # enhance_uov_method
            'Before First Enhancement',  # enhance_uov_processing_order
            'Original Prompts'  # enhance_uov_prompt_type
        ]
        
        # Add enhance control tabs
        for _ in range(modules.config.default_enhance_tabs):
            tab_args = [
                False,  # enhance_enabled
                '',     # enhance_mask_dino_prompt_text 
                '',     # enhance_prompt
                '',     # enhance_negative_prompt
                'sam', # enhance_mask_model
                'full',  # enhance_mask_cloth_category
                'vit_b',# enhance_mask_sam_model
                0.25,   # enhance_mask_text_threshold 
                0.3,    # enhance_mask_box_threshold
                0,      # enhance_mask_sam_max_detections
                False,  # enhance_inpaint_disable_initial_latent
                'v2.6', # enhance_inpaint_engine
                1.0,    # enhance_inpaint_strength
                0.618,  # enhance_inpaint_respective_field
                0,      # enhance_inpaint_erode_or_dilate
                False   # enhance_mask_invert
            ]
            task_args.extend(tab_args)
            
        return task_args
        
    def generate(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        prompt: str,
        negative_prompt: str = "",
        inpaint_engine: str = "v1",
        inpaint_strength: float = 1.0,
        inpaint_respective_field: float = 0.618,
    ) -> List[str]:
        """Generate inpainted images using Fooocus engine"""
        try:
            image,empty_mask = self.load_image(image_path)
            mask = self.load_mask(mask_path)
            
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(
                    f"Image and mask dimensions do not match: "
                    f"image {image.shape[:2]} vs mask {mask.shape[:2]}"
                )
            
            task_args = self.prepare_inputs(
                image=image,
                mask=mask,
                empty_image=empty_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                inpaint_engine=inpaint_engine,
                inpaint_strength=inpaint_strength,
                inpaint_respective_field=inpaint_respective_field
            )
            
            task = worker.AsyncTask(args=task_args)
            self.current_task = task
            worker.async_tasks.append(task)
            finished =False
            results = []
            while not finished:
                time.sleep(0.01)
                if len(task.yields) > 0:
                    flag, product = task.yields.pop(0)
                    if flag == 'finish':
                        results = product
                        finished = True
                        break
            
            if args_manager.args.disable_image_log:
                for filepath in results:
                    if isinstance(filepath, str) and os.path.exists(filepath):
                        os.remove(filepath)
            first_image_path = results[0]
            first_image = cv2.imread(first_image_path)

            return results, first_image
            
        except Exception as e:
            print(f"Error during inpainting: {str(e)}")
            raise

    def stop_generation(self):
        """Stop the current generation task"""
        if self.current_task and self.current_task.processing:
            import ldm_patched.modules.model_management as model_management
            self.current_task.last_stop = 'stop'
            model_management.interrupt_current_processing()
            
def main():
    pipeline = InpaintingPipeline()

    # Generate inpainted image
    results = pipeline.generate(
        image_path="test_sample_1.jpg",
        mask_path="inverted_mask.png",
        prompt="fireworks and a city skyline at midnight as the background, realistic  ",
        negative_prompt="worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3), anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        inpaint_engine="v1",
        inpaint_strength=1.0,
        inpaint_respective_field=0.618
    )

    # Results will contain paths to generated images
    print(f"Generated images saved to: {results}")

if __name__ == "__main__":
    main()