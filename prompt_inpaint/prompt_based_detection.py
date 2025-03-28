import cv2
import numpy as np
import torch
import torchvision

import supervision as sv

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


class GroundedSAMPipeline:
    def __init__(
        self,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_checkpoint_path: str,
        sam_encoder_version: str = "vit_h",
        device: torch.device = None,
    ):
        """
        Initializes the GroundedSAMPipeline class by loading:
          - GroundingDINO model
          - Segment Anything model (SAM)
        
        Args:
            grounding_dino_config_path: Path to the GroundingDINO config file.
            grounding_dino_checkpoint_path: Path to the GroundingDINO checkpoint file.
            sam_checkpoint_path: Path to the Segment Anything checkpoint file.
            sam_encoder_version: Model type for SAM (default is "vit_h").
            device: Torch device (cuda if available, else cpu).
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # 1. Build GroundingDINO Inference Model
        self.grounding_dino_model = Model(
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_checkpoint_path
        )

        # 2. Build SAM Model and SAM Predictor
        self.sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def detect_objects(
        self,
        image: np.ndarray,
        classes: list,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> sv.Detections:
        """
        Runs the GroundingDINO model on an image with a list of class prompts.
        """
        print(f"Running detection with classes: {classes}")
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        print(f"Detection results:")
        print(f"- xyxy shape: {detections.xyxy.shape if detections.xyxy is not None else None}")
        print(f"- confidence: {detections.confidence}")
        print(f"- class_id before: {detections.class_id}")
        
        # Ensure class_id is properly set (0 for single class detection)
        if detections.class_id is None or all(cid is None for cid in detections.class_id):
            detections.class_id = np.zeros(len(detections.xyxy), dtype=int)

        
        print(f"- class_id after: {detections.class_id}")
        return detections

    def apply_nms(
        self,
        detections: sv.Detections,
        nms_threshold: float
    ) -> sv.Detections:
        """
        Applies Non-Maximum Suppression (NMS) on detection bounding boxes.
        
        Args:
            detections: A Supervision Detections object.
            nms_threshold: IoU threshold for NMS.

        Returns:
            Updated Supervision Detections object with fewer boxes.
        """
        if len(detections.xyxy) == 0:
            return detections  # No detections to process

        keep_idxs = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[keep_idxs]
        detections.confidence = detections.confidence[keep_idxs]
        detections.class_id = detections.class_id[keep_idxs]
        if detections.mask is not None:
            detections.mask = detections.mask[keep_idxs]

        return detections

    def segment_masks(
        self,
        image: np.ndarray,
        detections: sv.Detections
    ) -> np.ndarray:
        """
        Uses Segment-Anything to generate masks for each bounding box in detections.

        Args:
            image: Input image (numpy array in BGR).
            detections: A Supervision Detections object with bounding boxes.

        Returns:
            A numpy array of shape (N, H, W), where N is the number of detections.
            Each entry is a boolean mask for that detection.
        """
        if len(detections.xyxy) == 0:
            return np.array([])

        # Convert from BGR to RGB for SAM
        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        result_masks = []
        for box in detections.xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            # Choose the mask with the highest predicted score
            best_mask_index = np.argmax(scores)
            result_masks.append(masks[best_mask_index])

        return np.array(result_masks)

    def create_merged_binary_mask(
        self,
        detection_masks: np.ndarray,
        image_shape: tuple
    ) -> np.ndarray:
        """
        Merges all instance masks into one binary mask (255 = foreground, 0 = background).

        Args:
            detection_masks: numpy array of shape (N, H, W) where each entry is a boolean mask.
            image_shape: (height, width) of the original image.

        Returns:
            A single-channel binary mask (uint8) with shape (H, W).
        """
        if len(detection_masks) == 0:
            # Return an empty (all-zero) mask if there are no detections
            return np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        # Initialize an all-zero mask
        merged_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        # For each detection mask, set foreground pixels to 255
        for mask in detection_masks:
            # mask is typically boolean; cast to uint8
            merged_mask[mask] = 255

        return merged_mask

    def annotate_image(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        class_list: list
    ) -> np.ndarray:
        """
        Annotate the image with bounding boxes and masks from the detections.

        Args:
            image: The original image (numpy array in BGR).
            detections: A Supervision Detections object containing bounding boxes, masks, etc.
            class_list: List of classes used in detection (e.g. ["ear rings"]).

        Returns:
            The annotated image as a numpy array (in BGR).
        """
        # Create labels for bounding boxes, handling case where class_id might be None
        if detections.class_id is None:
            labels = [f"{class_list[0]} {confidence:0.2f}" for confidence in detections.confidence]
        else:
            labels = [
                f"{class_list[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
        
        annotated_image = image.copy()

        # Annotate with masks (if present)
        if detections.mask is not None and len(detections.mask) > 0:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )
        # Annotate with boxes
        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        return annotated_image

    def run(
        self,
        source_image_path: str,
        classes: list,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.8,
        output_grounding_dino_annotated_path: str = None,
        output_grounded_sam_annotated_path: str = None,
        output_binary_mask_path: str = None
    ):
        """
        Full pipeline for detection and segmentation.
        """
        # 1. Load image
        image = cv2.imread(source_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {source_image_path}")

        # 2. Detect objects with GroundingDINO
        print("\nStarting object detection...")
        detections = self.detect_objects(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(f"\nInitial detections:")
        print(f"- Number of detections: {len(detections.xyxy)}")
        print(f"- Class IDs: {detections.class_id}")
        print(f"- Confidences: {detections.confidence}")

        # 3. NMS post-processing
        print(f"\nApplying NMS...")
        detections = self.apply_nms(detections=detections, nms_threshold=nms_threshold)
        print(f"After NMS:")
        print(f"- Number of detections: {len(detections.xyxy)}")
        print(f"- Class IDs: {detections.class_id}")
        print(f"- Confidences: {detections.confidence}")

        # 4. Segment each detected box using SAM
        masks = self.segment_masks(image=image, detections=detections)
        detections.mask = masks

        # 5. Merge all instance masks into a single binary mask
        (h, w) = image.shape[:2]
        merged_binary_mask = self.create_merged_binary_mask(detection_masks=masks, image_shape=(h, w))

        # Save merged binary mask if a path is provided
        if output_binary_mask_path is not None:
            cv2.imwrite(output_binary_mask_path, merged_binary_mask)
            print(f"Binary mask saved at {output_binary_mask_path}")

        # 6. Annotate final results (boxes + masks)
        annotated_image = self.annotate_image(
            image=image,
            detections=detections,
            class_list=classes
        )

        # Optionally save the final annotated image
        if output_grounded_sam_annotated_path is not None:
            cv2.imwrite(output_grounded_sam_annotated_path, annotated_image)
            print(f"Grounded SAM annotated image saved at {output_grounded_sam_annotated_path}")

        # Return annotated image, the merged binary mask, and detections
        return annotated_image, merged_binary_mask, detection