import os
import sys
import argparse
import json
import time
import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union, Any
import Levenshtein
import multiprocessing
import queue
from functools import lru_cache

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image
from tools.utils.logging import get_logger

logger = get_logger()

###############################################################################
# Utility Functions
###############################################################################

def parse_roi(roi_str: str) -> Optional[List[float]]:
    """
    Parse ROI string in format 'x1,y1,x2,y2' to [x1, y1, x2, y2].
    Values can be percentages (0-100) or ratios (0-1).
    """
    if not roi_str:
        return None
    try:
        vals = [float(v.strip()) for v in roi_str.split(',')]
        if len(vals) != 4:
            return None
        if max(vals) > 1:
            vals = [v / 100.0 for v in vals]
        return vals
    except ValueError:
        return None

def get_box_xywh(quad: Union[List[List[float]], np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Convert quad points to (x, y, width, height) format.
    """
    quad = np.array(quad)
    x_min, y_min = np.min(quad, axis=0)
    x_max, y_max = np.max(quad, axis=0)
    return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

def iou(boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union for two boxes in (x, y, w, h) format.
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    
    # Calculate intersection coordinates
    x_start = max(xA, xB)
    y_start = max(yA, yB)
    x_end = min(xA + wA, xB + wB)
    y_end = min(yA + hA, yB + hB)
    
    # Calculate area of intersection
    inter = max(0, x_end - x_start) * max(0, y_end - y_start)
    
    # Calculate area of both boxes
    areaA = wA * hA
    areaB = wB * hB
    
    # Calculate IoU
    return inter / float(areaA + areaB - inter + 1e-6)

@lru_cache(maxsize=128)
def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity with caching for performance.
    """
    # Remove spaces for better comparison
    clean_text1 = text1.replace(" ", "").lower()
    clean_text2 = text2.replace(" ", "").lower()
    
    return Levenshtein.ratio(clean_text1, clean_text2)

def is_box_in_roi(box: np.ndarray, roi: List[float], frame_shape: Tuple[int, int]) -> bool:
    """
    Check if a box is completely inside the ROI.
    box: np.ndarray of shape (4, 2) with coordinates
    roi: [x1_ratio, y1_ratio, x2_ratio, y2_ratio]
    """
    if roi is None:
        return True
        
    h, w = frame_shape
    roi_x1 = int(roi[0] * w)
    roi_y1 = int(roi[1] * h)
    roi_x2 = int(roi[2] * w)
    roi_y2 = int(roi[3] * h)
    
    # Get box bounds
    x_min = np.min(box[:, 0])
    y_min = np.min(box[:, 1])
    x_max = np.max(box[:, 0])
    y_max = np.max(box[:, 1])
    
    # Check if box is inside ROI
    return (x_min >= roi_x1 and x_max <= roi_x2 and 
            y_min >= roi_y1 and y_max <= roi_y2)

def sorted_boxes(dt_boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sort boxes from top to bottom, then left to right.
    """
    if dt_boxes is None or len(dt_boxes) == 0:
        return []
    
    # Sort by y-coordinate first
    boxes = sorted(dt_boxes, key=lambda box: np.min(box[:, 1]))
    
    # Group boxes by similar y-coordinate
    groups = []
    current_group = [boxes[0]]
    current_y = np.min(boxes[0][:, 1])
    
    for box in boxes[1:]:
        box_y = np.min(box[:, 1])
        
        # If y is similar to current group, add to group
        if abs(box_y - current_y) < 10:
            current_group.append(box)
        else:
            # Sort current group by x-coordinate
            groups.append(sorted(current_group, key=lambda b: np.min(b[:, 0])))
            # Start new group
            current_group = [box]
            current_y = box_y
    
    # Add last group
    if current_group:
        groups.append(sorted(current_group, key=lambda b: np.min(b[:, 0])))
    
    # Flatten groups
    return [box for group in groups for box in group]

###############################################################################
# Image Processing Functions
###############################################################################

def blur_and_reduce_contrast(image, kernel_size=15, sigmaX=0, contrast_alpha=0.3):
    """
    Blur and reduce contrast to handle faint background text that might
    confuse the detection model.
    """
    # Apply Gaussian blur to reduce noise and background text
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)
    
    # Reduce contrast to make faint text less visible
    processed_image = cv2.convertScaleAbs(blurred_image, alpha=contrast_alpha, beta=0)
    
    return processed_image

def get_text_crop(image: np.ndarray, box: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """
    Extract a rectangular crop around text with padding.
    """
    # Get bounding rectangle
    x_min = max(0, int(np.min(box[:, 0])))
    y_min = max(0, int(np.min(box[:, 1])))
    x_max = min(image.shape[1], int(np.max(box[:, 0])))
    y_max = min(image.shape[0], int(np.max(box[:, 1])))
    
    # Add padding
    h = y_max - y_min
    w = x_max - x_min
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image.shape[1], x_max + pad_x)
    y_max = min(image.shape[0], y_max + pad_y)
    
    # Crop image
    return image[y_min:y_max, x_min:x_max].copy()

def enhance_text_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance text image for better recognition.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Sharpen image
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Enhance contrast
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=10)
    
    # Convert back to color if input was color
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced

def filter_boxes_by_size(boxes: List[np.ndarray],
                         img_shape: Tuple[int, int],
                         config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    """
    Filter boxes based on relative size constraints.
    """
    if config is None:
        config = {
            'min_w_ratio': 0.01,
            'min_h_ratio': 0.01,
            'max_w_ratio': 0.9,
            'max_h_ratio': 0.15  # Subtitles are usually not very tall
        }
    
    min_w_ratio = config.get('min_w_ratio', 0.01)
    min_h_ratio = config.get('min_h_ratio', 0.01)
    max_w_ratio = config.get('max_w_ratio', 0.9)
    max_h_ratio = config.get('max_h_ratio', 0.15)

    filtered_boxes = []
    img_h, img_w = img_shape
    
    for box in boxes:
        x, y, bw, bh = get_box_xywh(box)
        w_ratio = bw / img_w
        h_ratio = bh / img_h
        
        if (min_w_ratio <= w_ratio <= max_w_ratio and 
            min_h_ratio <= h_ratio <= max_h_ratio):
            filtered_boxes.append(box)
            
    return filtered_boxes

def same_line_merge(
    dt_boxes: List[np.ndarray],
    line_y_thresh_ratio: float = 0.5,
    line_x_gap_ratio: float = 0.3
) -> List[np.ndarray]:
    """
    Merge boxes that are on the same line into a single box.
    """
    if not dt_boxes:
        return []
    
    # Sort boxes by y-coordinate
    boxes_sorted = sorted(dt_boxes, key=lambda box: np.min(box[:, 1]))
    lines = []  # Each element is a list of boxes on the same line
    
    for box in boxes_sorted:
        center_y = (np.min(box[:, 1]) + np.max(box[:, 1])) / 2.0
        height = np.max(box[:, 1]) - np.min(box[:, 1])
        added = False
        
        # Try to add to existing line
        for line in lines:
            # Calculate average center and height of current line
            line_centers = [(np.min(b[:, 1]) + np.max(b[:, 1])) / 2.0 for b in line]
            avg_center = sum(line_centers) / len(line_centers)
            line_heights = [np.max(b[:, 1]) - np.min(b[:, 1]) for b in line]
            avg_height = sum(line_heights) / len(line_heights)
            
            # Check if box is on same line based on y-coordinate
            if abs(center_y - avg_center) <= line_y_thresh_ratio * avg_height:
                # Check x-gap with last box in line
                sorted_line = sorted(line, key=lambda b: np.min(b[:, 0]))
                last_box = sorted_line[-1]
                gap = np.min(box[:, 0]) - np.max(last_box[:, 0])
                
                line_widths = [np.max(b[:, 0]) - np.min(b[:, 0]) for b in line]
                avg_width = sum(line_widths) / len(line_widths)
                
                # Add to line if gap is small enough
                if gap < line_x_gap_ratio * avg_width:
                    line.append(box)
                    added = True
                    break
        
        # Create new line if not added to existing one
        if not added:
            lines.append([box])
    
    # Create bounding box for each line
    merged_boxes = []
    for line in lines:
        # Find min/max coordinates
        min_x = min(np.min(b[:, 0]) for b in line)
        min_y = min(np.min(b[:, 1]) for b in line)
        max_x = max(np.max(b[:, 0]) for b in line)
        max_y = max(np.max(b[:, 1]) for b in line)
        
        # Create merged box
        merged_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype=np.float32)
        
        merged_boxes.append(merged_box)
    
    return merged_boxes

###############################################################################
# OpenOCR Class
###############################################################################

class OpenOCR:
    def __init__(self,
                 cfg_det_path: str,
                 cfg_rec_path: str,
                 drop_score: float = 0.5,
                 det_box_type: str = 'quad',
                 det_batch_size: int = 1,
                 rec_batch_size: int = 6):
        """
        Initialize OCR engine with detection and recognition models.
        """
        cfg_det = Config(cfg_det_path).cfg
        cfg_rec = Config(cfg_rec_path).cfg

        self.text_detector = OpenDetector(cfg_det)
        self.text_recognizer = OpenRecognizer(cfg_rec)

        self.det_box_type = det_box_type
        self.drop_score = drop_score
        self.det_batch_size = det_batch_size
        self.rec_batch_size = rec_batch_size

    def infer_batch_image_det(self,
                              img_numpy_list: List[np.ndarray]
                              ) -> Tuple[List[List[np.ndarray]], List[Dict[str, float]]]:
        """
        Run text detection on a batch of images.
        """
        all_dt_boxes = []
        all_time_dicts = []
        
        # Process in batches
        for i in range(0, len(img_numpy_list), self.det_batch_size):
            batch_imgs = img_numpy_list[i : i + self.det_batch_size]
            
            # Run detection
            batch_results = self.text_detector(img_numpy_list=batch_imgs)
            
            # Process results
            for det_res in batch_results:
                dt_boxes = det_res.get('boxes', [])
                elapse = det_res.get('elapse', 0.0)
                time_dict = {'detection_time': elapse}
                
                # Sort boxes if any
                if dt_boxes is not None and len(dt_boxes) > 0:
                    dt_boxes = sorted_boxes(dt_boxes)
                else:
                    dt_boxes = []
                    
                all_dt_boxes.append(dt_boxes)
                all_time_dicts.append(time_dict)
                
        return all_dt_boxes, all_time_dicts

    def infer_batch_image_rec(self,
                              img_list: List[Union[np.ndarray, Image.Image]]
                              ) -> Tuple[List[List[Union[str, float, float]]], float]:
        """
        Run text recognition on a batch of cropped images.
        """
        rec_res_full = []
        total_rec_time = 0.0
        
        # Convert all images to PIL format
        pil_images = []
        for img in img_list:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        # Process in batches
        for i in range(0, len(pil_images), self.rec_batch_size):
            batch_imgs = pil_images[i : i + self.rec_batch_size]
            
            # Run recognition
            batch_rec = self.text_recognizer(img_numpy_list=batch_imgs)
            
            # Process results
            for r in batch_rec:
                text = r.get('text', '')
                score = r.get('score', 0.0)
                elapse = r.get('elapse', 0.0)
                total_rec_time += elapse
                rec_res_full.append([text, score, elapse])
                
        return rec_res_full, total_rec_time



# def process_images(
#     ocr_engine: OpenOCR,
#     image_list: List[Union[str, np.ndarray]],
#     roi: Optional[List[float]] = None,
#     line_y_thresh: float = 0.5,
#     line_x_gap: float = 0.3,
#     do_merge: bool = True,
#     filter_config: Optional[Dict[str, Any]] = None,
#     apply_blur_contrast: bool = True,
#     resize_factor: float = 1.0,
#     debug_det_dir: Optional[str] = None,
#     debug_box_dir: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Process a list of images for OCR, returning text results in the same order as input images.
    
#     Args:
#         ocr_engine: OCR engine instance
#         image_list: List of images as file paths, numpy arrays, or PIL Images
#         roi: Optional region of interest [x1, y1, x2, y2] as ratios
#         line_y_thresh: Threshold for merging lines vertically
#         line_x_gap: Maximum gap ratio for merging text horizontally 
#         do_merge: Whether to merge adjacent text boxes
#         filter_config: Configuration for filtering text boxes by size
#         apply_blur_contrast: Whether to apply blur and contrast adjustment to handle background text
#         resize_factor: Factor to resize images (e.g., 0.5 for half size)
#         debug_det_dir: Directory to save detection visualization
#         debug_box_dir: Directory to save text box crops
        
#     Returns:
#         List of dictionaries containing detected text for each image, preserving input order
#     """
#     import os
#     import time
    
#     if not image_list:
#         logger.warning("No images provided for processing")
#         return []
    
#     t_start = time.time()
#     total_images = len(image_list)
#     logger.info(f"Processing {total_images} images")
    
#     # Create debug directories if needed
#     if debug_det_dir:
#         os.makedirs(debug_det_dir, exist_ok=True)
#     if debug_box_dir:
#         os.makedirs(debug_box_dir, exist_ok=True)
    
#     # Load and preprocess images
#     t_load = time.time()
#     processed_images = []
#     original_images = []
    
#     for i, img_src in enumerate(image_list):
#         # Load image from various sources
#         if isinstance(img_src, str):
#             if os.path.exists(img_src):
#                 img = cv2.imread(img_src)
#                 if img is None:
#                     logger.warning(f"Failed to load image {i}: {img_src}")
#                     # Add placeholder to maintain order
#                     processed_images.append(None)
#                     original_images.append(None)
#                     continue
#             else:
#                 logger.warning(f"Image file not found: {img_src}")
#                 processed_images.append(None)
#                 original_images.append(None)
#                 continue
#         elif isinstance(img_src, np.ndarray):
#             img = img_src
#         else:
#             logger.warning(f"Unsupported image type for image {i}: {type(img_src)}")
#             processed_images.append(None)
#             original_images.append(None)
#             continue
        
#         # Resize if needed
#         if resize_factor != 1.0:
#             h, w = img.shape[:2]
#             new_w, new_h = int(w * resize_factor), int(h * resize_factor)
#             img = cv2.resize(img, (new_w, new_h))
        
#         # Store original image
#         original_images.append(img.copy())
        
#         # Apply blur and contrast adjustment if enabled
#         if apply_blur_contrast:
#             img = blur_and_reduce_contrast(img, kernel_size=15, sigmaX=0, contrast_alpha=0.3)
        
#         processed_images.append(img)
    
#     logger.info(f"Loaded and preprocessed {len(processed_images)} images in {time.time() - t_load:.2f}s")
    
#     # Filter out None values while keeping track of indices
#     valid_processed_images = []
#     valid_indices = []
    
#     for i, img in enumerate(processed_images):
#         if img is not None:
#             valid_processed_images.append(img)
#             valid_indices.append(i)
    
#     if not valid_processed_images:
#         logger.warning("No valid images to process")
#         return [{} for _ in range(total_images)]
    
#     # Run text detection in batch
#     t_det = time.time()
#     all_dt_boxes, _ = ocr_engine.infer_batch_image_det(valid_processed_images)
#     logger.info(f"Detection completed in {time.time() - t_det:.2f}s")
    
#     # Process results for each image
#     all_results = [{} for _ in range(total_images)]  # Initialize with empty dict to maintain indices
    
#     for i, (img_idx, dt_boxes) in enumerate(zip(valid_indices, all_dt_boxes)):
#         # Get original image for this index
#         orig_img = original_images[img_idx]
        
#         # Skip if no boxes detected
#         if dt_boxes is None or len(dt_boxes) == 0:
#             all_results[img_idx] = {"texts": []}
#             continue
        
#         # Filter boxes by ROI and size
#         filtered_boxes = []
#         for box in dt_boxes:
#             # Apply size filtering
#             if filter_config:
#                 x, y, bw, bh = get_box_xywh(box)
#                 h, w = orig_img.shape[:2]
#                 w_ratio = bw / w
#                 h_ratio = bh / h
                
#                 min_w_ratio = filter_config.get('min_w_ratio', 0.01)
#                 min_h_ratio = filter_config.get('min_h_ratio', 0.01)
#                 max_w_ratio = filter_config.get('max_w_ratio', 0.9)
#                 max_h_ratio = filter_config.get('max_h_ratio', 0.15)
                
#                 if not (min_w_ratio <= w_ratio <= max_w_ratio and 
#                         min_h_ratio <= h_ratio <= max_h_ratio):
#                     continue
            
#             # Check if box is within ROI
#             if roi is None or is_box_in_roi(box, roi, orig_img.shape[:2]):
#                 filtered_boxes.append(box)
        
#         # Save detection debug image if requested
#         if debug_det_dir:
#             debug_img = orig_img.copy()
#             for box in filtered_boxes:
#                 poly = np.array(box, dtype=np.int32)
#                 cv2.polylines(debug_img, [poly], isClosed=True, color=(0,255,0), thickness=2)
            
#             # Draw ROI if specified
#             if roi:
#                 h, w = debug_img.shape[:2]
#                 roi_x1, roi_y1 = int(roi[0] * w), int(roi[1] * h)
#                 roi_x2, roi_y2 = int(roi[2] * w), int(roi[3] * h)
#                 cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
            
#             out_path = os.path.join(debug_det_dir, f"image_{img_idx:04d}_det.jpg")
#             cv2.imwrite(out_path, debug_img)
        
#         # Merge boxes into lines if requested
#         if do_merge:
#             process_boxes = same_line_merge(
#                 filtered_boxes, 
#                 line_y_thresh_ratio=line_y_thresh, 
#                 line_x_gap_ratio=line_x_gap
#             )
            
#             # Save merged box debug image if requested
#             if debug_det_dir:
#                 merge_debug_img = orig_img.copy()
#                 # Draw original boxes in green
#                 for box in filtered_boxes:
#                     poly = np.array(box, dtype=np.int32)
#                     cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(0,255,0), thickness=1)
                
#                 # Draw merged lines in blue with thicker lines
#                 for merged_box in process_boxes:
#                     poly = np.array(merged_box, dtype=np.int32)
#                     cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(255,0,0), thickness=2)
                
#                 out_path = os.path.join(debug_det_dir, f"image_{img_idx:04d}_merged.jpg")
#                 cv2.imwrite(out_path, merge_debug_img)
#         else:
#             process_boxes = filtered_boxes
        
#         # Create crops for recognition
#         crops = []
#         crop_boxes = []
        
#         for j, box in enumerate(process_boxes):
#             # Get tight crop around text
#             text_crop = get_text_crop(orig_img, box)
            
#             # Save debug image if needed
#             if debug_box_dir:
#                 debug_path = os.path.join(debug_box_dir, f"image_{img_idx:04d}_box_{j:03d}.jpg")
#                 cv2.imwrite(debug_path, text_crop)
            
#             # Enhance the crop for recognition
#             enhanced_crop = enhance_text_image(text_crop)
            
#             # Add to crops list
#             crops.append(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
#             crop_boxes.append(box)
        
#         # Skip if no crops
#         if not crops:
#             all_results[img_idx] = {"texts": []}
#             continue
        
#         # Perform text recognition
#         # t_rec = time.time()
#         rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
#         # logger.debug(f"Recognition for image {img_idx} completed in {time.time() - t_rec:.2f}s")
        
#         # Collect recognition results
#         texts = []
#         for j, (box, rec_result) in enumerate(zip(crop_boxes, rec_res_all)):
#             text, score, _ = rec_result
            
#             # Only keep results with high score and non-empty text
#             if score >= ocr_engine.drop_score and text.strip():
#                 texts.append({
#                     "text": text,
#                     "score": float(score),
#                     "box": get_box_xywh(box)
#                 })
        
#         # Sort texts from top to bottom
#         texts = sorted(texts, key=lambda x: x["box"][1])
        
#         # Save result for this image
#         all_results[img_idx] = {"texts": texts}
    
#     # Ensure all images have results
#     for i in range(total_images):
#         if i not in valid_indices:
#             all_results[i] = {"texts": []}
    
#     logger.info(f"Processed {total_images} images in {time.time() - t_start:.2f}s")
#     return all_results


def process_images(
    ocr_engine: OpenOCR,
    image_list: List[Union[str, np.ndarray]],
    roi: Optional[List[float]] = None,
    line_y_thresh: float = 0.5,
    line_x_gap: float = 0.3,
    do_merge: bool = True,
    filter_config: Optional[Dict[str, Any]] = None,
    apply_blur_contrast: bool = True,
    resize_factor: float = 1.0,
    debug_det_dir: Optional[str] = None,
    debug_box_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process a list of images for OCR, returning text results in the same order as input images.
    Uses batch processing for recognition to maximize efficiency.
    
    Args:
        ocr_engine: OCR engine instance
        image_list: List of images as file paths, numpy arrays, or PIL Images
        roi: Optional region of interest [x1, y1, x2, y2] as ratios
        line_y_thresh: Threshold for merging lines vertically
        line_x_gap: Maximum gap ratio for merging text horizontally 
        do_merge: Whether to merge adjacent text boxes
        filter_config: Configuration for filtering text boxes by size
        apply_blur_contrast: Whether to apply blur and contrast adjustment to handle background text
        resize_factor: Factor to resize images (e.g., 0.5 for half size)
        debug_det_dir: Directory to save detection visualization
        debug_box_dir: Directory to save text box crops
        
    Returns:
        List of dictionaries containing detected text for each image, preserving input order
    """
    import os
    import time
    
    if not image_list:
        logger.warning("No images provided for processing")
        return []
    
    t_start = time.time()
    total_images = len(image_list)
    logger.info(f"Processing {total_images} images")
    
    # Create debug directories if needed
    if debug_det_dir:
        os.makedirs(debug_det_dir, exist_ok=True)
    if debug_box_dir:
        os.makedirs(debug_box_dir, exist_ok=True)
    
    # Load and preprocess images
    t_load = time.time()
    processed_images = []
    original_images = []
    
    for i, img_src in enumerate(image_list):
        # Load image from various sources
        if isinstance(img_src, str):
            if os.path.exists(img_src):
                img = cv2.imread(img_src)
                if img is None:
                    logger.warning(f"Failed to load image {i}: {img_src}")
                    # Add placeholder to maintain order
                    processed_images.append(None)
                    original_images.append(None)
                    continue
            else:
                logger.warning(f"Image file not found: {img_src}")
                processed_images.append(None)
                original_images.append(None)
                continue
        elif isinstance(img_src, np.ndarray):
            img = img_src
        else:
            logger.warning(f"Unsupported image type for image {i}: {type(img_src)}")
            processed_images.append(None)
            original_images.append(None)
            continue
        
        # Resize if needed
        if resize_factor != 1.0:
            h, w = img.shape[:2]
            new_w, new_h = int(w * resize_factor), int(h * resize_factor)
            img = cv2.resize(img, (new_w, new_h))
        
        # Store original image
        original_images.append(img.copy())
        
        # Apply blur and contrast adjustment if enabled
        if apply_blur_contrast:
            img = blur_and_reduce_contrast(img, kernel_size=15, sigmaX=0, contrast_alpha=0.3)
        
        processed_images.append(img)
    
    logger.info(f"Loaded and preprocessed {len(processed_images)} images in {time.time() - t_load:.2f}s")
    
    # Filter out None values while keeping track of indices
    valid_processed_images = []
    valid_indices = []
    
    for i, img in enumerate(processed_images):
        if img is not None:
            valid_processed_images.append(img)
            valid_indices.append(i)
    
    if not valid_processed_images:
        logger.warning("No valid images to process")
        return [{"texts": []} for _ in range(total_images)]
    
    # Run text detection in batch
    t_det = time.time()
    all_dt_boxes, _ = ocr_engine.infer_batch_image_det(valid_processed_images)
    logger.info(f"Detection completed in {time.time() - t_det:.2f}s")
    
    # Initialize storage for crops and tracking information
    all_crops = []  # Store all crops from all images
    crop_to_image_map = []  # Maps crop index to original image index
    crop_to_box_map = []  # Maps crop index to box index within the image
    box_lists = []  # Stores boxes for each image
    
    # Process each image to prepare crops for a single batch recognition
    for idx, (img_idx, dt_boxes) in enumerate(zip(valid_indices, all_dt_boxes)):
        orig_img = original_images[img_idx]
        
        # Skip if no boxes detected
        if dt_boxes is None or len(dt_boxes) == 0:
            box_lists.append([])
            continue
        
        # Filter boxes by ROI and size
        filtered_boxes = []
        for box in dt_boxes:
            # Apply size filtering
            if filter_config:
                x, y, bw, bh = get_box_xywh(box)
                h, w = orig_img.shape[:2]
                w_ratio = bw / w
                h_ratio = bh / h
                
                min_w_ratio = filter_config.get('min_w_ratio', 0.01)
                min_h_ratio = filter_config.get('min_h_ratio', 0.01)
                max_w_ratio = filter_config.get('max_w_ratio', 0.9)
                max_h_ratio = filter_config.get('max_h_ratio', 0.15)
                
                if not (min_w_ratio <= w_ratio <= max_w_ratio and 
                        min_h_ratio <= h_ratio <= max_h_ratio):
                    continue
            
            # Check if box is within ROI
            if roi is None or is_box_in_roi(box, roi, orig_img.shape[:2]):
                filtered_boxes.append(box)
        
        # Save detection debug image if requested
        if debug_det_dir:
            debug_img = orig_img.copy()
            for box in filtered_boxes:
                poly = np.array(box, dtype=np.int32)
                cv2.polylines(debug_img, [poly], isClosed=True, color=(0,255,0), thickness=2)
            
            # Draw ROI if specified
            if roi:
                h, w = debug_img.shape[:2]
                roi_x1, roi_y1 = int(roi[0] * w), int(roi[1] * h)
                roi_x2, roi_y2 = int(roi[2] * w), int(roi[3] * h)
                cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
            
            out_path = os.path.join(debug_det_dir, f"image_{img_idx:04d}_det.jpg")
            cv2.imwrite(out_path, debug_img)
        
        # Merge boxes into lines if requested
        if do_merge:
            process_boxes = same_line_merge(
                filtered_boxes, 
                line_y_thresh_ratio=line_y_thresh, 
                line_x_gap_ratio=line_x_gap
            )
            
            # Save merged box debug image if requested
            if debug_det_dir:
                merge_debug_img = orig_img.copy()
                # Draw original boxes in green
                for box in filtered_boxes:
                    poly = np.array(box, dtype=np.int32)
                    cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(0,255,0), thickness=1)
                
                # Draw merged lines in blue with thicker lines
                for merged_box in process_boxes:
                    poly = np.array(merged_box, dtype=np.int32)
                    cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(255,0,0), thickness=2)
                
                out_path = os.path.join(debug_det_dir, f"image_{img_idx:04d}_merged.jpg")
                cv2.imwrite(out_path, merge_debug_img)
        else:
            process_boxes = filtered_boxes
        
        # Store process_boxes for this image
        box_lists.append(process_boxes)
        
        # Create crops for each box in this image
        for box_idx, box in enumerate(process_boxes):
            # Get tight crop around text
            text_crop = get_text_crop(orig_img, box)
            
            # Save debug image if needed
            if debug_box_dir:
                debug_path = os.path.join(debug_box_dir, f"image_{img_idx:04d}_box_{box_idx:03d}.jpg")
                cv2.imwrite(debug_path, text_crop)
            
            # Enhance the crop for recognition
            enhanced_crop = enhance_text_image(text_crop)
            
            # Add to crops list and track mapping
            all_crops.append(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
            crop_to_image_map.append(img_idx)
            crop_to_box_map.append(box_idx)
    
    # Run batch recognition on all crops at once
    t_rec = time.time()
    if all_crops:
        rec_res_all, _ = ocr_engine.infer_batch_image_rec(all_crops)
        logger.info(f"Recognition completed in {time.time() - t_rec:.2f}s for {len(all_crops)} text regions")
    else:
        rec_res_all = []
        logger.info("No text regions found for recognition")
    
    # Initialize results for all images
    all_results = [{} for _ in range(total_images)]
    
    # Group recognition results by original image
    image_text_results = {img_idx: [] for img_idx in valid_indices}
    
    # Process recognition results and map back to original images
    for crop_idx, (rec_result, img_idx, box_idx) in enumerate(zip(rec_res_all, crop_to_image_map, crop_to_box_map)):
        text, score, _ = rec_result
        
        # Only keep results with high score and non-empty text
        if score >= ocr_engine.drop_score and text.strip():
            # Get corresponding box
            box = box_lists[valid_indices.index(img_idx)][box_idx]
            
            # Add to image results
            image_text_results[img_idx].append({
                "text": text,
                "score": float(score),
                "box": get_box_xywh(box)
            })
    
    # Sort text results by vertical position and assign to final results
    for img_idx, texts in image_text_results.items():
        # Sort texts from top to bottom
        sorted_texts = sorted(texts, key=lambda x: x["box"][1])
        all_results[img_idx] = {"texts": sorted_texts}
    
    # Ensure all images have results
    for i in range(total_images):
        if i not in valid_indices or "texts" not in all_results[i]:
            all_results[i] = {"texts": []}
    
    logger.info(f"Processed {total_images} images in {time.time() - t_start:.2f}s")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Image OCR for text extraction')
    # Core file parameters
    parser.add_argument('--image_dir', type=str, help='Directory containing images to process')
    parser.add_argument('--image_paths', type=str, help='Comma-separated list of image paths')
    parser.add_argument('--output_json', type=str, default='output_results.json', help='Path to save JSON results')
    
    # OCR model parameters
    parser.add_argument('--cfg_det_path', type=str, default="configs/det/dbnet/repvit_db.yml")
    parser.add_argument('--cfg_rec_path', type=str, default="configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml")
    parser.add_argument('--drop_score', type=float, default=0.9, help='Minimum confidence score for text detection')
    parser.add_argument('--det_batch_size', type=int, default=8)
    parser.add_argument('--rec_batch_size', type=int, default=8)
    
    # Processing parameters
    parser.add_argument('--line_y_thresh', type=float, default=0.5)
    parser.add_argument('--line_x_gap', type=float, default=0.3)
    parser.add_argument('--do_merge', action='store_true', default=True, help='Merge nearby text boxes')
    parser.add_argument('--no_merge', dest='do_merge', action='store_false', help='Do not merge text boxes')
    parser.add_argument('--roi', type=str, default="0.1,0.6,0.9,0.97", help='ROI format: x1,y1,x2,y2')
    
    # Image pre-processing parameters
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Resize factor for images (e.g., 0.5 for half size)')
    parser.add_argument('--apply_blur_contrast', action='store_true', default=True, help='Apply blur and contrast adjustment')
    parser.add_argument('--no_blur_contrast', dest='apply_blur_contrast', action='store_false', help='Disable blur and contrast adjustment')
    
    # Box filtering parameters
    parser.add_argument('--min_w_ratio', type=float, default=0.02)
    parser.add_argument('--min_h_ratio', type=float, default=0.02)
    parser.add_argument('--max_w_ratio', type=float, default=0.9)
    parser.add_argument('--max_h_ratio', type=float, default=0.09)
    
    # Debug parameters
    parser.add_argument('--debug_det_dir', type=str, default="det_dir", help='Directory to save detection visualization')
    parser.add_argument('--debug_box_dir', type=str, default=None, help='Directory to save text boxes')
    
    args = parser.parse_args()

    # Get list of image paths
    image_paths = []
    if args.image_dir:
        # Get all image files from directory
        if os.path.isdir(args.image_dir):
            for root, _, files in os.walk(args.image_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()  # Sort for deterministic order
        else:
            logger.error(f"Image directory not found: {args.image_dir}")
            return
    elif args.image_paths:
        # Parse comma-separated list
        image_paths = [path.strip() for path in args.image_paths.split(',')]
    else:
        logger.error("Either --image_dir or --image_paths must be specified")
        return
    
    if not image_paths:
        logger.error("No images found to process")
        return
    
    # Parse ROI if provided
    roi = parse_roi(args.roi) if args.roi else None
    
    # Box filtering configuration
    filter_config = {
        'min_w_ratio': args.min_w_ratio,
        'min_h_ratio': args.min_h_ratio,
        'max_w_ratio': args.max_w_ratio,
        'max_h_ratio': args.max_h_ratio
    }
    
    # Initialize OCR engine
    logger.info("Initializing OCR engine...")
    ocr_engine = OpenOCR(
        cfg_det_path=args.cfg_det_path,
        cfg_rec_path=args.cfg_rec_path,
        drop_score=args.drop_score,
        det_batch_size=args.det_batch_size,
        rec_batch_size=args.rec_batch_size
    )
    
    # Process images
    logger.info(f"Processing {len(image_paths)} images...")
    time_start = time.time()
    
    results = process_images(
        ocr_engine=ocr_engine,
        image_list=image_paths,
        roi=roi,
        line_y_thresh=args.line_y_thresh,
        line_x_gap=args.line_x_gap,
        do_merge=args.do_merge,
        filter_config=filter_config,
        apply_blur_contrast=args.apply_blur_contrast,
        resize_factor=args.resize_factor,
        debug_det_dir=args.debug_det_dir,
        debug_box_dir=args.debug_box_dir
    )
    
    # Add filenames to results
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        result['filename'] = os.path.basename(image_path)
    
    # Save results to JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print processing summary
    total_time = time.time() - time_start
    total_images = len(image_paths)
    total_text_boxes = sum(len(result.get("texts", [])) for result in results)
    images_with_text = sum(1 for result in results if result.get("texts", []))
    
    logger.info("=" * 50)
    logger.info("Processing Summary:")
    logger.info(f"- Total processing time: {total_time:.2f} seconds")
    logger.info(f"- Images processed: {total_images}")
    logger.info(f"- Images with text: {images_with_text} ({images_with_text/total_images*100:.1f}%)")
    logger.info(f"- Total text boxes detected: {total_text_boxes}")
    logger.info(f"- Results saved to: {args.output_json}")
    logger.info("=" * 50)
    
    # Print example of first few results
    logger.info("Sample Results:")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        if i >= len(results):
            break
        texts = result.get("texts", [])
        filename = result.get("filename", f"image_{i}")
        logger.info(f"- {filename}: {len(texts)} text regions")
        for j, text_item in enumerate(texts[:3]):  # Show first 3 text items
            if j >= len(texts):
                break
            logger.info(f"  * \"{text_item['text']}\" (score: {text_item['score']:.2f})")
        if len(texts) > 3:
            logger.info(f"  * ... and {len(texts) - 3} more")
    
    if len(results) > 3:
        logger.info(f"... and {len(results) - 3} more images")

if __name__ == '__main__':
    main()