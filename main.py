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

###############################################################################
# Video Processing Functions
###############################################################################

def read_video_segment(video_path: str, start_frame: int, end_frame: int,
                       skip_interval: int, fps: float, output_queue: multiprocessing.Queue):
    """
    Read a segment of video frames and put them in the output queue.
    Simplified to just read original frames without cropping.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            output_queue.put(("ERROR", f"Cannot open video segment {start_frame}-{end_frame}"))
            return
            
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames according to interval
            if (current_frame % skip_interval) != 0:
                current_frame += 1
                continue
                
            # Calculate current time
            current_sec = current_frame / fps
            
            # Send data to queue (just the original frame)
            output_queue.put((current_frame, frame, current_sec))
            current_frame += 1
            
    except Exception as e:
        output_queue.put(("ERROR", f"Exception in segment {start_frame}-{end_frame}: {str(e)}"))
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        output_queue.put(None)  # Signal segment completion

def track_and_deduplicate_text(
    video_results: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    text_sim_threshold: float = 0.95,
    min_display_time: float = 0.5,  # Minimum display time for single-frame text
    debug_logging: bool = False
):
    """
    Track text across frames and create continuous subtitle segments with start and end times,
    while preserving frame grouping (texts in the same frame share timing).
    """
    if not video_results or len(video_results) <= 1:
        return
    
    if debug_logging:
        logger.info(f"Starting subtitle tracking with {len(video_results)} frames")
    
    # Sort by timestamp
    video_results.sort(key=lambda x: x["timestamp"])
    
    # Track frame groups instead of individual text boxes
    active_frame_groups = {}  # {group_id: group_data}
    next_group_id = 0
    
    # Final list of subtitle segments
    subtitle_segments = []
    
    # Process each frame chronologically
    for frame_idx, frame in enumerate(video_results):
        timestamp = frame["timestamp"]
        texts_in_frame = frame.get("texts", [])
        
        if debug_logging:
            logger.info(f"Processing frame {frame_idx} at time {timestamp:.2f} with {len(texts_in_frame)} texts")
        
        # Skip empty frames
        if not texts_in_frame:
            continue
            
        # Check if this frame matches any active group
        matched_group_id = None
        best_match_score = 0
        
        for group_id, group in active_frame_groups.items():
            # Frame matching requires matching text boxes
            box_matches = 0
            total_boxes = max(len(texts_in_frame), len(group["texts"]))
            
            # Try to match each text in this frame to the group's texts
            for text_item in texts_in_frame:
                text = text_item["text"]
                box = text_item["box"]
                
                # Find best match for this text in the group
                for group_text in group["texts"]:
                    group_text_str = group_text["text"]
                    group_box = group_text["box"]
                    
                    # Check box overlap
                    iou_val = iou(group_box, box)
                    
                    # Check text similarity
                    exact_match = group_text_str == text
                    sim_val = 1.0 if exact_match else calculate_text_similarity(group_text_str, text)
                    
                    # If this text matches one in the group
                    if (exact_match and iou_val > iou_threshold * 0.7) or \
                       (iou_val > iou_threshold and sim_val > text_sim_threshold):
                        box_matches += 1
                        break
            
            # Calculate what percentage of boxes matched
            match_percentage = box_matches / total_boxes if total_boxes > 0 else 0
            
            # If most boxes match, consider it the same frame group
            if match_percentage >= 0.75 and match_percentage > best_match_score:
                matched_group_id = group_id
                best_match_score = match_percentage
        
        # Handle matched frame group
        if matched_group_id is not None:
            group = active_frame_groups[matched_group_id]
            
            # Update end time
            group["end_time"] = timestamp
            
            # Update texts with higher scores if they're the same
            for text_item in texts_in_frame:
                text = text_item["text"]
                box = text_item["box"]
                score = text_item["score"]
                
                for group_text in group["texts"]:
                    if group_text["text"] == text and score > group_text["score"]:
                        group_text["score"] = score
                        group_text["box"] = box
                        break
        else:
            # Create new frame group
            if debug_logging:
                text_list = [t["text"] for t in texts_in_frame]
                logger.info(f"  Creating new frame group at {timestamp:.2f}: {text_list}")
            
            new_group = {
                "group_id": next_group_id,
                "start_time": timestamp,
                "end_time": timestamp,  # Will be updated as tracking continues
                "texts": texts_in_frame.copy(),  # Make a copy to avoid reference issues
                "frame_count": 1
            }
            active_frame_groups[next_group_id] = new_group
            next_group_id += 1
        
        # Check for frame groups that didn't get updated in this pass
        current_timestamp = timestamp
        groups_to_remove = []
        
        for group_id, group in active_frame_groups.items():
            # If this group wasn't updated at this timestamp and wasn't just created
            if group["end_time"] < current_timestamp and group["frame_count"] > 1:
                # The group has ended
                if debug_logging:
                    duration = group["end_time"] - group["start_time"]
                    text_list = [t["text"] for t in group["texts"]]
                    logger.info(f"  Frame group ended: {text_list} ({duration:.2f}s, {group['start_time']:.2f}-{group['end_time']:.2f})")
                
                # Add to final results
                subtitle_segments.append(group)
                groups_to_remove.append(group_id)
            elif group["end_time"] == current_timestamp and group_id != matched_group_id:
                # Update frame count for active groups
                group["frame_count"] += 1
        
        # Remove ended groups
        for group_id in groups_to_remove:
            del active_frame_groups[group_id]
    
    # Add any remaining active frame groups to the results
    for group in active_frame_groups.values():
        # Ensure single-frame texts have minimum display time
        if group["start_time"] == group["end_time"]:
            group["end_time"] += min_display_time
        
        subtitle_segments.append(group)
    
    # Sort segments by start time
    subtitle_segments.sort(key=lambda x: x["start_time"])
    
    if debug_logging:
        logger.info(f"Created {len(subtitle_segments)} subtitle segments")
        for i, segment in enumerate(subtitle_segments):
            text_list = [t["text"] for t in segment["texts"]]
            duration = segment["end_time"] - segment["start_time"]
            logger.info(f"Segment {i+1}: {text_list} ({duration:.2f}s, {segment['start_time']:.2f}-{segment['end_time']:.2f})")
    
    # Replace video_results with the subtitle segments
    video_results.clear()
    for segment in subtitle_segments:
        video_results.append({
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "timestamp": segment["start_time"],  # For backward compatibility
            "texts": segment["texts"]
        })

    return video_results
def process_video(
    ocr_engine: OpenOCR,
    video_path: str,
    roi: Optional[List[float]] = None,
    line_y_thresh: float = 0.5,
    line_x_gap: float = 0.3,
    do_merge: bool = True,
    iou_threshold: float = 0.5,
    min_interval: float = 5.0,
    sec_skip: float = 1.0,
    text_sim_threshold: float = 0.8,
    filter_config: Optional[Dict[str, Any]] = None,
    debug_det_dir: Optional[str] = None,
    debug_box_dir: Optional[str] = None,
    num_processes: int = 4
) -> List[Dict[str, Union[float, List[Dict[str, Union[str, float, List[float]]]]]]]:
    """
    Process video for OCR with optimized subtitle extraction.
    Simplified to detect directly on original frames.
    """
    # Open video to get info
    cap_temp = cv2.VideoCapture(video_path)
    if not cap_temp.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
        
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shape = (height, width)
    cap_temp.release()
    
    logger.info(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Calculate frame skip interval
    skip_interval = max(1, int(round(sec_skip * fps)))
    logger.info(f"Processing every {skip_interval} frames ({sec_skip} seconds)")
    
    # Divide video into segments for parallel processing
    num_processes = min(num_processes, multiprocessing.cpu_count())
    segment_size = total_frames // num_processes
    processes = []
    queues = []
    
    t1 = time.time()
    # Initialize video reading processes
    for i in range(num_processes):
        q = multiprocessing.Queue(maxsize=512)
        start_frame = i * segment_size
        end_frame = total_frames if i == num_processes - 1 else (i + 1) * segment_size
        p = multiprocessing.Process(
            target=read_video_segment,
            args=(video_path, start_frame, end_frame, skip_interval, fps, q)
        )
        processes.append(p)
        queues.append(q)
        p.start()
    
    # Collect frames from processes
    frames_data = []  # Each element: (original_frame)
    timestamps = []
    finished_count = 0
    error_messages = []
    
    while finished_count < num_processes:
        for idx, q in enumerate(queues):
            if q is None:
                continue
            try:
                item = q.get(timeout=0.1)
                if item is None:
                    finished_count += 1
                    queues[idx] = None
                elif isinstance(item, tuple) and item[0] == "ERROR":
                    error_messages.append(item[1])
                else:
                    # item: (frame_index, frame, current_sec)
                    _, frame, current_sec = item
                    frames_data.append(frame)
                    timestamps.append(current_sec)
            except queue.Empty:
                continue
        queues = [q for q in queues if q is not None]
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    if error_messages:
        for msg in error_messages:
            logger.error(msg)
    
    if not frames_data:
        logger.warning("No frames captured")
        return []
    
    logger.info(f"Read {len(frames_data)} frames in {time.time() - t1:.2f} seconds")
    
    # Sort frames by timestamp
    sorted_data = sorted(zip(timestamps, frames_data), key=lambda x: x[0])
    timestamps = [item[0] for item in sorted_data]
    frames_data = [item[1] for item in sorted_data]
    
    # 1) Batch detection on the original images
    t_det = time.time()
    # Apply blur_and_reduce_contrast to handle faint background text
    detection_images = [blur_and_reduce_contrast(frame, kernel_size=15, sigmaX=0, contrast_alpha=0.3) for frame in frames_data]
    
    # Perform text detection on the batch
    all_dt_boxes, _ = ocr_engine.infer_batch_image_det(detection_images)
    logger.info(f"Detection completed in {time.time() - t_det:.2f} seconds")
    
    # 2) Filter boxes by ROI and size
    t_filter = time.time()
    filtered_boxes = []
    
    for i, dt_boxes_i in enumerate(all_dt_boxes):
        frame = frames_data[i]
        frame_boxes = []
        
        # Handle case with no detections
        if dt_boxes_i is None or len(dt_boxes_i) == 0:
            filtered_boxes.append([])
            continue
            
        for box in dt_boxes_i:
            # Apply size filtering
            if filter_config:
                x, y, bw, bh = get_box_xywh(box)
                h, w = frame.shape[:2]
                w_ratio = bw / w
                h_ratio = bh / h
                
                min_w_ratio = filter_config.get('min_w_ratio', 0.01)
                min_h_ratio = filter_config.get('min_h_ratio', 0.01)
                max_w_ratio = filter_config.get('max_w_ratio', 0.9)
                max_h_ratio = filter_config.get('max_h_ratio', 0.15)
                
                if not (min_w_ratio <= w_ratio <= max_w_ratio and 
                        min_h_ratio <= h_ratio <= max_h_ratio):
                    continue
            
            # Only keep boxes that are within ROI
            if roi is None or is_box_in_roi(box, roi, frame.shape[:2]):
                frame_boxes.append(box)
        
        filtered_boxes.append(frame_boxes)
    
    logger.info(f"Box filtering completed in {time.time() - t_filter:.2f} seconds")
    
    # Debug: Save detection visualization
    if debug_det_dir:
        os.makedirs(debug_det_dir, exist_ok=True)
        for i, boxes in enumerate(filtered_boxes):
            dbg_img = frames_data[i].copy()
            
            # Draw detected boxes
            for box in boxes:
                poly = np.array(box, dtype=np.int32)
                cv2.polylines(dbg_img, [poly], isClosed=True, color=(0,255,0), thickness=2)
            
            # Draw ROI if specified
            if roi:
                h, w = dbg_img.shape[:2]
                roi_x1 = int(roi[0] * w)
                roi_y1 = int(roi[1] * h)
                roi_x2 = int(roi[2] * w)
                roi_y2 = int(roi[3] * h)
                cv2.rectangle(dbg_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
            
            out_path = os.path.join(debug_det_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(out_path, dbg_img)
    
    # 3) Merge boxes into lines if needed and prepare for recognition
    t_prep = time.time()
    crops = []
    frame_idx_of_crop = []
    crop_boxes = []
    crop_counter = 0
    
    for i, boxes in enumerate(filtered_boxes):
        frame = frames_data[i]
        
        # Skip if no boxes for this frame
        if not boxes:
            continue
        
        # Merge boxes into lines if requested
        if do_merge:
            merged_boxes = same_line_merge(
                boxes, 
                line_y_thresh_ratio=line_y_thresh, 
                line_x_gap_ratio=line_x_gap
            )
            process_boxes = merged_boxes
            
            # Debug: Draw both original boxes and merged lines
            if debug_det_dir:
                merge_debug_img = frame.copy()
                
                # Draw original boxes in green
                for box in boxes:
                    poly = np.array(box, dtype=np.int32)
                    cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(0,255,0), thickness=1)
                
                # Draw merged lines in blue with thicker lines
                for merged_box in merged_boxes:
                    poly = np.array(merged_box, dtype=np.int32)
                    cv2.polylines(merge_debug_img, [poly], isClosed=True, color=(255,0,0), thickness=2)
                
                # Draw ROI if specified
                if roi:
                    h, w = merge_debug_img.shape[:2]
                    roi_x1 = int(roi[0] * w)
                    roi_y1 = int(roi[1] * h)
                    roi_x2 = int(roi[2] * w)
                    roi_y2 = int(roi[3] * h)
                    cv2.rectangle(merge_debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
                
                merge_debug_path = os.path.join(debug_det_dir, f"frame_{i:06d}_merged.jpg")
                cv2.imwrite(merge_debug_path, merge_debug_img)
        else:
            process_boxes = boxes
        
        # Create crops for each box/line
        for box in process_boxes:
            # Get tight crop around text
            text_crop = get_text_crop(frame, box)
            
            # Save debug image if needed
            if debug_box_dir:
                os.makedirs(debug_box_dir, exist_ok=True)
                debug_path = os.path.join(debug_box_dir, f"frame_{i:06d}_box_{crop_counter:03d}.jpg")
                cv2.imwrite(debug_path, text_crop)
                crop_counter += 1
            
            # Enhance the crop for recognition
            enhanced_crop = enhance_text_image(text_crop)
            
            # Add to crops list
            crops.append(Image.fromarray(enhanced_crop))
            frame_idx_of_crop.append(i)
            crop_boxes.append(box)
    
    logger.info(f"Prepared {len(crops)} crops in {time.time() - t_prep:.2f} seconds")
    
    # 4) Perform batch text recognition
    t_rec = time.time()
    
    # Skip recognition if no crops
    if not crops:
        logger.warning("No text regions found for recognition")
        return []
    
    # Perform text recognition
    rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
    logger.info(f"Recognition completed in {time.time() - t_rec:.2f} seconds")
    
    # 5) Assign recognition results to frames
    final_texts_per_frame = [[] for _ in range(len(frames_data))]
    
    for j, (frame_idx, box, rec_result) in enumerate(zip(frame_idx_of_crop, crop_boxes, rec_res_all)):
        text_j, score_j, _ = rec_result
        
        # Only keep results with high score and non-empty text
        if score_j >= ocr_engine.drop_score and text_j.strip():
            final_texts_per_frame[frame_idx].append({
                "text": text_j,
                "score": float(score_j),
                "box": get_box_xywh(box)
            })
    
    # 6) Create initial results and apply text tracking
    t_post = time.time()
    
    # Create initial results from frames with text
    video_results = []
    for i, texts in enumerate(final_texts_per_frame):
        if texts:  # Only add frames with text
            video_results.append({
                "timestamp": timestamps[i],
                "texts": texts
            })
    
    # Apply unified text tracking and deduplication
    track_and_deduplicate_text(video_results, iou_threshold, text_sim_threshold)
    
    logger.info(f"Post-processing completed in {time.time() - t_post:.2f} seconds")
    return video_results

###############################################################################
# SRT Generation
###############################################################################

def generate_srt_from_results(
    video_results: List[Dict[str, Any]],
    ext_time_sec: float = 1.0,  # Extra time to show each subtitle
    min_duration_sec: float = 1.0,  # Minimum subtitle duration
    output_srt: str = "output.srt",
    debug_logging: bool = False
):
    """
    Generate SRT subtitle file from segmented OCR results.
    """
    if not video_results:
        logger.warning("No results to generate SRT.")
        return

    if debug_logging:
        logger.info(f"Starting SRT generation with {len(video_results)} segments")
    
    # Convert to SRT format
    srt_entries = []
    
    for item in video_results:
        start_time = item["start_time"]
        end_time = item.get("end_time", start_time + 1.0)  # Default to start+1s for backward compatibility
        
        # Ensure minimum duration
        if end_time - start_time < min_duration_sec:
            end_time = start_time + min_duration_sec
        
        # Add extra display time
        end_time += ext_time_sec
        
        # Sort texts by vertical position (top to bottom)
        texts = item["texts"]
        sorted_texts = sorted(texts, key=lambda x: x["box"][1])
        
        # Join multiple lines with pipe
        text = " | ".join([x["text"] for x in sorted_texts])
        
        srt_entries.append({
            "start": start_time,
            "end": end_time,
            "text": text
        })
    
    # Sort by start time and write SRT file
    srt_entries.sort(key=lambda x: x["start"])
    
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(srt_entries, start=1):
            # Convert seconds to HH:MM:SS,mmm format
            def sec_to_hmsms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                ms = int((sec - int(sec)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                
            start_time_str = sec_to_hmsms(entry["start"])
            end_time_str = sec_to_hmsms(entry["end"])
            
            # Write SRT format
            f.write(f"{idx}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{entry['text']}\n\n")
            
            if debug_logging:
                logger.info(f"Subtitle {idx}: {start_time_str} --> {end_time_str}")
                logger.info(f"  Text: {entry['text']}")
    
    logger.info(f"SRT saved to: {output_srt}")


def process_video_with_batches(
    ocr_engine: OpenOCR,
    video_path: str,
    roi: Optional[List[float]] = None,
    line_y_thresh: float = 0.5,
    line_x_gap: float = 0.3,
    do_merge: bool = True,
    iou_threshold: float = 0.5,
    min_interval: float = 5.0,
    sec_skip: float = 1.0,
    text_sim_threshold: float = 0.8,
    filter_config: Optional[Dict[str, Any]] = None,
    debug_det_dir: Optional[str] = None,
    debug_box_dir: Optional[str] = None,
    batch_size: int = 3000,
    resize_factor: float = 1.0  # Set to <1.0 to reduce frame size
):
    """
    Process video for OCR with optimized memory usage using batches.
    """
    # Open video to get info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shape = (height, width)
    
    logger.info(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Calculate frame skip interval
    skip_interval = max(1, int(round(sec_skip * fps)))
    logger.info(f"Processing every {skip_interval} frames ({sec_skip} seconds)")
    
    # Prepare for processing in batches
    all_results = []
    detection_queue = []
    timestamps = []
    frame_count = 0
    
    # Calculate new dimensions if resizing
    new_width, new_height = width, height
    if resize_factor != 1.0:
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        logger.info(f"Resizing frames to {new_width}x{new_height}")
    
    try:
        # Start frame reading and processing loop
        while True:
            # Collect a batch of frames
            batch_frames = []
            batch_timestamps = []
            
            # Read frames for this batch
            for _ in range(batch_size):
                # Skip frames according to interval
                for _ in range(skip_interval):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_count / fps
                
                # Resize if needed
                if resize_factor != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Add to batch
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
            
            # If no frames were read, we're done
            if not batch_frames:
                break
            
            # Apply blur and contrast reduction to batch
            detection_images = [blur_and_reduce_contrast(frame) for frame in batch_frames]
            
            # Perform detection on the batch
            all_dt_boxes, _ = ocr_engine.infer_batch_image_det(detection_images)
            
            # Filter boxes by ROI and size
            filtered_boxes = []
            for i, dt_boxes_i in enumerate(all_dt_boxes):
                frame = batch_frames[i]
                frame_boxes = []
                
                if dt_boxes_i is None or len(dt_boxes_i) == 0:
                    filtered_boxes.append([])
                    continue
                    
                for box in dt_boxes_i:
                    # Apply filtering
                    if filter_config:
                        x, y, bw, bh = get_box_xywh(box)
                        h, w = frame.shape[:2]
                        w_ratio = bw / w
                        h_ratio = bh / h
                        
                        min_w_ratio = filter_config.get('min_w_ratio', 0.01)
                        min_h_ratio = filter_config.get('min_h_ratio', 0.01)
                        max_w_ratio = filter_config.get('max_w_ratio', 0.9)
                        max_h_ratio = filter_config.get('max_h_ratio', 0.15)
                        
                        if not (min_w_ratio <= w_ratio <= max_w_ratio and 
                                min_h_ratio <= h_ratio <= max_h_ratio):
                            continue
                    
                    # Check ROI
                    if roi is None or is_box_in_roi(box, roi, frame.shape[:2]):
                        frame_boxes.append(box)
                
                filtered_boxes.append(frame_boxes)
            
            # Process each frame with text
            for i, (frame, boxes, timestamp) in enumerate(zip(batch_frames, filtered_boxes, batch_timestamps)):
                if not boxes:
                    continue
                
                # Merge boxes into lines if needed
                if do_merge:
                    process_boxes = same_line_merge(
                        boxes, 
                        line_y_thresh_ratio=line_y_thresh, 
                        line_x_gap_ratio=line_x_gap
                    )
                else:
                    process_boxes = boxes
                
                # Create crops for each box
                crops = []
                crop_boxes = []
                
                for box in process_boxes:
                    text_crop = get_text_crop(frame, box)
                    enhanced_crop = enhance_text_image(text_crop)
                    crops.append(Image.fromarray(enhanced_crop))
                    crop_boxes.append(box)
                
                # Skip if no crops
                if not crops:
                    continue
                
                # Perform recognition
                rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
                
                # Collect results
                frame_results = []
                for j, (box, rec_result) in enumerate(zip(crop_boxes, rec_res_all)):
                    text_j, score_j, _ = rec_result
                    
                    if score_j >= ocr_engine.drop_score and text_j.strip():
                        frame_results.append({
                            "text": text_j,
                            "score": float(score_j),
                            "box": get_box_xywh(box)
                        })
                
                # Add to results if text found
                if frame_results:
                    all_results.append({
                        "timestamp": timestamp,
                        "texts": frame_results
                    })
            
            # Log progress
            logger.info(f"Processed {frame_count}/{total_frames} frames, found text in {len(all_results)} frames")
            
            # Free memory
            del batch_frames
            del detection_images
            # gc.collect()
    
    finally:
        cap.release()
    
    # Apply text tracking to results
    if all_results:
        track_and_deduplicate_text(
            all_results,
            iou_threshold=iou_threshold,
            text_sim_threshold=text_sim_threshold,
            min_display_time=0.5
        )
    
    return all_results



def calculate_optimal_batch_size(frame_shape, max_memory_gb, current_memory_gb=None):
    """
    Calculate optimal batch size based on frame dimensions and memory constraints.
    
    Args:
        frame_shape: Tuple of (height, width, channels)
        max_memory_gb: Maximum allowed memory in GB
        current_memory_gb: Current memory usage in GB (optional)
    
    Returns:
        Optimal batch size
    """
    import psutil
    
    # Get current memory usage if not provided
    if current_memory_gb is None:
        process = psutil.Process()
        current_memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    
    # Calculate bytes per frame (uncompressed)
    height, width, channels = frame_shape
    bytes_per_pixel = 3  # RGB format, 3 bytes per pixel
    bytes_per_frame = height * width * channels * bytes_per_pixel
    
    # Factor in memory for processing (OCR, detection, etc.)
    # Typically processing requires ~3-5x the raw frame size
    processing_factor = 5.0
    effective_bytes_per_frame = bytes_per_frame * processing_factor
    
    # Calculate available memory in bytes
    available_memory_bytes = (max_memory_gb - current_memory_gb) * 1024 * 1024 * 1024
    
    # Reserve 20% of max memory for other operations
    reserve_factor = 0.8
    available_memory_bytes *= reserve_factor
    
    # Calculate maximum frames that fit in available memory
    max_frames = int(available_memory_bytes / effective_bytes_per_frame)
    
    # Ensure batch size is at least 1 and not too large
    optimal_batch_size = max(1, min(max_frames, 32))
    
    return optimal_batch_size

def calculate_optimal_batch_size(frame_shape, max_memory_gb, current_memory_gb=None):
    """
    Calculate optimal batch size based on frame dimensions and memory constraints.
    
    Args:
        frame_shape: Tuple of (height, width, channels)
        max_memory_gb: Maximum allowed memory in GB
        current_memory_gb: Current memory usage in GB (optional)
    
    Returns:
        Optimal batch size
    """
    import psutil
    
    # Get current memory usage if not provided
    if current_memory_gb is None:
        process = psutil.Process()
        current_memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    
    # Calculate bytes per frame (uncompressed)
    height, width, channels = frame_shape
    bytes_per_pixel = 4  # RGB with potential alpha/padding, 4 bytes per pixel for safety
    bytes_per_frame = height * width * channels * bytes_per_pixel
    
    # Factor in memory for processing (OCR, detection, etc.)
    # Typically processing requires ~3-5x the raw frame size
    processing_factor = 5.0
    effective_bytes_per_frame = bytes_per_frame * processing_factor
    
    # Calculate available memory in bytes
    available_memory_bytes = (max_memory_gb - current_memory_gb) * 1024 * 1024 * 1024
    
    # Reserve 20% of max memory for other operations
    reserve_factor = 0.8
    available_memory_bytes *= reserve_factor
    
    # Calculate maximum frames that fit in available memory
    max_frames = int(available_memory_bytes / effective_bytes_per_frame)
    
    # Ensure batch size is at least 1 and not too large
    optimal_batch_size = max(1, min(max_frames, 32))
    
    return optimal_batch_size


def process_video_memory_optimized(
    ocr_engine: OpenOCR,
    video_path: str,
    roi: Optional[List[float]] = None,
    sec_skip: float = 1.0,
    initial_batch_size: int = 8,
    max_memory_gb: float = 4.0,
    resize_factor: float = 0.5,
    segment_duration_sec: int = 60,
    dynamic_sizing: bool = True,
    **kwargs
):
    """
    Memory-optimized video processing with improved handling for videos of all lengths.
    
    Args:
        ocr_engine: OCR engine instance
        video_path: Path to video file
        roi: Optional region of interest [x1, y1, x2, y2] as ratios
        sec_skip: Process every n seconds
        initial_batch_size: Starting batch size for processing
        max_memory_gb: Maximum memory limit in GB
        resize_factor: Frame size reduction factor (0-1)
        segment_duration_sec: Duration of each video segment in seconds
        dynamic_sizing: Whether to dynamically adjust batch and segment sizes
        **kwargs: Additional parameters for OCR processing
    
    Returns:
        List of OCR results with timestamps
    """
    import gc
    import psutil
    import time
    from tqdm import tqdm
    import os
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    import logging
    file_handler = logging.FileHandler("logs/video_processing.log", mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Get video info
    logger.info(f"Opening video: {video_path}")
    video_start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    cap.release()
    
    logger.info(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames, {duration_sec:.2f} seconds")
    
    # Calculate skip interval
    skip_interval = max(1, int(round(sec_skip * fps)))
    processed_frame_count = total_frames // skip_interval
    logger.info(f"Processing every {skip_interval} frames ({sec_skip}s), approx. {processed_frame_count} frames")
    
    # Calculate resize dimensions
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    logger.info(f"Resizing frames to {new_width}x{new_height} ({resize_factor*100:.0f}%)")
    
    # Calculate frame memory footprint
    frame_shape = (new_height, new_width, 3)
    bytes_per_pixel = 4  # RGB + alignment padding
    bytes_per_frame = new_height * new_width * 3 * bytes_per_pixel
    mb_per_frame = bytes_per_frame / (1024 * 1024)
    logger.info(f"Estimated memory per frame: {mb_per_frame:.2f} MB")
    
    # Initialize batch size
    batch_size = initial_batch_size
    
    # Calculate optimal parameters based on video length
    if dynamic_sizing:
        # For short videos, use larger batch sizes
        if duration_sec < 300:  # Less than 5 minutes
            batch_size = min(64, initial_batch_size * 2)
            logger.info(f"Short video detected: Increasing batch size to {batch_size}")
        
        # Calculate optimal segment duration based on video length
        if segment_duration_sec == 60:  # Only if user didn't specify custom value
            # For very short videos (under 3 minutes), process in one segment
            if duration_sec < 180:  # 3 minutes
                segment_duration_sec = duration_sec
            # For short videos (3-10 minutes), use fewer segments
            elif duration_sec < 600:  # 10 minutes
                segment_duration_sec = max(60, duration_sec / 3)  # 3 segments max
            # For medium videos (10-30 minutes), balance segments
            elif duration_sec < 1800:  # 30 minutes
                segment_duration_sec = max(60, duration_sec / 10)  # 10 segments max
            # For long videos, use memory-based calculation
            else:
                # Calculate how many frames we can hold in memory
                frames_in_memory = (max_memory_gb * 0.6 * 1024) / (mb_per_frame * 5)  # 5x for processing overhead
                segment_duration_sec = max(30, min(300, frames_in_memory / fps))
            
            logger.info(f"Optimized segment duration: {segment_duration_sec:.2f} seconds")
    
    # For very short videos, process in a single segment
    single_segment = duration_sec < 180 and dynamic_sizing
    
    if single_segment:
        logger.info(f"Short video detected ({duration_sec:.2f}s): Processing in a single segment")
        frames_per_segment = total_frames
    else:
        frames_per_segment = int(segment_duration_sec * fps)
    
    # Initialize results and tracking
    all_results = []
    memory_usage_log = []
    timings_log = []
    
    # Function to log memory usage
    def log_memory_usage():
        process = psutil.Process()
        memory_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        logger.info(f"Memory usage: {memory_usage_gb:.2f} GB of {max_memory_gb:.2f} GB limit")
        return memory_usage_gb
    
    # Log initial memory usage
    initial_memory_gb = log_memory_usage()
    memory_usage_log.append(("initial", 0, initial_memory_gb))
    
    # Calculate segments
    segments = []
    for start_frame in range(0, total_frames, int(frames_per_segment)):
        end_frame = min(start_frame + int(frames_per_segment), total_frames)
        segments.append((start_frame, end_frame))
    
    logger.info(f"Divided video into {len(segments)} segments")
    
    # Process each segment
    for segment_idx, (start_frame, end_frame) in enumerate(segments):
        segment_start_time = time.time()
        segment_frame_count = end_frame - start_frame
        
        logger.info(f"Processing segment {segment_idx+1}/{len(segments)}: frames {start_frame}-{end_frame} " 
                   f"({segment_frame_count} frames, {segment_frame_count/fps:.2f} seconds)")
        
        # Open video for this segment
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video for segment {segment_idx+1}")
            continue
        
        # Track segment processing
        segment_results = []
        segment_frame_processed = 0
        batch_count = 0
        current_frame = start_frame
        
        # Progress bar for this segment
        pbar = tqdm(total=segment_frame_count, desc=f"Segment {segment_idx+1}", unit="frames")
        
        # Process the segment in batches
        while current_frame < end_frame:
            batch_start_time = time.time()
            
            # Check memory and adjust batch size if needed
            current_memory_gb = log_memory_usage()
            memory_usage_log.append((f"batch_start_{segment_idx}_{batch_count}", 
                                    segment_frame_processed, current_memory_gb))
            
            if dynamic_sizing:
                if current_memory_gb > max_memory_gb * 0.8:
                    # Memory usage is high, reduce batch size
                    old_batch_size = batch_size
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"Memory usage high ({current_memory_gb:.2f} GB). "
                                 f"Reducing batch size: {old_batch_size} -> {batch_size}")
                    
                    # Force garbage collection
                    gc.collect()
                elif current_memory_gb < max_memory_gb * 0.4 and batch_size < initial_batch_size * 2:
                    # Memory usage is low, gradually increase batch size
                    old_batch_size = batch_size
                    batch_size = min(initial_batch_size * 2, batch_size + 2)
                    logger.info(f"Memory usage low ({current_memory_gb:.2f} GB). "
                              f"Increasing batch size: {old_batch_size} -> {batch_size}")
            
            # Read batch of frames
            read_start_time = time.time()
            batch_frames = []
            batch_timestamps = []
            
            # Calculate how many frames to read in this batch
            frames_to_read = min(batch_size, (end_frame - current_frame + skip_interval - 1) // skip_interval)
            
            for _ in range(frames_to_read):
                # Position the video at the exact frame
                frame_to_read = current_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_read)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame {frame_to_read}")
                    break
                
                # Update progress
                progress_frames = min(skip_interval, end_frame - current_frame)
                pbar.update(progress_frames)
                
                # Calculate timestamp
                timestamp = frame_to_read / fps
                
                # Resize frame to save memory
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Add to batch
                batch_frames.append(frame)
                batch_timestamps.append(timestamp)
                
                # Move to next frame to process
                current_frame = frame_to_read + skip_interval
                segment_frame_processed += 1
                
                if current_frame >= end_frame:
                    break
            
            read_time = time.time() - read_start_time
            timings_log.append(("read", segment_idx, batch_count, frames_to_read, read_time))
            logger.debug(f"Batch {batch_count+1}: Read {len(batch_frames)} frames in {read_time:.2f}s " 
                        f"({len(batch_frames)/read_time if read_time > 0 else 0:.2f} fps)")
            
            # If no frames were read, we're done with this segment
            if not batch_frames:
                break
            
            # Apply blur_and_reduce_contrast to handle background text
            process_start_time = time.time()
            detection_images = [blur_and_reduce_contrast(frame) for frame in batch_frames]
            preprocess_time = time.time() - process_start_time
            timings_log.append(("preprocess", segment_idx, batch_count, len(batch_frames), preprocess_time))
            
            # Perform text detection on the batch
            detect_start_time = time.time()
            all_dt_boxes, _ = ocr_engine.infer_batch_image_det(detection_images)
            detect_time = time.time() - detect_start_time
            timings_log.append(("detection", segment_idx, batch_count, len(batch_frames), detect_time))
            logger.debug(f"Detection completed in {detect_time:.2f}s " 
                        f"({len(batch_frames)/detect_time if detect_time > 0 else 0:.2f} frames/s)")
            
            # Release detection images to save memory
            del detection_images
            
            # Process each frame with detections
            process_frames_start_time = time.time()
            batch_results = []
            
            for i, (frame, dt_boxes, timestamp) in enumerate(zip(batch_frames, all_dt_boxes, batch_timestamps)):
                # Skip if no boxes detected
                if dt_boxes is None or len(dt_boxes) == 0:
                    continue
                
                # Filter boxes by ROI and size
                filter_start_time = time.time()
                filtered_boxes = []
                
                for box in dt_boxes:
                    # Apply size filtering
                    if kwargs.get('filter_config'):
                        x, y, bw, bh = get_box_xywh(box)
                        h, w = frame.shape[:2]
                        w_ratio = bw / w
                        h_ratio = bh / h
                        
                        filter_config = kwargs.get('filter_config')
                        min_w_ratio = filter_config.get('min_w_ratio', 0.01)
                        min_h_ratio = filter_config.get('min_h_ratio', 0.01)
                        max_w_ratio = filter_config.get('max_w_ratio', 0.9)
                        max_h_ratio = filter_config.get('max_h_ratio', 0.15)
                        
                        if not (min_w_ratio <= w_ratio <= max_w_ratio and 
                                min_h_ratio <= h_ratio <= max_h_ratio):
                            continue
                    
                    # Check if box is within ROI
                    if roi is None or is_box_in_roi(box, roi, frame.shape[:2]):
                        filtered_boxes.append(box)
                
                filter_time = time.time() - filter_start_time
                
                # Skip if no boxes after filtering
                if not filtered_boxes:
                    continue
                
                # Merge boxes into lines if needed
                if kwargs.get('do_merge', True):
                    merge_start_time = time.time()
                    process_boxes = same_line_merge(
                        filtered_boxes, 
                        line_y_thresh_ratio=kwargs.get('line_y_thresh', 0.5), 
                        line_x_gap_ratio=kwargs.get('line_x_gap', 0.3)
                    )
                    merge_time = time.time() - merge_start_time
                else:
                    process_boxes = filtered_boxes
                    merge_time = 0
                
                # Create crops for text recognition
                crop_start_time = time.time()
                crops = []
                crop_boxes = []
                
                for box in process_boxes:
                    text_crop = get_text_crop(frame, box)
                    enhanced_crop = enhance_text_image(text_crop)
                    crops.append(Image.fromarray(enhanced_crop))
                    crop_boxes.append(box)
                
                crop_time = time.time() - crop_start_time
                
                # Skip if no crops
                if not crops:
                    continue
                
                # Perform text recognition
                rec_start_time = time.time()
                rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
                rec_time = time.time() - rec_start_time
                
                # Collect recognition results
                frame_texts = []
                for j, (box, rec_result) in enumerate(zip(crop_boxes, rec_res_all)):
                    text_j, score_j, _ = rec_result
                    
                    if score_j >= ocr_engine.drop_score and text_j.strip() and not re.fullmatch(r"[A-Z]{1,2}", text_j.strip()):
                        frame_texts.append({
                            "text": text_j,
                            "score": float(score_j),
                            "box": get_box_xywh(box)
                        })
                
                # Add frame results if text found
                if frame_texts:
                    batch_results.append({
                        "timestamp": timestamp,
                        "texts": frame_texts
                    })
                    
                    # Log timing for frames with text
                    timings_log.append(("frame_processing", segment_idx, batch_count, i, {
                        "filter": filter_time,
                        "merge": merge_time,
                        "crop": crop_time,
                        "recognition": rec_time,
                        "total": filter_time + merge_time + crop_time + rec_time
                    }))
            
            process_frames_time = time.time() - process_frames_start_time
            timings_log.append(("process_frames", segment_idx, batch_count, len(batch_frames), process_frames_time))
            
            # Add batch results to segment results
            segment_results.extend(batch_results)
            
            # Log batch completion
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {batch_count+1}: Processed {len(batch_frames)} frames in {batch_time:.2f}s, "
                      f"Found text in {len(batch_results)} frames")
            
            # Log memory usage after batch
            end_memory_gb = log_memory_usage()
            memory_usage_log.append((f"batch_end_{segment_idx}_{batch_count}", 
                                    segment_frame_processed, end_memory_gb))
            
            # Clean up batch data
            del batch_frames
            gc.collect()
            
            batch_count += 1
        
        # Close video for this segment
        cap.release()
        pbar.close()
        
        # Add segment results to overall results
        all_results.extend(segment_results)
        
        # Log segment completion
        segment_time = time.time() - segment_start_time
        logger.info(f"Segment {segment_idx+1}/{len(segments)} completed in {segment_time:.2f}s - "
                  f"Found text in {len(segment_results)}/{segment_frame_processed} frames")
        
        # Force garbage collection between segments
        gc.collect()
        
        # Log memory after segment
        segment_end_memory_gb = log_memory_usage()
        memory_usage_log.append((f"segment_end_{segment_idx}", 
                                segment_frame_processed, segment_end_memory_gb))
    
    # Apply text tracking and deduplication
    tracking_start_time = time.time()
    if all_results:
        logger.info(f"Applying text tracking to {len(all_results)} frames with text")
        track_and_deduplicate_text(
            all_results,
            iou_threshold=kwargs.get('iou_threshold', 0.5),
            text_sim_threshold=kwargs.get('text_sim_threshold', 0.95),
            min_display_time=0.5,
            debug_logging=kwargs.get('debug_logging', True)
        )
    tracking_time = time.time() - tracking_start_time
    logger.info(f"Text tracking completed in {tracking_time:.2f}s, final result: {len(all_results)} subtitle segments")
    
    # Log total processing time
    total_time = time.time() - video_start_time
    processing_rate = duration_sec / total_time if total_time > 0 else 0
    logger.info(f"Total processing time: {total_time:.2f}s for {duration_sec:.2f}s video ({processing_rate:.2f}x realtime)")
    
    
    return all_results


###############################################################################
# Main Function
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Video OCR for subtitle extraction with memory optimization.')
    # Core file parameters
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video.')
    parser.add_argument('--output_json', type=str, default='output_results.json', help='Path to save JSON results.')
    parser.add_argument('--output_srt', type=str, default='output_results.srt', help='Path to save SRT.')
    parser.add_argument('--generate_srt', action='store_true', default=True, help='Generate SRT file.')
    
    # OCR model parameters
    parser.add_argument('--cfg_det_path', type=str, default="configs/det/dbnet/repvit_db.yml")
    parser.add_argument('--cfg_rec_path', type=str, default="configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml")
    parser.add_argument('--drop_score', type=float, default=0.9, help='Minimum confidence score for text detection.')
    parser.add_argument('--det_batch_size', type=int, default=32)
    parser.add_argument('--rec_batch_size', type=int, default=32)
    
    # Processing parameters
    parser.add_argument('--sec_skip', type=float, default=0.5, help='Skip frames by seconds.')
    parser.add_argument('--line_y_thresh', type=float, default=0.5)
    parser.add_argument('--line_x_gap', type=float, default=0.3)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--min_interval', type=float, default=5.0)
    parser.add_argument('--text_sim_threshold', type=float, default=0.8)
    parser.add_argument('--roi', type=str, default="0.1,0.6,0.9,0.97", help='ROI format: x1,y1,x2,y2')
    
    # Debug parameters
    parser.add_argument('--debug_det_dir', type=str, default=None, help='Directory to save detection visualization.')
    parser.add_argument('--debug_box_dir', type=str, default=None, help='Directory to save text boxes.')
    
    # Box filtering parameters
    parser.add_argument('--min_w_ratio', type=float, default=0.02)
    parser.add_argument('--min_h_ratio', type=float, default=0.02)
    parser.add_argument('--max_w_ratio', type=float, default=0.9)
    parser.add_argument('--max_h_ratio', type=float, default=0.09)
    
    # Memory optimization parameters
    parser.add_argument('--max_memory_gb', type=float, default=8.0, help='Maximum memory usage in GB.')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Frame resize factor (0.5 = half size).')
    parser.add_argument('--initial_batch_size', type=int, default=8, help='Initial batch size for processing.')
    parser.add_argument('--segment_duration_sec', type=int, default=120, help='Duration of each video segment in seconds.')
    parser.add_argument('--dynamic_sizing', action='store_true', default=True, help='Enable dynamic batch and segment sizing.')
    parser.add_argument('--no_dynamic_sizing', dest='dynamic_sizing', action='store_false', help='Disable dynamic sizing.')
    
    args = parser.parse_args()

    # Parse ROI and filter config
    roi = parse_roi(args.roi)
    filter_config = {
        'min_w_ratio': args.min_w_ratio,
        'min_h_ratio': args.min_h_ratio,
        'max_w_ratio': args.max_w_ratio,
        'max_h_ratio': args.max_h_ratio
    }

    # Initialize OCR engine
    ocr_engine = OpenOCR(
        cfg_det_path=args.cfg_det_path,
        cfg_rec_path=args.cfg_rec_path,
        drop_score=args.drop_score,
        det_box_type='quad',
        det_batch_size=args.det_batch_size,
        rec_batch_size=args.rec_batch_size
    )
    
    # Start processing
    time_start = time.time()
    logger.info(f"Starting memory-optimized video processing: {args.video_path}")
    
    try:
        # Process video with memory optimization
        results = process_video_memory_optimized(
            ocr_engine=ocr_engine,
            video_path=args.video_path,
            roi=roi,
            sec_skip=args.sec_skip,
            initial_batch_size=args.initial_batch_size,
            max_memory_gb=args.max_memory_gb,
            resize_factor=args.resize_factor,
            segment_duration_sec=args.segment_duration_sec,
            dynamic_sizing=args.dynamic_sizing,
            line_y_thresh=args.line_y_thresh,
            line_x_gap=args.line_x_gap,
            do_merge=True,
            iou_threshold=args.iou_thresh,
            min_interval=args.min_interval,
            text_sim_threshold=args.text_sim_threshold,
            filter_config=filter_config,
            debug_det_dir=args.debug_det_dir,
            debug_box_dir=args.debug_box_dir
        )

        time_process = time.time() - time_start
        logger.info(f"Processing completed in {time_process:.2f} seconds")

        # Save JSON results
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {args.output_json}")

        # Generate SRT if requested
        if args.generate_srt:
            generate_srt_from_results(
                results, 
                min_duration_sec=1.0,
                ext_time_sec=1.0, 
                output_srt=args.output_srt
            )
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # Generate summary information
        try:
            total_time = time.time() - time_start
            if os.path.exists(args.output_json):
                with open(args.output_json, 'r', encoding='utf-8') as f:
                    saved_results = json.load(f)
                    
                num_subtitles = len(saved_results)
                total_text_count = sum(len(frame.get("texts", [])) for frame in saved_results)
                
                logger.info("=" * 50)
                logger.info(f"Processing Summary:")
                logger.info(f"- Total processing time: {total_time:.2f} seconds")
                logger.info(f"- Subtitle segments detected: {num_subtitles}")
                logger.info(f"- Total text boxes: {total_text_count}")
                logger.info(f"- Results saved to: {args.output_json}")
                if args.generate_srt:
                    logger.info(f"- SRT file created: {args.output_srt}")
                logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")

if __name__ == '__main__':
    # Ensure multiprocessing works correctly on all platforms
    multiprocessing.set_start_method('spawn')
    main()