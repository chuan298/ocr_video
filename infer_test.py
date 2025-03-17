# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import sys
# import argparse
# import time
# import json
# import copy
# import cv2
# import numpy as np
# from PIL import Image
# from typing import List, Tuple, Dict, Optional, Union, Any

# # Đảm bảo import các module nội bộ
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from tools.infer_rec import OpenRecognizer
# from tools.infer_det import OpenDetector
# from tools.engine import Config
# from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
# from tools.utils.logging import get_logger

# logger = get_logger()


# def parse_roi(roi_str: str) -> Optional[List[int]]:
#     """Parses a ROI string (x1,y1,x2,y2) into [x1, y1, x2, y2]."""
#     if not roi_str:
#         return None
#     try:
#         # Split the string và convert
#         vals = [int(v.strip()) for v in roi_str.split(',')]
#         return vals if len(vals) == 4 else None
#     except ValueError:
#         return None


# def _crop_with_offset(frame: np.ndarray, roi: Optional[List[int]]) -> Tuple[np.ndarray, int, int]:
#     """Crops an image to the ROI và returns the cropped image + offsets."""
#     if roi is None:
#         return frame, 0, 0

#     x1, y1, x2, y2 = roi
#     h, w = frame.shape[:2]

#     # Clamp
#     x1 = np.clip(x1, 0, w)
#     x2 = np.clip(x2, 0, w)
#     y1 = np.clip(y1, 0, h)
#     y2 = np.clip(y2, 0, h)

#     if x2 <= x1 or y2 <= y1:
#         # ROI không hợp lệ => return nguyên ảnh
#         return frame, 0, 0

#     cropped = frame[y1:y2, x1:x2]
#     return cropped, x1, y1


# def get_box_xywh(quad: Union[List[List[float]], np.ndarray]) -> Tuple[float, float, float, float]:
#     """Converts a quadrilateral (4 points) to (x, y, w, h)."""
#     quad = np.array(quad)
#     x_min, y_min = np.min(quad, axis=0)
#     x_max, y_max = np.max(quad, axis=0)
#     return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))


# def sorted_boxes(dt_boxes: List[Union[List[List[float]], np.ndarray]]) -> List[np.ndarray]:
#     """Sorts detected boxes top-to-bottom, left-to-right."""
#     boxes = np.array(dt_boxes)
#     if len(boxes) == 0:
#         return []
#     # Sort y rồi sort x
#     sorted_indices = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
#     sorted_boxes = boxes[sorted_indices]

#     # Refine sorting cho các box gần nhau trên trục y
#     for i in range(len(sorted_boxes) - 1):
#         if abs(sorted_boxes[i + 1, 0, 1] - sorted_boxes[i, 0, 1]) < 10:
#             if sorted_boxes[i + 1, 0, 0] < sorted_boxes[i, 0, 0]:
#                 sorted_boxes[[i, i + 1]] = sorted_boxes[[i + 1, i]]
#     return list(sorted_boxes)


# def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
#     """Computes Intersection over Union (IoU) between two boxes."""
#     xA, yA, wA, hA = boxA
#     xB, yB, wB, hB = boxB

#     x_start = max(xA, xB)
#     y_start = max(yA, yB)
#     x_end = min(xA + wA, xB + wB)
#     y_end = min(yA + hA, yB + hB)

#     inter_area = max(0, x_end - x_start) * max(0, y_end - y_start)
#     boxA_area = wA * hA
#     boxB_area = wB * hB

#     return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)


# def same_line_merge(dt_boxes: List[np.ndarray],
#                     rec_res: List[List[Union[str, float]]],
#                     line_y_thresh_ratio: float = 0.5,
#                     line_x_gap_ratio: float = 0.3
#                     ) -> Tuple[List[np.ndarray], List[List[Union[str, float]]]]:
#     """Gộp các bounding boxes nằm trên cùng một dòng (dựa vào khoảng cách)."""
#     if not dt_boxes:
#         return [], []

#     data = list(zip(dt_boxes, rec_res))
#     # Sort theo y rồi x
#     data.sort(key=lambda x: (x[0][0, 1], x[0][0, 0]))

#     merged = []
#     used = [False] * len(data)

#     for i in range(len(data)):
#         if used[i]:
#             continue

#         box_i, (text_i, score_i) = data[i]
#         used[i] = True
#         box_i = np.array(box_i)
#         min_x, min_y = np.min(box_i, axis=0)
#         max_x, max_y = np.max(box_i, axis=0)
#         group_text = str(text_i)
#         group_score = float(score_i)
#         group_count = 1

#         for j in range(i + 1, len(data)):
#             if used[j]:
#                 continue

#             box_j, (text_j, score_j) = data[j]
#             box_j = np.array(box_j)
#             min_xj, min_yj = np.min(box_j, axis=0)
#             max_xj, max_yj = np.max(box_j, axis=0)

#             # Kiểm tra line (theo y)
#             avg_h = (max_y - min_y + max_yj - min_yj) / 2.0
#             center_i_y = (min_y + max_y) / 2.0
#             center_j_y = (min_yj + max_yj) / 2.0
#             if abs(center_j_y - center_i_y) <= line_y_thresh_ratio * avg_h:
#                 # Kiểm tra gap ngang
#                 avg_w = (max_x - min_x + max_xj - min_xj) / 2.0
#                 gap_x = min_xj - max_x
#                 if 0 <= gap_x < line_x_gap_ratio * avg_w:
#                     used[j] = True
#                     group_text += " " + str(text_j)
#                     group_score += float(score_j)
#                     group_count += 1
#                     min_x = min(min_x, min_xj)
#                     max_x = max(max_x, max_xj)
#                     min_y = min(min_y, min_yj)
#                     max_y = max(max_y, max_yj)

#         merged_box = np.array([
#             [min_x, min_y],
#             [max_x, min_y],
#             [max_x, max_y],
#             [min_x, max_y]
#         ], dtype=np.float32)
#         merged.append((merged_box, (group_text, group_score / group_count)))

#     if merged:
#         merged_boxes, merged_texts = zip(*merged)
#         return list(merged_boxes), list(merged_texts)
#     else:
#         return [], []


# class OpenOCR(object):
#     def __init__(self,
#                  cfg_det_path: str,
#                  cfg_rec_path: str,
#                  drop_score: float = 0.5,
#                  det_box_type: str = 'quad',
#                  det_batch_size: int = 1,
#                  rec_batch_size: int = 6):
#         # Load model
#         cfg_det = Config(cfg_det_path).cfg
#         cfg_rec = Config(cfg_rec_path).cfg

#         self.text_detector = OpenDetector(cfg_det)
#         self.text_recognizer = OpenRecognizer(cfg_rec)
#         self.det_box_type = det_box_type
#         self.drop_score = drop_score
#         self.det_batch_size = det_batch_size
#         self.rec_batch_size = rec_batch_size

#     def infer_batch_image_det(self,
#                               img_numpy_list: List[np.ndarray]
#                               ) -> Tuple[List[List[np.ndarray]], List[Dict[str, float]]]:
#         """Phát hiện text (detection) cho list ảnh."""
#         all_dt_boxes = []
#         all_time_dicts = []

#         # Xử lý theo batch
#         for i in range(0, len(img_numpy_list), self.det_batch_size):
#             batch_imgs = img_numpy_list[i: i + self.det_batch_size]
#             batch_results = self.text_detector(img_numpy_list=batch_imgs)

#             for det_result in batch_results:
#                 dt_boxes = det_result.get('boxes', [])
#                 time_dict = {'detection_time': det_result.get('elapse', 0.0)}

#                 if dt_boxes is not None and len(dt_boxes) > 0:
#                     dt_boxes = sorted_boxes(dt_boxes)
#                 else:
#                     dt_boxes = []
#                 all_dt_boxes.append(dt_boxes)
#                 all_time_dicts.append(time_dict)

#         return all_dt_boxes, all_time_dicts

#     def infer_batch_image_rec(self,
#                               img_crop_list: List[Image.Image]
#                               ) -> Tuple[List[List[Union[str, float, float]]], float]:
#         """
#         Nhận dạng text (recognition) cho list ảnh đã crop.
#         Trả về mảng cùng kích thước với img_crop_list, mỗi phần tử: [text, score, time_cost].
#         """
#         rec_res_full = []
#         total_rec_time = 0.0

#         for i in range(0, len(img_crop_list), self.rec_batch_size):
#             batch_imgs = img_crop_list[i: i + self.rec_batch_size]
#             batch_results = self.text_recognizer(img_numpy_list=batch_imgs)

#             for r in batch_results:
#                 text = r.get('text', '')
#                 score = r.get('score', 0.0)
#                 elapse = r.get('elapse', 0.0)
#                 total_rec_time += elapse
#                 # Lưu tất cả kết quả (kể cả score thấp), để giữ thứ tự index
#                 rec_res_full.append([text, score, elapse])

#         return rec_res_full, total_rec_time

#     def infer_single_image(
#             self,
#             img_numpy: np.ndarray,
#             crop_infer: bool = False
#     ) -> Tuple[Optional[List[np.ndarray]],
#                Optional[List[List[Union[str, float]]]],
#                Optional[Dict[str, float]]]:
#         """Chạy OCR trên 1 ảnh đơn."""
#         if img_numpy is None:
#             return None, None, None

#         # Detection
#         all_dt_boxes, all_time_dicts = self.infer_batch_image_det([img_numpy])
#         dt_boxes = all_dt_boxes[0]
#         det_time_cost = all_time_dicts[0]['detection_time']

#         if not dt_boxes:
#             return None, None, None

#         # Crop
#         img_crop_list = []
#         for box in dt_boxes:
#             box_np = np.array(box).astype(np.float32)
#             if self.det_box_type == 'quad':
#                 img_crop = get_rotate_crop_image(img_numpy, box_np)
#             else:
#                 img_crop = get_minarea_rect_crop(img_numpy, box_np)
#             img_crop_list.append(Image.fromarray(img_crop))

#         # Recognition
#         rec_res_full, total_rec_time = self.infer_batch_image_rec(img_crop_list)

#         # rec_res_full có length = len(dt_boxes)
#         # => Bắt đầu lọc theo drop_score
#         filter_boxes = []
#         filter_rec_res = []
#         for box_i, rec_i in zip(dt_boxes, rec_res_full):
#             text_i, score_i, _ = rec_i
#             if score_i >= self.drop_score and text_i.strip():
#                 filter_boxes.append(box_i)
#                 filter_rec_res.append([text_i, score_i])

#         if not filter_boxes:
#             return None, None, None

#         # Timing
#         time_dict = {
#             'time_cost': det_time_cost + total_rec_time,
#             'detection_time': det_time_cost,
#             'recognition_time': total_rec_time,
#             'avg_rec_time_cost': total_rec_time / len(dt_boxes) if len(dt_boxes) else 0.0
#         }
#         return filter_boxes, filter_rec_res, time_dict

#     def infer_batch_image(
#             self,
#             img_numpy_list: List[np.ndarray]
#     ) -> List[Tuple[List[np.ndarray], List[List[Union[str, float]]], Dict[str, float]]]:
#         """Chạy OCR cho nhiều ảnh (batch)."""
#         all_results = []

#         # 1) Detection batch
#         all_dt_boxes, all_time_dicts = self.infer_batch_image_det(img_numpy_list)

#         # 2) Cho từng ảnh, crop rồi recognition
#         for idx, dt_boxes in enumerate(all_dt_boxes):
#             if not dt_boxes:
#                 all_results.append(([], [], {}))
#                 continue

#             img_numpy = img_numpy_list[idx]
#             det_time_cost = all_time_dicts[idx]['detection_time']

#             # Crop
#             img_crop_list = []
#             for box in dt_boxes:
#                 box_np = np.array(box).astype(np.float32)
#                 if self.det_box_type == 'quad':
#                     img_crop = get_rotate_crop_image(img_numpy, box_np)
#                 else:
#                     img_crop = get_minarea_rect_crop(img_numpy, box_np)
#                 img_crop_list.append(Image.fromarray(img_crop))

#             # Recognition
#             rec_res_full, total_rec_time = self.infer_batch_image_rec(img_crop_list)

#             if not rec_res_full:
#                 all_results.append(([], [], {}))
#                 continue

#             # Lọc theo drop_score
#             filter_boxes = []
#             filter_rec_res = []
#             for box_i, rec_i in zip(dt_boxes, rec_res_full):
#                 text_i, score_i, _ = rec_i
#                 if score_i >= self.drop_score and text_i.strip():
#                     filter_boxes.append(box_i)
#                     filter_rec_res.append([text_i, score_i])

#             avg_rec_time = total_rec_time / len(dt_boxes) if len(dt_boxes) else 0.0
#             time_dict = {
#                 'time_cost': det_time_cost + total_rec_time,
#                 'detection_time': det_time_cost,
#                 'recognition_time': total_rec_time,
#                 'avg_rec_time_cost': avg_rec_time
#             }
#             all_results.append((filter_boxes, filter_rec_res, time_dict))

#         return all_results


# def process_images(
#         ocr_engine: OpenOCR,
#         image_paths: List[str],
#         roi: Optional[List[int]] = None,
#         line_y_thresh: Optional[float] = 0.5,
#         line_x_gap: Optional[float] = 0.3
# ) -> List[Dict[str, Any]]:
#     """Chạy OCR cho nhiều ảnh, có thể cắt ROI rồi merge dòng."""
#     images = []
#     valid_image_paths = []
#     for path in image_paths:
#         img = cv2.imread(path)
#         if img is not None:
#             images.append(img)
#             valid_image_paths.append(path)
#         else:
#             logger.error(f"Cannot read image: {path}")

#     if not images:
#         logger.error("No images were successfully loaded.")
#         return []

#     images_with_offset = []
#     offsets = []
#     for img in images:
#         img_cropped, offset_x, offset_y = _crop_with_offset(img, roi)
#         images_with_offset.append(img_cropped)
#         offsets.append((offset_x, offset_y))

#     # OCR batch
#     batch_results = ocr_engine.infer_batch_image(images_with_offset)
#     final_result = []

#     for i, (dt_boxes, rec_res, _) in enumerate(batch_results):
#         this_path = valid_image_paths[i]
#         if not dt_boxes or not rec_res:
#             final_result.append({"image_path": this_path, "results": []})
#             continue

#         offset_x, offset_y = offsets[i]

#         # Merge lines (nếu có threshold)
#         if line_y_thresh is not None and line_x_gap is not None:
#             dt_boxes, rec_res = same_line_merge(dt_boxes, rec_res,
#                                                 line_y_thresh, line_x_gap)

#         image_results = []
#         for box, (text, score) in zip(dt_boxes, rec_res):
#             x, y, w, h = get_box_xywh(box)
#             x += offset_x
#             y += offset_y
#             image_results.append({
#                 "text": str(text),
#                 "score": float(score),
#                 "box": [float(x), float(y), float(w), float(h)]
#             })
#         final_result.append({"image_path": this_path, "results": image_results})

#     return final_result


# def process_video(
#         ocr_engine: OpenOCR,
#         video_path: str,
#         roi: Optional[List[int]] = None,
#         line_y_thresh: float = 0.5,
#         line_x_gap: float = 0.3,
#         do_merge: bool = True,
#         iou_threshold: float = 0.5,
#         vanish_time: float = 2.0,
#         min_interval: float = 5.0,
#         sec_skip: float = 1.0
# ) -> List[Dict[str, Union[float, List[Dict[str, Union[str, float, List[float]]]]]]]:
#     """
#     Xử lý video với OCR, dùng thời gian video (current_sec) để:
#     - skip frame (sec_skip)
#     - vanish text sau vanish_time giây không xuất hiện
#     - min_interval: khoảng cách tối thiểu giữa 2 lần output cùng 1 text
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logger.error(f"Cannot open video: {video_path}")
#         return []

#     video_results = []
#     active_texts: List[Dict[str, Any]] = []
#     last_ocr_time = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#         # Skip
#         if (current_sec - last_ocr_time) < sec_skip:
#             continue

#         last_ocr_time = current_sec
#         # Crop ROI
#         frame_cropped, offset_x, offset_y = _crop_with_offset(frame, roi)

#         # OCR
#         dt_boxes, rec_res, _ = ocr_engine.infer_single_image(frame_cropped)
#         if dt_boxes is None or rec_res is None:
#             # Không detect được gì
#             # Cũng nên xóa các text cũ trong active_texts
#             active_texts = [t for t in active_texts
#                             if (current_sec - t['last_seen']) < vanish_time]
#             continue

#         # Cập nhật active_texts => remove vanished
#         active_texts = [
#             t for t in active_texts
#             if (current_sec - t['last_seen']) < vanish_time
#         ]

#         # Merge line nếu cần
#         if do_merge:
#             dt_boxes, rec_res = same_line_merge(dt_boxes, rec_res,
#                                                 line_y_thresh, line_x_gap)

#         frame_texts: List[Dict[str, Union[str, float, List[float]]]] = []

#         # Xử lý logic tránh lặp text
#         for box, (text, score) in zip(dt_boxes, rec_res):
#             x, y, w, h = get_box_xywh(box)
#             x += offset_x
#             y += offset_y
#             new_box_xywh = (x, y, w, h)

#             matched_idx = -1
#             for i, atext in enumerate(active_texts):
#                 if iou(atext['box'], new_box_xywh) > iou_threshold \
#                    and atext['text'] == text:
#                     matched_idx = i
#                     break

#             if matched_idx >= 0:
#                 # Đã thấy text này => update
#                 active_texts[matched_idx]['last_seen'] = current_sec
#                 # check xem đã đủ thời gian để output lại chưa
#                 if (current_sec - active_texts[matched_idx]['last_output_time']) >= min_interval:
#                     frame_texts.append({
#                         "text": str(text),
#                         "score": float(score),
#                         "box": [float(x), float(y), float(w), float(h)]
#                     })
#                     active_texts[matched_idx]['last_output_time'] = current_sec
#             else:
#                 # Text mới
#                 frame_texts.append({
#                     "text": str(text),
#                     "score": float(score),
#                     "box": [float(x), float(y), float(w), float(h)]
#                 })
#                 active_texts.append({
#                     'text': str(text),
#                     'box': new_box_xywh,
#                     'first_seen': current_sec,
#                     'last_seen': current_sec,
#                     'last_output_time': current_sec
#                 })

#         if frame_texts:
#             video_results.append({"timestamp": current_sec, "texts": frame_texts})

#     cap.release()
#     return video_results


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



def process_images(
    ocr_engine: OpenOCR,
    image_list: List[Union[str, np.ndarray, Image.Image]],
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                logger.warning(f"Image file not found: {img_src}")
                processed_images.append(None)
                original_images.append(None)
                continue
        elif isinstance(img_src, np.ndarray):
            img = img_src.copy()
            # Convert BGR to RGB if needed
            if img.shape[2] == 3 and img.dtype == np.uint8:
                # Check if the image is in BGR format (common with OpenCV)
                # This is a heuristic and might not always be correct
                if np.mean(img[:,:,0]) < np.mean(img[:,:,2]):  # Typically blue channel has lower values than red
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_src, Image.Image):
            img = np.array(img_src)
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
        return [{} for _ in range(total_images)]
    
    # Run text detection in batch
    t_det = time.time()
    all_dt_boxes, _ = ocr_engine.infer_batch_image_det(valid_processed_images)
    logger.info(f"Detection completed in {time.time() - t_det:.2f}s")
    
    # Process results for each image
    all_results = [{} for _ in range(total_images)]  # Initialize with empty dict to maintain indices
    
    for i, (img_idx, dt_boxes) in enumerate(zip(valid_indices, all_dt_boxes)):
        # Get original image for this index
        orig_img = original_images[img_idx]
        
        # Skip if no boxes detected
        if dt_boxes is None or len(dt_boxes) == 0:
            all_results[img_idx] = {"texts": []}
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
            cv2.imwrite(out_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
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
                cv2.imwrite(out_path, cv2.cvtColor(merge_debug_img, cv2.COLOR_RGB2BGR))
        else:
            process_boxes = filtered_boxes
        
        # Create crops for recognition
        crops = []
        crop_boxes = []
        
        for j, box in enumerate(process_boxes):
            # Get tight crop around text
            text_crop = get_text_crop(orig_img, box)
            
            # Save debug image if needed
            if debug_box_dir:
                debug_path = os.path.join(debug_box_dir, f"image_{img_idx:04d}_box_{j:03d}.jpg")
                cv2.imwrite(debug_path, cv2.cvtColor(text_crop, cv2.COLOR_RGB2BGR))
            
            # Enhance the crop for recognition
            enhanced_crop = enhance_text_image(text_crop)
            
            # Add to crops list
            crops.append(Image.fromarray(enhanced_crop))
            crop_boxes.append(box)
        
        # Skip if no crops
        if not crops:
            all_results[img_idx] = {"texts": []}
            continue
        
        # Perform text recognition
        t_rec = time.time()
        rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
        logger.debug(f"Recognition for image {img_idx} completed in {time.time() - t_rec:.2f}s")
        
        # Collect recognition results
        texts = []
        for j, (box, rec_result) in enumerate(zip(crop_boxes, rec_res_all)):
            text, score, _ = rec_result
            
            # Only keep results with high score and non-empty text
            if score >= ocr_engine.drop_score and text.strip():
                texts.append({
                    "text": text,
                    "score": float(score),
                    "box": get_box_xywh(box)
                })
        
        # Sort texts from top to bottom
        texts = sorted(texts, key=lambda x: x["box"][1])
        
        # Save result for this image
        all_results[img_idx] = {"texts": texts}
    
    # Ensure all images have results
    for i in range(total_images):
        if i not in valid_indices:
            all_results[i] = {"texts": []}
    
    logger.info(f"Processed {total_images} images in {time.time() - t_start:.2f}s")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='OpenOCR system with time-based skipping.')
    parser.add_argument('--img_paths', type=str, nargs='+', help='Paths to multiple input images.')
    parser.add_argument('--video_path', type=str, help='Path to an input video.')
    parser.add_argument('--cfg_det_path', type=str, default="configs/det/dbnet/repvit_db.yml",
                        help='Path to the detection config (YAML).')
    parser.add_argument('--cfg_rec_path', type=str, default="configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml",
                        help='Path to the recognition config (YAML).')
    parser.add_argument('--drop_score', type=float, default=0.9, help='Recognition score threshold.')
    parser.add_argument('--output_json', type=str, default='output_results.json', help='Path to output JSON file.')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI in "x1,y1,x2,y2" format. Full image/video if not set.')
    parser.add_argument('--det_batch_size', type=int, default=1, help='Batch size for detection.')
    parser.add_argument('--rec_batch_size', type=int, default=6, help='Batch size for recognition.')
    parser.add_argument('--line_y_thresh', type=float, default=0.5,
                        help='Vertical ratio threshold for line merging.')
    parser.add_argument('--line_x_gap', type=float, default=0.3,
                        help='Horizontal gap ratio threshold for line merging.')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='IoU threshold for matching bounding boxes.')
    parser.add_argument('--vanish_time', type=float, default=2.0,
                        help='Time (seconds) after which unseen texts are removed.')
    parser.add_argument('--min_interval', type=float, default=5.0,
                        help='Minimum time (seconds) for re-output of a text.')
    parser.add_argument('--sec_skip', type=float, default=2.0,
                        help='Skip threshold (seconds) for video frame processing.')
    parser.add_argument('--do_merge', action='store_true', default=True,
                        help='Whether to merge bounding boxes on the same line.')

    args = parser.parse_args()

    # Kiểm tra input
    if (args.img_paths is None and args.video_path is None) or \
       (args.img_paths is not None and args.video_path is not None):
        raise ValueError("Must provide either --img_paths (multiple images) OR --video_path (single).")

    roi = parse_roi(args.roi)

    # Khởi tạo OCR
    ocr_engine = OpenOCR(cfg_det_path=args.cfg_det_path,
                         cfg_rec_path=args.cfg_rec_path,
                         drop_score=args.drop_score,
                         det_batch_size=args.det_batch_size,
                         rec_batch_size=args.rec_batch_size)

    if args.img_paths:
        valid_image_paths = []
        images = []
        for path in args.img_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                valid_image_paths.append(path)
            else:
                logger.error(f"Cannot read image: {path}")
        # Nhiều ảnh
        final_result = process_images(ocr_engine=ocr_engine,
                                      image_list=images,
                                      roi=roi,
                                      line_y_thresh=args.line_y_thresh,
                                      line_x_gap=args.line_x_gap)
    else:
        # Video
        final_result = process_video(
            ocr_engine=ocr_engine,
            video_path=args.video_path,
            roi=roi,
            line_y_thresh=args.line_y_thresh,
            line_x_gap=args.line_x_gap,
            do_merge=args.do_merge,
            iou_threshold=args.iou_thresh,
            vanish_time=args.vanish_time,
            min_interval=args.min_interval,
            sec_skip=args.sec_skip
        )

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output_json}.")


if __name__ == '__main__':
    main()
