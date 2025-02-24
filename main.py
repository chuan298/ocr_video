#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import math
import time
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union, Any
import Levenshtein
import multiprocessing
import queue  # dùng cho exception queue.Empty trong main process

# Thêm thư mục cha vào sys.path nếu cần
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from tools.utils.logging import get_logger

logger = get_logger()

###############################################################################
# Các hàm hỗ trợ chung
###############################################################################

def parse_roi(roi_str: str) -> Optional[List[float]]:
    """
    Parse chuỗi 'x1,y1,x2,y2' thành list [x1, y1, x2, y2] dưới dạng phần trăm hoặc tỉ lệ.
    Nếu giá trị > 1 (ví dụ: 10,20,90,80) sẽ được chia cho 100.
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

# def _crop_with_offset(frame: np.ndarray, roi: Optional[List[float]]) -> Tuple[np.ndarray, int, int]:
#     """
#     Cắt ROI khỏi frame theo tỉ lệ phần trăm kích thước ảnh.
#     Trả về (ảnh_cropped, offset_x, offset_y) để khôi phục tọa độ box trên ảnh gốc.
#     """
#     if roi is None:
#         return frame, 0, 0
#     x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi
#     h, w = frame.shape[:2]
#     x1 = int(np.clip(x1_ratio * w, 0, w))
#     x2 = int(np.clip(x2_ratio * w, 0, w))
#     y1 = int(np.clip(y1_ratio * h, 0, h))
#     y2 = int(np.clip(y2_ratio * h, 0, h))
#     if x2 <= x1 or y2 <= y1:
#         return frame, 0, 0
#     cropped = frame[y1:y2, x1:x2]
#     return cropped, x1, y1

def _crop_with_offset(frame: np.ndarray, roi: Optional[List[float]]) -> Tuple[np.ndarray, int, int]:
    if roi is None:
        return frame, 0, 0
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi
    h, w = frame.shape[:2]
    x1 = int(np.clip(x1_ratio * w, 0, w))
    x2 = int(np.clip(x2_ratio * w, 0, w))
    y1 = int(np.clip(y1_ratio * h, 0, h))
    y2 = int(np.clip(y2_ratio * h, 0, h))
    if x2 <= x1 or y2 <= y1:
        return frame, 0, 0
    # Tạo ảnh đen có kích thước ban đầu
    padded = np.zeros_like(frame)
    # Copy vùng ROI của frame gốc vào đúng vị trí
    padded[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    # Vì ảnh padded giữ nguyên kích thước ban đầu, nên offset = 0
    return padded, 0, 0


def iou(boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float]) -> float:
    """
    Tính IoU giữa 2 box (định dạng: x, y, w, h).
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x_start = max(xA, xB)
    y_start = max(yA, yB)
    x_end = min(xA + wA, xB + wB)
    y_end = min(yA + hA, yB + hB)
    inter = max(0, x_end - x_start) * max(0, y_end - y_start)
    areaA = wA * hA
    areaB = wB * hB
    return inter / float(areaA + areaB - inter + 1e-6)

def get_box_xywh(quad: Union[List[List[float]], np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Chuyển 4 điểm (quad) thành box dạng (x, y, w, h).
    """
    quad = np.array(quad)
    x_min, y_min = np.min(quad, axis=0)
    x_max, y_max = np.max(quad, axis=0)
    return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

def sorted_boxes(dt_boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sắp xếp các box từ trên xuống dưới, trái sang phải.
    """
    if dt_boxes is None or len(dt_boxes) <= 0:
        return []
    boxes = np.array(dt_boxes)
    sorted_indices = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
    sorted_boxes_arr = boxes[sorted_indices]
    # Nếu y của 2 box gần nhau, sắp xếp lại theo x
    for i in range(len(sorted_boxes_arr) - 1):
        if abs(sorted_boxes_arr[i+1,0,1] - sorted_boxes_arr[i,0,1]) < 10:
            if sorted_boxes_arr[i+1,0,0] < sorted_boxes_arr[i,0,0]:
                sorted_boxes_arr[[i, i+1]] = sorted_boxes_arr[[i+1, i]]
    return list(sorted_boxes_arr)

###############################################################################
# Hàm merge các box thành các dòng (lines)
###############################################################################

def same_line_merge(
    dt_boxes: List[np.ndarray],
    line_y_thresh_ratio: float = 0.5,
    line_x_gap_ratio: float = 0.3
) -> List[np.ndarray]:
    """
    Nhóm các box thành các dòng (lines) dựa trên sự gần nhau theo trục Y và khoảng cách theo trục X.
    Mỗi line được sắp xếp theo thứ tự tăng dần theo tọa độ X, sau đó gộp thành một box (đường bao) duy nhất.
    
    Args:
        dt_boxes: Danh sách các box (mỗi box là np.ndarray, shape (4,2)).
        line_y_thresh_ratio: Ngưỡng chênh lệch tâm Y để xem box có cùng 1 line hay không.
        line_x_gap_ratio: Tỉ lệ khoảng cách cho phép theo trục X giữa các box.
    
    Returns:
        merged_boxes: Danh sách các box sau khi gộp theo từng dòng.
    """
    if not dt_boxes:
        return []
    
    boxes_sorted = sorted(dt_boxes, key=lambda box: (np.min(box[:, 1]), np.min(box[:, 0])))
    lines = []  # Mỗi phần tử là list các box nằm cùng một dòng

    for box in boxes_sorted:
        center_y = (np.min(box[:, 1]) + np.max(box[:, 1])) / 2.0
        height = np.max(box[:, 1]) - np.min(box[:, 1])
        added = False
        for line in lines:
            line_centers = [ (np.min(b[:, 1]) + np.max(b[:, 1])) / 2.0 for b in line ]
            avg_center = sum(line_centers) / len(line_centers)
            line_heights = [ np.max(b[:, 1]) - np.min(b[:, 1]) for b in line ]
            avg_height = sum(line_heights) / len(line_heights)
            if abs(center_y - avg_center) <= line_y_thresh_ratio * avg_height:
                sorted_line = sorted(line, key=lambda b: np.min(b[:, 0]))
                last_box = sorted_line[-1]
                gap = np.min(box[:, 0]) - np.max(last_box[:, 0])
                line_widths = [ np.max(b[:, 0]) - np.min(b[:, 0]) for b in line ]
                avg_width = sum(line_widths) / len(line_widths)
                if gap < line_x_gap_ratio * avg_width:
                    line.append(box)
                    added = True
                    break
        if not added:
            lines.append([box])
    
    merged_boxes = []
    for line in lines:
        sorted_line = sorted(line, key=lambda b: np.min(b[:, 0]))
        min_x = min(np.min(b[:, 0]) for b in sorted_line)
        min_y = min(np.min(b[:, 1]) for b in sorted_line)
        max_x = max(np.max(b[:, 0]) for b in sorted_line)
        max_y = max(np.max(b[:, 1]) for b in sorted_line)
        merged_box = np.array([[min_x, min_y],
                                [max_x, min_y],
                                [max_x, max_y],
                                [min_x, max_y]], dtype=np.float32)
        merged_boxes.append(merged_box)
    
    return merged_boxes

###############################################################################
# Lọc box theo kích thước của ảnh crop (ROI)
###############################################################################

# def filter_boxes_by_size(boxes: List[np.ndarray],
#                          img_shape: Tuple[int, int],
#                          roi: Optional[List[float]] = None,
#                          config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
#     if config is None:
#         config = {
#             'min_w_ratio': 0.01,
#             'min_h_ratio': 0.01,
#             'max_w_ratio': 0.9,
#             'max_h_ratio': 0.9
#         }
#     min_w_ratio = config.get('min_w_ratio', 0.01)
#     min_h_ratio = config.get('min_h_ratio', 0.01)
#     max_w_ratio = config.get('max_w_ratio', 0.9)
#     max_h_ratio = config.get('max_h_ratio', 0.9)

#     filtered_boxes = []
#     img_h, img_w = img_shape

#     if roi is not None:
#         roi_x1, roi_y1, roi_x2, roi_y2 = roi
#         roi_w_ratio = roi_x2 - roi_x1
#         roi_h_ratio = roi_y2 - roi_y1
#     else:
#         roi_w_ratio = 1.0
#         roi_h_ratio = 1.0

#     min_w_ratio_roi = min_w_ratio / roi_w_ratio
#     min_h_ratio_roi = min_h_ratio / roi_h_ratio
#     max_w_ratio_roi = max_w_ratio / roi_w_ratio
#     max_h_ratio_roi = max_h_ratio / roi_h_ratio

#     for box in boxes:
#         x, y, bw, bh = get_box_xywh(box)
#         if (bw / img_w >= min_w_ratio_roi and bw / img_w <= max_w_ratio_roi and
#             bh / img_h >= min_h_ratio_roi and bh / img_h <= max_h_ratio_roi):
#             filtered_boxes.append(box)
#     return filtered_boxes

def filter_boxes_by_size(boxes: List[np.ndarray],
                         img_shape: Tuple[int, int],
                         roi: Optional[List[float]] = None,
                         config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    if config is None:
        config = {
            'min_w_ratio': 0.01,
            'min_h_ratio': 0.01,
            'max_w_ratio': 0.9,
            'max_h_ratio': 0.9
        }
    min_w_ratio = config.get('min_w_ratio', 0.01)
    min_h_ratio = config.get('min_h_ratio', 0.01)
    max_w_ratio = config.get('max_w_ratio', 0.9)
    max_h_ratio = config.get('max_h_ratio', 0.9)

    filtered_boxes = []
    img_h, img_w = img_shape

    if roi is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        roi_w_ratio = roi_x2 - roi_x1
        roi_h_ratio = roi_y2 - roi_y1
    else:
        roi_w_ratio = 1.0
        roi_h_ratio = 1.0

    min_w_ratio_roi = min_w_ratio / roi_w_ratio
    min_h_ratio_roi = min_h_ratio / roi_h_ratio
    max_w_ratio_roi = max_w_ratio / roi_w_ratio
    max_h_ratio_roi = max_h_ratio / roi_h_ratio

    for box in boxes:
        x, y, bw, bh = get_box_xywh(box)
        if (bw / img_w >= min_w_ratio_roi and bw / img_w <= max_w_ratio_roi and
            bh / img_h >= min_h_ratio_roi and bh / img_h <= max_h_ratio_roi):
            filtered_boxes.append(box)
    return filtered_boxes


###############################################################################
# Hàm tiền xử lý crop (sử dụng get_rotate_crop_image)
###############################################################################

def preprocess_crop(orig_img: np.ndarray, box: np.ndarray,
                    pad_ratio: float = 0.1,
                    blur_kernel: Tuple[int,int] = (7,7),
                    morph_kernel_size: Tuple[int,int] = (3,3),
                    morph_iterations: int = 1) -> np.ndarray:
    """
    Mở rộng vùng crop theo pad_ratio rồi crop bằng get_rotate_crop_image.
    """
    x_min = int(np.min(box[:, 0]))
    y_min = int(np.min(box[:, 1]))
    x_max = int(np.max(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    width = x_max - x_min
    height = y_max - y_min
    pad_x = int(width * pad_ratio)
    pad_y = int(height * pad_ratio)
    
    ext_x1 = max(0, x_min - pad_x)
    ext_y1 = max(0, y_min - pad_y)
    ext_x2 = min(orig_img.shape[1], x_max + pad_x)
    ext_y2 = min(orig_img.shape[0], y_max + pad_y)
    
    ext_img = orig_img[ext_y1:ext_y2, ext_x1:ext_x2].copy()
    
    adjusted_box = box.copy()
    adjusted_box[:, 0] -= ext_x1
    adjusted_box[:, 1] -= ext_y1
    
    crop = get_rotate_crop_image(ext_img, adjusted_box.astype(np.float32))
    return crop

###############################################################################
# Định nghĩa class OpenOCR (detection + recognition)
###############################################################################

class OpenOCR(object):
    def __init__(self,
                 cfg_det_path: str,
                 cfg_rec_path: str,
                 drop_score: float = 0.5,
                 det_box_type: str = 'quad',
                 det_batch_size: int = 1,
                 rec_batch_size: int = 6):
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
        all_dt_boxes = []
        all_time_dicts = []
        for i in range(0, len(img_numpy_list), self.det_batch_size):
            batch_imgs = img_numpy_list[i : i + self.det_batch_size]
            batch_results = self.text_detector(img_numpy_list=batch_imgs)
            for det_res in batch_results:
                dt_boxes = det_res.get('boxes', [])
                elapse = det_res.get('elapse', 0.0)
                time_dict = {'detection_time': elapse}
                if dt_boxes is not None and len(dt_boxes) > 0:
                    dt_boxes = sorted_boxes(dt_boxes)
                else:
                    dt_boxes = []
                all_dt_boxes.append(dt_boxes)
                all_time_dicts.append(time_dict)
        return all_dt_boxes, all_time_dicts

    def infer_batch_image_rec(self,
                              img_crop_list: List[Image.Image]
                              ) -> Tuple[List[List[Union[str, float, float]]], float]:
        rec_res_full = []
        total_rec_time = 0.0
        for i in range(0, len(img_crop_list), self.rec_batch_size):
            batch_imgs = img_crop_list[i : i + self.rec_batch_size]
            batch_rec = self.text_recognizer(img_numpy_list=batch_imgs)
            for r in batch_rec:
                text = r.get('text', '')
                score = r.get('score', 0.0)
                elapse = r.get('elapse', 0.0)
                total_rec_time += elapse
                rec_res_full.append([text, score, elapse])
        return rec_res_full, total_rec_time

###############################################################################
# Hàm đọc video bằng multiprocessing (đọc đồng thời các segment)
###############################################################################

def read_video_segment(video_path: str, start_frame: int, end_frame: int,
                       skip_interval: int, roi: Optional[List[float]],
                       fps: float, output_queue: multiprocessing.Queue):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # Lấy frame theo khoảng cách
        if (current_frame % skip_interval) != 0:
            current_frame += 1
            continue
        current_sec = current_frame / fps
        original_frame = frame.copy()
        cropped, ox, oy = _crop_with_offset(frame, roi)
        output_queue.put((current_frame, cropped, ox, oy, original_frame, current_sec))
        current_frame += 1
    cap.release()
    output_queue.put(None)


def blur_and_reduce_contrast(image, kernel_size=15, sigmaX=0, contrast_alpha=0.3):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)
    processed_image = cv2.convertScaleAbs(blurred_image, alpha=contrast_alpha, beta=0)
    return processed_image


###############################################################################
# Hàm process_video: sử dụng multiprocessing để đọc video đồng thời
###############################################################################

def process_video(
    ocr_engine: OpenOCR,
    video_path: str,
    roi: Optional[List[float]] = None,
    line_y_thresh: float = 0.5,
    line_x_gap: float = 0.3,
    do_merge: bool = True,
    iou_threshold: float = 0.5,
    vanish_time: float = 2.0,
    min_interval: float = 5.0,
    sec_skip: float = 1.0,
    text_sim_threshold: float = 0.8,
    filter_config: Optional[Dict[str, Any]] = None,
    debug_det_dir: Optional[str] = None,
    debug_box_dir: Optional[str] = None,
    num_processes: int = 4
) -> List[Dict[str, Union[float, List[Dict[str, Union[str, float, List[float]]]]]]]:
    # Mở video để lấy fps và tổng số frame
    cap_temp = cv2.VideoCapture(video_path)
    if not cap_temp.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    # Tính số frame bỏ qua theo sec_skip (ví dụ: nếu fps=30 và sec_skip=2 -> skip_interval=60)
    skip_interval = max(1, int(round(sec_skip * fps)))
    
    # Sử dụng multiprocessing để đọc video đồng thời
    num_processes = num_processes
    segment_size = total_frames // num_processes
    processes = []
    queues = []
    
    t1 = time.time()
    for i in range(num_processes):
        q = multiprocessing.Queue(maxsize=512)
        start_frame = i * segment_size
        end_frame = total_frames if i == num_processes - 1 else (i + 1) * segment_size
        p = multiprocessing.Process(target=read_video_segment,
                                    args=(video_path, start_frame, end_frame,
                                          skip_interval, roi, fps, q))
        processes.append(p)
        queues.append(q)
        p.start()
    
    frames_data = []  # Mỗi phần tử: (cropped_frame, offset_x, offset_y, original_frame)
    timestamps = []
    finished_count = 0
    # Thu thập frame từ các tiến trình qua các Queue
    while finished_count < num_processes:
        for idx, q in enumerate(queues):
            if q is None:
                continue
            try:
                item = q.get(timeout=0.1)
                if item is None:
                    finished_count += 1
                    queues[idx] = None
                else:
                    # item: (frame_index, cropped, ox, oy, original_frame, current_sec)
                    _, cropped, ox, oy, original_frame, current_sec = item
                    frames_data.append((cropped, ox, oy, original_frame))
                    timestamps.append(current_sec)
            except queue.Empty:
                continue
        queues = [q for q in queues if q is not None]
    
    for p in processes:
        p.join()
    
    if not frames_data:
        logger.warning("No frames captured")
        return []
    
    print("time_read_video", time.time() - t1)
    # Sắp xếp các frame theo thời gian
    sorted_data = sorted(zip(timestamps, frames_data), key=lambda x: x[0])
    timestamps = [item[0] for item in sorted_data]
    frames_data = [item[1] for item in sorted_data]
    
    # 1) Batch detection trên ảnh crop (có thể làm mờ & giảm tương phản)
    all_images = [blur_and_reduce_contrast(x[0]) for x in frames_data]
    t_det = time.time()
    all_dt_boxes, _ = ocr_engine.infer_batch_image_det(all_images)
    print("time_det_process", time.time() - t_det)
    # Lọc box theo kích thước
    for i in range(len(all_dt_boxes)):
        all_dt_boxes[i] = filter_boxes_by_size(
            all_dt_boxes[i],
            frames_data[i][0].shape[:2],
            config=filter_config
        )
    
    # Debug: Lưu ảnh detection với các box
    if debug_det_dir:
        os.makedirs(debug_det_dir, exist_ok=True)
        for i, dt_boxes_i in enumerate(all_dt_boxes):
            dbg_img = frames_data[i][0].copy()
            offset_x = frames_data[i][1]
            offset_y = frames_data[i][2]
            for box_ in dt_boxes_i:
                poly = np.array(box_, dtype=np.int32).copy()
                poly[:, 0] += offset_x
                poly[:, 1] += offset_y
                cv2.polylines(dbg_img, [poly], isClosed=True, color=(0,255,0), thickness=2)
            if roi is not None:
                h_orig, w_orig = dbg_img.shape[:2]
                roi_x1 = int(roi[0] * w_orig)
                roi_y1 = int(roi[1] * h_orig)
                roi_x2 = int(roi[2] * w_orig)
                roi_y2 = int(roi[3] * h_orig)
                cv2.rectangle(dbg_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
            out_path = os.path.join(debug_det_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(out_path, dbg_img)
    
    # 2) Crop vùng nhận dạng cho từng frame
    crops = []
    frame_idx_of_crop = []  # lưu lại frame index ứng với mỗi crop
    crop_boxes = []         # lưu lại box (đã adjust offset) cho mỗi crop
    crop_counter = 0
    for i, dt_boxes_i in enumerate(all_dt_boxes):
        offset_x, offset_y = frames_data[i][1], frames_data[i][2]
        original_frame = frames_data[i][3]
        if do_merge:
            merged_dt_boxes = same_line_merge(dt_boxes_i, line_y_thresh_ratio=line_y_thresh, line_x_gap_ratio=line_x_gap)
            for merged_box in merged_dt_boxes:
                adjusted_box = merged_box.copy()
                adjusted_box[:, 0] += offset_x
                adjusted_box[:, 1] += offset_y
                processed_crop = preprocess_crop(original_frame, adjusted_box)
                if debug_box_dir:
                    os.makedirs(debug_box_dir, exist_ok=True)
                    debug_path = os.path.join(debug_box_dir, f"frame_{i:06d}_line_{crop_counter:03d}.jpg")
                    cv2.imwrite(debug_path, processed_crop)
                    crop_counter += 1
                crops.append(Image.fromarray(processed_crop))
                frame_idx_of_crop.append(i)
                crop_boxes.append(adjusted_box)
        else:
            for box in dt_boxes_i:
                adjusted_box = np.array(box, dtype=np.float32)
                adjusted_box[:, 0] += offset_x
                adjusted_box[:, 1] += offset_y
                processed_crop = preprocess_crop(original_frame, adjusted_box)
                if debug_box_dir:
                    os.makedirs(debug_box_dir, exist_ok=True)
                    debug_path = os.path.join(debug_box_dir, f"frame_{i:06d}_box_{crop_counter:03d}.jpg")
                    cv2.imwrite(debug_path, processed_crop)
                    crop_counter += 1
                crops.append(Image.fromarray(processed_crop))
                frame_idx_of_crop.append(i)
                crop_boxes.append(adjusted_box)
    
    t_rec = time.time()
    # 3) Batch recognition trên các crop
    rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
    print("time_rec_process", time.time() - t_rec)
    # Gán kết quả nhận dạng vào từng frame (dựa trên frame_idx_of_crop)
    final_texts_per_frame = [[] for _ in range(len(frames_data))]
    c_idx = 0
    for j, frame_idx in enumerate(frame_idx_of_crop):
        box = crop_boxes[j]
        text_j, score_j, _ = rec_res_all[c_idx]
        c_idx += 1
        if score_j >= ocr_engine.drop_score and text_j.strip():
            final_texts_per_frame[frame_idx].append({
                "text": text_j,
                "score": float(score_j),
                "box": get_box_xywh(box)
            })
    
    # 4) Post-process: Ghép kết quả theo thời gian và loại bỏ trùng lặp
    video_results = []
    active_texts: List[Dict[str, Any]] = []
    for i in range(len(frames_data)):
        current_sec = timestamps[i]
        offset_x, offset_y = frames_data[i][1], frames_data[i][2]
        active_texts = [t for t in active_texts if (current_sec - t['last_seen']) < vanish_time]
        rec_texts = final_texts_per_frame[i]
        frame_texts = []
        for item in rec_texts:
            x, y, w, h = item["box"]
            x += offset_x
            y += offset_y
            new_box_xywh = (x, y, w, h)
            matched_idx = -1
            best_sim = 0.0
            for idx_a, atext in enumerate(active_texts):
                iou_val = iou(atext['box'], new_box_xywh)
                sim_val = Levenshtein.ratio(
                    atext['text'].replace(" ", ""),
                    item['text'].replace(" ", "")
                )
                if iou_val > iou_threshold or sim_val >= text_sim_threshold:
                    if sim_val > best_sim:
                        best_sim = sim_val
                        matched_idx = idx_a
            if matched_idx >= 0:
                active_texts[matched_idx]['last_seen'] = current_sec
                if (current_sec - active_texts[matched_idx]['last_output_time']) >= min_interval:
                    frame_texts.append({
                        "text": item["text"],
                        "score": float(item["score"]),
                        "box": new_box_xywh
                    })
                    active_texts[matched_idx]['last_output_time'] = current_sec
            else:
                frame_texts.append({
                    "text": item["text"],
                    "score": float(item["score"]),
                    "box": new_box_xywh
                })
                active_texts.append({
                    'text': item["text"],
                    'box': new_box_xywh,
                    'first_seen': current_sec,
                    'last_seen': current_sec,
                    'last_output_time': current_sec
                })
        if frame_texts:
            video_results.append({
                "timestamp": current_sec,
                "texts": frame_texts
            })
    return video_results

###############################################################################
# Hàm loại bỏ subtitle trùng nhau ở các frame liên tiếp
###############################################################################

def remove_subtitle_duplicates_consecutive_frames(
    video_results,
    iou_thresh=0.5,
    text_sim_thresh=0.9
):
    def local_iou(boxA, boxB):
        xA, yA, wA, hA = boxA
        xB, yB, wB, hB = boxB
        x_start = max(xA, xB)
        y_start = max(yA, yB)
        x_end = min(xA + wA, xB + wB)
        y_end = min(yA + hA, yB + hB)
        inter = max(0, x_end - x_start) * max(0, y_end - y_start)
        areaA = wA * hA
        areaB = wB * hB
        return inter / float(areaA + areaB - inter + 1e-9)

    i = 0
    while i < len(video_results) - 1:
        prev_frame = video_results[i]
        next_frame = video_results[i + 1]
        prev_texts = prev_frame["texts"]
        next_texts = next_frame["texts"]
        new_next_texts = []
        for nt in next_texts:
            textN = nt["text"]
            boxN = nt["box"]
            is_duplicate = False
            for pt in prev_texts:
                textP = pt["text"]
                boxP = pt["box"]
                iou_val = local_iou(boxN, boxP)
                sim_val = Levenshtein.ratio(
                    textN.replace(" ", ""),
                    textP.replace(" ", "")
                )
                if iou_val >= iou_thresh and sim_val >= text_sim_thresh:
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_next_texts.append(nt)
        next_frame["texts"] = new_next_texts
        if len(new_next_texts) == 0:
            video_results.pop(i + 1)
        else:
            i += 1

###############################################################################
# Hàm tạo file SRT từ kết quả
###############################################################################

def generate_srt_from_results(
    video_results: List[Dict[str, Any]],
    min_gap_sec: float = 0.5,
    ext_time_sec: float = 1.0,
    output_srt: str = "output.srt"
):
    if not video_results:
        logger.warning("No results to generate SRT.")
        return
    subs = []
    for item in video_results:
        tsec = item["timestamp"]
        lines = [x["text"] for x in item["texts"]]
        joined_text = " | ".join(lines)
        subs.append((tsec, joined_text))
    subs.sort(key=lambda x: x[0])
    final_subs = []
    current_block = {
        "start": subs[0][0],
        "end": subs[0][0],
        "text": subs[0][1]
    }
    for i in range(1, len(subs)):
        tsec, text_line = subs[i]
        if (tsec - current_block["end"]) <= min_gap_sec:
            current_block["end"] = tsec
            if text_line not in current_block["text"]:
                current_block["text"] += (" " + text_line)
        else:
            final_subs.append(current_block)
            current_block = {
                "start": tsec,
                "end": tsec,
                "text": text_line
            }
    final_subs.append(current_block)
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, block in enumerate(final_subs, start=1):
            start_s = block["start"]
            end_s = block["end"] + ext_time_sec
            def sec_to_hmsms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                ms = int((sec - int(sec)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            start_time_str = sec_to_hmsms(start_s)
            end_time_str = sec_to_hmsms(end_s)
            f.write(f"{idx}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{block['text']}\n\n")
    logger.info(f"SRT saved to: {output_srt}")

###############################################################################
# Hàm main() và chạy chương trình
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Video OCR with multiprocessing reading + SRT.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video.')
    parser.add_argument('--cfg_det_path', type=str, default="configs/det/dbnet/repvit_db.yml")
    parser.add_argument('--cfg_rec_path', type=str, default="configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml")
    parser.add_argument('--drop_score', type=float, default=0.9)
    parser.add_argument('--det_batch_size', type=int, default=32)
    parser.add_argument('--rec_batch_size', type=int, default=32)
    parser.add_argument('--sec_skip', type=float, default=0.5, help='Skip frames by seconds.')
    parser.add_argument('--line_y_thresh', type=float, default=0.5)
    parser.add_argument('--line_x_gap', type=float, default=0.3)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--vanish_time', type=float, default=2.0)
    parser.add_argument('--min_interval', type=float, default=5.0)
    parser.add_argument('--text_sim_threshold', type=float, default=0.8)
    parser.add_argument('--roi', type=str, default="0.1,0.6,0.9,0.97", help='ROI format: x1,y1,x2,y2 (theo % hoặc tỉ lệ)')
    parser.add_argument('--output_json', type=str, default='output_results.json')
    parser.add_argument('--output_srt', type=str, default='output_results.srt', help='Path to save SRT.')
    parser.add_argument('--generate_srt', action='store_true', default=True, help='Whether to generate SRT from results.')
    parser.add_argument('--remove_dup_iou_thresh', type=float, default=0.5)
    parser.add_argument('--remove_dup_text_sim_thresh', type=float, default=0.9)
    parser.add_argument('--debug_det_dir', type=str, default=None)
    parser.add_argument('--debug_box_dir', type=str, default=None)
    parser.add_argument('--min_w_ratio', type=float, default=0.02)
    parser.add_argument('--min_h_ratio', type=float, default=0.02)
    parser.add_argument('--max_w_ratio', type=float, default=0.9)
    parser.add_argument('--max_h_ratio', type=float, default=0.09)
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()

    roi = parse_roi(args.roi)
    filter_config = {
        'min_w_ratio': args.min_w_ratio,
        'min_h_ratio': args.min_h_ratio,
        'max_w_ratio': args.max_w_ratio,
        'max_h_ratio': args.max_h_ratio
    }

    ocr_engine = OpenOCR(
        cfg_det_path=args.cfg_det_path,
        cfg_rec_path=args.cfg_rec_path,
        drop_score=args.drop_score,
        det_box_type='quad',
        det_batch_size=args.det_batch_size,
        rec_batch_size=args.rec_batch_size
    )
    time_start = time.time()
    results = process_video(
        ocr_engine=ocr_engine,
        video_path=args.video_path,
        roi=roi,
        line_y_thresh=args.line_y_thresh,
        line_x_gap=args.line_x_gap,
        do_merge=True,
        iou_threshold=args.iou_thresh,
        vanish_time=args.vanish_time,
        min_interval=args.min_interval,
        sec_skip=args.sec_skip,
        text_sim_threshold=args.text_sim_threshold,
        filter_config=filter_config,
        debug_det_dir=args.debug_det_dir,
        debug_box_dir=args.debug_box_dir,
        num_processes=args.num_processes
    )

    remove_subtitle_duplicates_consecutive_frames(
        video_results=results,
        iou_thresh=args.remove_dup_iou_thresh,
        text_sim_thresh=args.remove_dup_text_sim_thresh
    )

    time_process = time.time() - time_start
    print("time_process", time_process)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output_json}")

    if args.generate_srt:
        generate_srt_from_results(results, min_gap_sec=0.5, ext_time_sec=1.0, output_srt=args.output_srt)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
