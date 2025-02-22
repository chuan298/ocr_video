

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from tools.utils.logging import get_logger

logger = get_logger()


###############################################################################
# Các hàm phụ trợ
###############################################################################

def parse_roi(roi_str: str) -> Optional[List[float]]:
    """
    Parse chuỗi 'x1,y1,x2,y2' thành list [x1, y1, x2, y2] dưới dạng phần trăm hoặc tỉ lệ.
    Nếu giá trị lớn hơn 1 (ví dụ: 10,20,90,80) sẽ được chia cho 100.
    """
    if not roi_str:
        return None
    try:
        vals = [float(v.strip()) for v in roi_str.split(',')]
        if len(vals) != 4:
            return None
        # Nếu có giá trị > 1, giả sử nhập theo % => chuyển về tỉ lệ (0-1)
        if max(vals) > 1:
            vals = [v / 100.0 for v in vals]
        return vals
    except ValueError:
        return None

def _crop_with_offset(frame: np.ndarray, roi: Optional[List[float]]) -> Tuple[np.ndarray, int, int]:
    """
    Cắt ROI khỏi frame dựa theo tỉ lệ phần trăm của kích thước ảnh.
    Ví dụ: nếu roi = [0.1, 0.2, 0.9, 0.8] với kích thước ảnh (w, h),
    thì vùng cắt là (x1=0.1*w, y1=0.2*h, x2=0.9*w, y2=0.8*h).
    Trả về (ảnh_cắt, offset_x, offset_y) để khôi phục tọa độ box trên ảnh gốc.
    """
    if roi is None:
        return frame, 0, 0
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi
    h, w = frame.shape[:2]
    
    x1 = int(np.clip(x1_ratio * w, 0, w))
    x2 = int(np.clip(x2_ratio * w, 0, w))
    y1 = int(np.clip(y1_ratio * h, 0, h))
    y2 = int(np.clip(y2_ratio * h, 0, h))
    
    if x2 <= x1 or y2 <= y1:
        # ROI không hợp lệ => trả nguyên frame
        return frame, 0, 0

    cropped = frame[y1:y2, x1:x2]
    return cropped, x1, y1

def iou(boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float]) -> float:
    """
    Tính IoU giữa 2 box: (x, y, w, h).
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
    Chuyển 4 điểm (quad) => (x, y, w, h).
    """
    quad = np.array(quad)
    x_min, y_min = np.min(quad, axis=0)
    x_max, y_max = np.max(quad, axis=0)
    return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

def sorted_boxes(dt_boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sắp xếp box từ trên xuống dưới, trái sang phải.
    """
    if dt_boxes is None or len(dt_boxes) <= 0:
        return []
    boxes = np.array(dt_boxes)
    # sort by y, x
    sorted_indices = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
    sorted_boxes = boxes[sorted_indices]
    # refine: nếu hai box có y gần nhau, sắp xếp theo x
    for i in range(len(sorted_boxes) - 1):
        if abs(sorted_boxes[i + 1, 0, 1] - sorted_boxes[i, 0, 1]) < 10:
            if sorted_boxes[i + 1, 0, 0] < sorted_boxes[i, 0, 0]:
                sorted_boxes[[i, i + 1]] = sorted_boxes[[i + 1, i]]
    return list(sorted_boxes)

def same_line_merge(
    dt_boxes: List[np.ndarray],
    rec_res: List[List[Union[str, float]]],
    line_y_thresh_ratio: float = 0.5,
    line_x_gap_ratio: float = 0.3
) -> Tuple[List[np.ndarray], List[List[Union[str, float]]]]:
    """
    Gộp các box nằm trên cùng dòng (theo khoảng cách).
    """
    if not dt_boxes:
        return [], []

    data = list(zip(dt_boxes, rec_res))
    data.sort(key=lambda x: (x[0][0, 1], x[0][0, 0]))
    merged = []
    used = [False] * len(data)

    for i in range(len(data)):
        if used[i]:
            continue
        box_i, (text_i, score_i) = data[i]
        used[i] = True
        box_np = np.array(box_i)
        min_x, min_y = np.min(box_np, axis=0)
        max_x, max_y = np.max(box_np, axis=0)
        group_text = text_i
        group_score = float(score_i)
        group_count = 1

        for j in range(i + 1, len(data)):
            if used[j]:
                continue
            box_j, (text_j, score_j) = data[j]
            box_j_np = np.array(box_j)
            min_xj, min_yj = np.min(box_j_np, axis=0)
            max_xj, max_yj = np.max(box_j_np, axis=0)

            avg_h = (max_y - min_y + max_yj - min_yj) / 2.0
            center_i_y = (min_y + max_y) / 2.0
            center_j_y = (min_yj + max_yj) / 2.0
            if abs(center_j_y - center_i_y) <= line_y_thresh_ratio * avg_h:
                avg_w = (max_x - min_x + max_xj - min_xj) / 2.0
                gap_x = min_xj - max_x
                if 0 <= gap_x < line_x_gap_ratio * avg_w:
                    used[j] = True
                    group_text += " " + text_j
                    group_score += float(score_j)
                    group_count += 1
                    min_x = min(min_x, min_xj)
                    max_x = max(max_x, max_xj)
                    min_y = min(min_y, min_yj)
                    max_y = max(max_y, max_yj)

        merged_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype=np.float32)
        merged.append((merged_box, (group_text, group_score / group_count)))

    if not merged:
        return [], []

    merged_boxes, merged_texts = zip(*merged)
    return list(merged_boxes), list(merged_texts)

def filter_boxes_by_size(boxes: List[np.ndarray],
                         img_shape: Tuple[int, int],
                         roi: Optional[List[float]] = None,
                         config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    """
    Lọc các box dựa trên tỉ lệ kích thước so với ROI (ảnh đã crop).

    Args:
        boxes: Danh sách các box (mỗi box là một numpy array).
        img_shape: Kích thước của ROI (ảnh đã crop) (height, width).
        roi: Tọa độ ROI (x1, y1, x2, y2) theo tỉ lệ (0.0 - 1.0).  Không dùng trong tính toán.
        config: Dictionary chứa các tham số cấu hình:
            - min_w_ratio (float): Tỉ lệ chiều rộng tối thiểu của box so với ROI.
            - min_h_ratio (float): Tỉ lệ chiều cao tối thiểu của box so với ROI.
            - max_w_ratio (float): Tỉ lệ chiều rộng tối đa của box so với ROI.
            - max_h_ratio (float): Tỉ lệ chiều cao tối đa của box so với ROI.

    Returns:
        Danh sách các box đã được lọc.
    """
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
    img_h, img_w = img_shape  # Kích thước ảnh *gốc*

    # Xác định tỉ lệ ROI so với ảnh gốc
    if roi is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        roi_w_ratio = roi_x2 - roi_x1
        roi_h_ratio = roi_y2 - roi_y1
    else:
        roi_w_ratio = 1.0
        roi_h_ratio = 1.0

    # Điều chỉnh tỉ lệ lọc theo ROI
    min_w_ratio_roi = min_w_ratio / roi_w_ratio
    min_h_ratio_roi = min_h_ratio / roi_h_ratio
    max_w_ratio_roi = max_w_ratio / roi_w_ratio
    max_h_ratio_roi = max_h_ratio / roi_h_ratio


    for box in boxes:
        x, y, bw, bh = get_box_xywh(box)  # Tọa độ và kích thước trên ảnh crop ROI

        # Kích thước box đã là trên ROI, không cần scale nữa
        scaled_bw = bw
        scaled_bh = bh
        # So sánh với tỉ lệ đã điều chỉnh theo ROI
        if (scaled_bw / img_w >= min_w_ratio_roi and scaled_bw / img_w <= max_w_ratio_roi and
            scaled_bh / img_h >= min_h_ratio_roi and scaled_bh / img_h <= max_h_ratio_roi):
            filtered_boxes.append(box)

    return filtered_boxes


###############################################################################
# Hàm tiền xử lý crop: preprocess_crop
###############################################################################

# def preprocess_crop(crop_img: np.ndarray,
#                     pad_ratio: float = 0.1,
#                     blur_kernel: Tuple[int, int] = (15, 15),
#                     morph_kernel_size: Tuple[int, int] = (3, 3),
#                     morph_iterations: int = 1) -> np.ndarray:
#     """
#     Tiền xử lý crop bằng cách mở rộng ảnh với border padding,
#     sau đó áp dụng Gaussian blur, erosion và dilation để làm mờ watermark.
    
#     Args:
#       crop_img: Ảnh crop (numpy array) từ hàm get_rotate_crop_image hoặc get_minarea_rect_crop.
#       pad_ratio: Tỉ lệ mở rộng (padding) theo kích thước crop.
#       blur_kernel: Kích thước kernel cho GaussianBlur.
#       morph_kernel_size: Kích thước kernel cho erosion và dilation.
#       morph_iterations: Số vòng lặp của các thao tác erosion/dilation.
      
#     Returns:
#       processed_crop: Ảnh crop đã được xử lý, kích thước giữ nguyên crop ban đầu.
#     """
#     h, w = crop_img.shape[:2]
#     pad_x = int(w * pad_ratio)
#     pad_y = int(h * pad_ratio)
    
#     # Mở rộng crop bằng cách thêm border (dùng BORDER_REFLECT để tránh viền đen)
#     extended = cv2.copyMakeBorder(crop_img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    
#     # Áp dụng Gaussian blur để làm giảm độ tương phản watermark
#     blurred = cv2.GaussianBlur(extended, blur_kernel, 0)
    
#     # Áp dụng các thao tác hình thái: erosion sau đó dilation
#     morph_kernel = np.ones(morph_kernel_size, np.uint8)
#     eroded = cv2.erode(blurred, morph_kernel, iterations=morph_iterations)
#     processed_ext = cv2.dilate(eroded, morph_kernel, iterations=morph_iterations)
    
#     # Cắt lại phần trung tâm có kích thước ban đầu
#     processed_crop = processed_ext[pad_y:pad_y+h, pad_x:pad_x+w]
#     return processed_crop

def preprocess_crop(orig_img: np.ndarray, box: np.ndarray,
                    pad_ratio: float = 0.1,
                    blur_kernel: Tuple[int,int] = (7,7),
                    morph_kernel_size: Tuple[int,int] = (3,3),
                    morph_iterations: int = 1) -> np.ndarray:
    """
    Tiền xử lý crop sử dụng ảnh gốc để mở rộng vùng lấy crop.
    Args:
      orig_img: Ảnh gốc (BGR).
      box: 4 điểm của box (numpy array, shape (4,2)) với tọa độ trên ảnh gốc.
      pad_ratio: Tỉ lệ mở rộng của box.
      blur_kernel: Kích thước kernel cho GaussianBlur (giảm xuống để không làm mất chi tiết text).
      morph_kernel_size, morph_iterations: Tham số cho thao tác morphology (ở đây không sử dụng thêm erosion/dilation nếu không cần).
    Returns:
      processed_crop: Ảnh crop sau tiền xử lý, đã được tăng cường tương phản (CLAHE) để làm nổi bật nền và giảm watermark.
    """
    # Tính bounding box axis-aligned của box
    x_min = int(np.min(box[:, 0]))
    y_min = int(np.min(box[:, 1]))
    x_max = int(np.max(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    width = x_max - x_min
    height = y_max - y_min
    pad_x = int(width * pad_ratio)
    pad_y = int(height * pad_ratio)
    
    # Tính tọa độ mở rộng trên ảnh gốc
    ext_x1 = max(0, x_min - pad_x)
    ext_y1 = max(0, y_min - pad_y)
    ext_x2 = min(orig_img.shape[1], x_max + pad_x)
    ext_y2 = min(orig_img.shape[0], y_max + pad_y)
    
    # Crop vùng mở rộng từ ảnh gốc
    ext_img = orig_img[ext_y1:ext_y2, ext_x1:ext_x2].copy()
    
    # Điều chỉnh tọa độ box theo vùng mở rộng
    adjusted_box = box.copy()
    adjusted_box[:, 0] -= ext_x1
    adjusted_box[:, 1] -= ext_y1
    
    # Sử dụng hàm get_rotate_crop_image để lấy crop theo adjusted_box
    crop = get_rotate_crop_image(ext_img, adjusted_box.astype(np.float32))
    
    # # Áp dụng Gaussian blur nhẹ
    # crop_blurred = cv2.GaussianBlur(crop, blur_kernel, 0)
    
    # # Tăng cường độ tương phản bằng CLAHE
    # lab = cv2.cvtColor(crop_blurred, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl = clahe.apply(l)
    # lab_enhanced = cv2.merge((cl, a, b))
    # processed_crop = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return crop


###############################################################################
# Định nghĩa Class OpenOCR (batch detect & batch recog)
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
        """
        Phát hiện văn bản (Detection) dạng batch.
        Trả về:
          - all_dt_boxes: List[List[box]] với độ dài = len(img_numpy_list).
          - all_time_dicts: Thời gian detect cho mỗi ảnh.
        """
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
        """
        Nhận dạng (Recognition) dạng batch.
        Trả về list[[text, score, elapse]] có chiều dài = len(img_crop_list).
        """
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
# remove_subtitle_duplicates_consecutive_frames
###############################################################################

def remove_subtitle_duplicates_consecutive_frames(
    video_results,
    iou_thresh=0.5,
    text_sim_thresh=0.9
):
    """
    - So sánh cặp frame liên tiếp (i, i+1).
    - Xóa những text ở frame i+1 nếu trùng với frame i (IoU >= iou_thresh và similarity >= text_sim_thresh).
    - Nếu frame i+1 rỗng => xóa hẳn frame i+1 khỏi video_results.
    - Tiếp tục so sánh với frame kế tiếp (logic while).
    """
    def local_iou(boxA, boxB):
        # box = [x, y, w, h]
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
# Hàm xuất SRT
###############################################################################

def generate_srt_from_results(
    video_results: List[Dict[str, Any]],
    min_gap_sec: float = 0.5,
    ext_time_sec: float = 1.0,
    output_srt: str = "output.srt"
):
    """
    Sinh file SRT đơn giản từ danh sách video_results.
    
    video_results: List[{"timestamp": float, "texts": [ { "text", "score", "box"}, ... ]}, ...]
    Gộp các dòng có timestamp sát nhau (trong vòng `min_gap_sec`) thành 1 block.
    Mỗi block hiển thị từ `start` đến `end + ext_time_sec`.
    """
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
            text = block["text"]

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
            f.write(f"{text}\n\n")

    logger.info(f"SRT saved to: {output_srt}")


def blur_and_reduce_contrast(image, kernel_size=15, sigmaX=0, contrast_alpha=0.3):
    """
    Làm mờ Gaussian và giảm độ tương phản của ảnh.

    Args:
        image (numpy.ndarray): Ảnh đầu vào (NumPy array).
        kernel_size (int, optional): Kích thước kernel cho Gaussian Blur. Mặc định là 31.
        sigmaX (int, optional): Độ lệch chuẩn cho Gaussian Blur. 0 để tự động tính. Mặc định là 0.
        contrast_alpha (float, optional): Hệ số alpha để giảm độ tương phản (0.0 - 1.0). Mặc định là 0.7.

    Returns:
        numpy.ndarray: Ảnh đã được xử lý.
    """

    # 1. Làm mờ Gaussian
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)

    # 2. Giảm độ tương phản
    processed_image = cv2.convertScaleAbs(blurred_image, alpha=contrast_alpha, beta=0)

    return processed_image

###############################################################################
# Logic chính: process_video
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
    debug_box_dir: Optional[str] = None
) -> List[Dict[str, Union[float, List[Dict[str, Union[str, float, List[float]]]]]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    frames_data = []  # (cropped_frame, offset_x, offset_y, original_frame)
    timestamps = []
    last_ocr_time = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if (current_sec - last_ocr_time) < sec_skip:
            continue
        last_ocr_time = current_sec
        original_frame = frame.copy()
        cropped, ox, oy = _crop_with_offset(frame, roi)
        frames_data.append((cropped, ox, oy, original_frame))
        timestamps.append(current_sec)
    cap.release()
    if not frames_data:
        logger.warning("No frames captured")
        return []
    # 1) BATCH DETECT
    all_images = [blur_and_reduce_contrast(x[0]) for x in frames_data]
    all_dt_boxes, _ = ocr_engine.infer_batch_image_det(all_images)
    # Lọc box theo kích thước của ảnh crop
    for i in range(len(all_dt_boxes)):
        all_dt_boxes[i] = filter_boxes_by_size(
            all_dt_boxes[i],
            frames_data[i][0].shape[:2],
            roi=roi,
            config=filter_config
        )
    # 2) Debug: Lưu ảnh detection và vẽ ROI lên ảnh gốc
    if debug_det_dir:
        os.makedirs(debug_det_dir, exist_ok=True)
        for i, dt_boxes_i in enumerate(all_dt_boxes):
            dbg_img = frames_data[i][3].copy()
            offset_x = frames_data[i][1]
            offset_y = frames_data[i][2]
            for box_ in dt_boxes_i:
                poly = np.array(box_, dtype=np.int32).copy()
                poly[:, 0] += offset_x
                poly[:, 1] += offset_y
                cv2.polylines(dbg_img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            if roi is not None:
                h_orig, w_orig = dbg_img.shape[:2]
                roi_x1 = int(roi[0] * w_orig)
                roi_y1 = int(roi[1] * h_orig)
                roi_x2 = int(roi[2] * w_orig)
                roi_y2 = int(roi[3] * h_orig)
                cv2.rectangle(dbg_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            out_path = os.path.join(debug_det_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(out_path, dbg_img)
    # 3) BATCH RECOG: Lấy crop từ ảnh gốc dựa vào box đã được điều chỉnh offset
    crops = []
    frame_idx_of_crop = []
    crop_counter = 0
    for i, dt_boxes_i in enumerate(all_dt_boxes):
        offset_x, offset_y = frames_data[i][1], frames_data[i][2]
        original_frame = frames_data[i][3]
        for box in dt_boxes_i:
            # Điều chỉnh tọa độ box về ảnh gốc
            adjusted_box = np.array(box, dtype=np.float32)
            adjusted_box[:, 0] += offset_x
            adjusted_box[:, 1] += offset_y
            if ocr_engine.det_box_type == 'quad':
                processed_crop = preprocess_crop(original_frame, adjusted_box)
            else:
                # Với minarea, bạn có thể dùng hàm get_minarea_rect_crop và sau đó xử lý nhẹ
                crop_img = get_minarea_rect_crop(original_frame, adjusted_box)
                processed_crop = preprocess_crop(original_frame, adjusted_box)
            if debug_box_dir:
                os.makedirs(debug_box_dir, exist_ok=True)
                debug_path = os.path.join(debug_box_dir, f"frame_{i:06d}_box_{crop_counter:03d}.jpg")
                cv2.imwrite(debug_path, processed_crop)
                crop_counter += 1
            crops.append(Image.fromarray(processed_crop))
            frame_idx_of_crop.append(i)
    rec_res_all, _ = ocr_engine.infer_batch_image_rec(crops)
    # 4) Gộp kết quả về từng frame
    final_boxes_per_frame = [[] for _ in range(len(frames_data))]
    final_texts_per_frame = [[] for _ in range(len(frames_data))]
    c_idx = 0
    for i, dt_boxes_i in enumerate(all_dt_boxes):
        for box_j in dt_boxes_i:
            text_j, score_j, _ = rec_res_all[c_idx]
            c_idx += 1
            if score_j >= ocr_engine.drop_score and text_j.strip():
                final_boxes_per_frame[i].append(box_j)
                final_texts_per_frame[i].append([text_j, score_j])
    # 5) Post-process: vanish_time, IoU, merge dòng,...
    video_results = []
    active_texts: List[Dict[str, Any]] = []
    for i in range(len(frames_data)):
        current_sec = timestamps[i]
        offset_x, offset_y = frames_data[i][1], frames_data[i][2]
        active_texts = [t for t in active_texts if (current_sec - t['last_seen']) < vanish_time]
        dt_boxes_i = final_boxes_per_frame[i]
        rec_res_i = final_texts_per_frame[i]
        if not dt_boxes_i:
            continue
        if do_merge:
            dt_boxes_i, rec_res_i = same_line_merge(dt_boxes_i, rec_res_i,
                                                    line_y_thresh, line_x_gap)
        frame_texts = []
        for box, (text, score) in zip(dt_boxes_i, rec_res_i):
            x, y, w, h = get_box_xywh(box)
            x += offset_x
            y += offset_y
            new_box_xywh = (x, y, w, h)
            matched_idx = -1
            best_sim = 0.0
            for idx_a, atext in enumerate(active_texts):
                iou_val = iou(atext['box'], new_box_xywh)
                sim_val = Levenshtein.ratio(
                    atext['text'].replace(" ", ""),
                    text.replace(" ", "")
                )
                if iou_val > iou_threshold or sim_val >= text_sim_threshold:
                    if sim_val > best_sim:
                        best_sim = sim_val
                        matched_idx = idx_a
            if matched_idx >= 0:
                active_texts[matched_idx]['last_seen'] = current_sec
                if (current_sec - active_texts[matched_idx]['last_output_time']) >= min_interval:
                    frame_texts.append({
                        "text": text,
                        "score": float(score),
                        "box": [float(x), float(y), float(w), float(h)]
                    })
                    active_texts[matched_idx]['last_output_time'] = current_sec
            else:
                frame_texts.append({
                    "text": text,
                    "score": float(score),
                    "box": [float(x), float(y), float(w), float(h)]
                })
                active_texts.append({
                    'text': text,
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
# main()
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Video OCR with batch logic + SRT.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video.')
    parser.add_argument('--cfg_det_path', type=str, default="configs/det/dbnet/repvit_db.yml")
    parser.add_argument('--cfg_rec_path', type=str, default="configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml")
    parser.add_argument('--drop_score', type=float, default=0.9)
    parser.add_argument('--det_batch_size', type=int, default=4)
    parser.add_argument('--rec_batch_size', type=int, default=16)
    parser.add_argument('--sec_skip', type=float, default=2.0, help='Skip frames by seconds.')
    parser.add_argument('--line_y_thresh', type=float, default=0.5)
    parser.add_argument('--line_x_gap', type=float, default=0.3)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    parser.add_argument('--vanish_time', type=float, default=2.0)
    parser.add_argument('--min_interval', type=float, default=5.0)
    parser.add_argument('--text_sim_threshold', type=float, default=0.8)
    parser.add_argument('--roi', type=str, default="0.1,0.7,0.9,0.95", help='ROI format: x1,y1,x2,y2 (theo % hoặc tỉ lệ, vd: 10,20,90,80 hoặc 0.1,0.2,0.9,0.8)')
    parser.add_argument('--output_json', type=str, default='output_results.json')
    parser.add_argument('--output_srt', type=str, default='output_results.srt', help='Path to save SRT.')
    parser.add_argument('--generate_srt', action='store_true', default=True, help='Whether to generate SRT from results.')
    parser.add_argument('--remove_dup_iou_thresh', type=float, default=0.5, help='IoU threshold for removing duplicates in consecutive frames.')
    parser.add_argument('--remove_dup_text_sim_thresh', type=float, default=0.9, help='Text similarity threshold for removing duplicates in consecutive frames.')
    parser.add_argument('--debug_det_dir', type=str, default="det_debug", help='Folder to save debug detect images. If None, skip saving.')
    parser.add_argument('--debug_box_dir', type=str, default="recog_debug", help='Folder to save debug box (crop) images. If None, skip saving.')
    # Thêm các tham số cho filter_config
    parser.add_argument('--min_w_ratio', type=float, default=0.02, help='Minimum width ratio of box to ROI.')
    parser.add_argument('--min_h_ratio', type=float, default=0.02, help='Minimum height ratio of box to ROI.')
    parser.add_argument('--max_w_ratio', type=float, default=0.9, help='Maximum width ratio of box to ROI.')
    parser.add_argument('--max_h_ratio', type=float, default=0.09, help='Maximum height ratio of box to ROI.')

    args = parser.parse_args()

    roi = parse_roi(args.roi)

    # Tạo filter_config từ các tham số command line
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
        filter_config=filter_config,  # Truyền filter_config
        debug_det_dir=args.debug_det_dir,
        debug_box_dir=args.debug_box_dir
    )

    # Remove duplicates ở các frame liên tiếp
    remove_subtitle_duplicates_consecutive_frames(
        video_results=results,
        iou_thresh=args.remove_dup_iou_thresh,
        text_sim_thresh=args.remove_dup_text_sim_thresh
    )

    time_process = time.time() - time_start
    print("time_process", time_process)

    # Lưu JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output_json}")

    # Xuất SRT nếu có
    if args.generate_srt:
        generate_srt_from_results(results, min_gap_sec=0.5, ext_time_sec=1.0, output_srt=args.output_srt)

if __name__ == '__main__':
    main()