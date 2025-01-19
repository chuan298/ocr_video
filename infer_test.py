#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script demonstrates an end-to-end OCR pipeline (detection + recognition)
with the following features:
 - Optional ROI cropping (region of interest).
 - Optional line-merging logic (merging bounding boxes of the same line).
 - Logic to skip frames based on elapsed time (seconds) in video processing.
 - Logic to avoid reading the same text repeatedly:
    * Maintains a list of active texts and checks bounding box IoU + text content.
    * Allows re-reading text after a minimum interval has passed.
    * Removes inactive texts after a vanish time.
 - NEW: Skip any empty (whitespace-only) text or text with a score below the threshold (drop_score).
"""

import os
import sys
import argparse
import time
import json
import copy
import cv2
import numpy as np
from PIL import Image

# Adjust module paths as needed for your environment.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.infer_rec import OpenRecognizer
from tools.infer_det import OpenDetector
from tools.engine import Config
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from tools.utils.logging import get_logger

logger = get_logger()

def parse_roi(roi_str):
    """
    Parse a ROI string in the format "x1,y1,x2,y2" into a list of four integers.

    Args:
        roi_str (str): The string specifying the ROI, e.g. "100,200,300,400".

    Returns:
        list[int] or None: [x1, y1, x2, y2] if parsing is successful;
                           None if parsing fails or if roi_str is empty/invalid.
    """
    if not roi_str:
        return None
    try:
        vals = [int(v.strip()) for v in roi_str.split(',')]
        if len(vals) != 4:
            return None
        return vals
    except:
        return None

def _crop_with_offset(frame, roi):
    """
    Crop an image/frame to the specified ROI and return offset coordinates.

    Args:
        frame (np.ndarray): The original image/frame in BGR format.
        roi (list[int]): A list [x1, y1, x2, y2] describing the ROI.

    Returns:
        tuple:
            - cropped_frame (np.ndarray): The cropped region of the image.
            - offset_x (int): The x offset (x1).
            - offset_y (int): The y offset (y1).

    Notes:
        If the ROI is partially or completely outside the image boundaries,
        it will be clamped to valid coordinates. If x2<=x1 or y2<=y1, 
        no valid region is extracted, so the original frame is returned
        and offsets set to 0.
    """
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]

    # Clamp ROI to valid image boundaries
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        # Invalid or zero-area ROI => return the original image
        return frame, 0, 0

    cropped = frame[y1:y2, x1:x2]
    return cropped, x1, y1

def get_box_xywh(quad):
    """
    Convert a quadrilateral box with 4 points (x, y) into an (x, y, w, h) rectangle.

    Args:
        quad (np.ndarray): Shape (4,2), each row is (x_i, y_i).

    Returns:
        tuple: (x_min, y_min, width, height) as floats.
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min, y_min, (x_max - x_min), (y_max - y_min))

def sorted_boxes(dt_boxes):
    """
    Sort detected boxes top-to-bottom, left-to-right based on their top-left point.

    Args:
        dt_boxes (list[np.ndarray]): Each item is a shape (4,2) array for one box.

    Returns:
        list[np.ndarray]: The sorted list of boxes.
    """
    num_boxes = len(dt_boxes)
    sorted_bx = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_bx)

    # Bubble-like pass to refine ordering for boxes on nearly the same row
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
               (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes

def iou(boxA, boxB):
    """
    Compute IoU (Intersection over Union) for two axis-aligned bounding boxes.

    Args:
        boxA (tuple): (xA, yA, wA, hA).
        boxB (tuple): (xB, yB, wB, hB).

    Returns:
        float: IoU value in [0,1].
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    xA2, yA2 = xA + wA, yA + hA
    xB2, yB2 = xB + wB, yB + hB

    inter_x1 = max(xA, xB)
    inter_y1 = max(yA, yB)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = wA * hA
    areaB = wB * hB
    denom = (areaA + areaB - inter_area)
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)

def same_line_merge(dt_boxes, rec_res,
                    line_y_thresh_ratio=0.5,
                    line_x_gap_ratio=0.3):
    """
    Merge bounding boxes that lie on the same line based on vertical and horizontal
    distance ratios.

    Args:
        dt_boxes (list[np.ndarray]): Detected boxes, each is shape (4,2).
        rec_res (list[list]): Recognition results in the form [[text, score], ...].
        line_y_thresh_ratio (float): Vertical threshold ratio to consider boxes on the same line.
        line_x_gap_ratio (float): Horizontal gap ratio to consider boxes contiguous on the same line.

    Returns:
        tuple: (merged_boxes, merged_texts) where
            merged_boxes is a list of np.ndarray(4,2),
            merged_texts is a list of [text_merged, score_merged].
    """
    if not dt_boxes:
        return dt_boxes, rec_res

    data = []
    for b, r in zip(dt_boxes, rec_res):
        data.append((b, r))
    # Sort by top-left corner (y, x)
    data = sorted(data, key=lambda x: (x[0][0][1], x[0][0][0]))

    used = [False] * len(data)
    merged = []

    for i in range(len(data)):
        if used[i]:
            continue

        box_i, rec_i = data[i]
        text_i, score_i = rec_i
        used[i] = True

        # Determine bounding for the current group
        all_x = [p[0] for p in box_i]
        all_y = [p[1] for p in box_i]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        group_text = text_i
        group_score = score_i

        box_i_w = max_x - min_x
        box_i_h = max_y - min_y

        # Attempt to merge subsequent boxes j
        for j in range(i + 1, len(data)):
            if used[j]:
                continue

            box_j, rec_j = data[j]
            text_j, score_j = rec_j

            xj = [p[0] for p in box_j]
            yj = [p[1] for p in box_j]
            min_xj, max_xj = min(xj), max(xj)
            min_yj, max_yj = min(yj), max(yj)
            w_j = max_xj - min_xj
            h_j = max_yj - min_yj

            # 1) Vertical check for same line
            avg_h = (box_i_h + h_j) / 2.0
            center_i_y = (min_y + max_y) / 2.0
            center_j_y = (min_yj + max_yj) / 2.0
            diff_y = abs(center_j_y - center_i_y)

            if diff_y <= line_y_thresh_ratio * avg_h:
                # 2) Horizontal check for adjacency
                avg_w = (box_i_w + w_j) / 2.0
                gap_x = min_xj - max_x  # distance between the groups horizontally
                if gap_x >= 0 and gap_x < line_x_gap_ratio * avg_w:
                    # Merge them
                    used[j] = True
                    # Concatenate text
                    group_text = group_text + " " + text_j
                    # Update score by taking average
                    group_score = (group_score + score_j) / 2.0
                    # Update bounding
                    min_x = min(min_x, min_xj)
                    max_x = max(max_x, max_xj)
                    min_y = min(min_y, min_yj)
                    max_y = max(max_y, max_yj)
                    box_i_w = max_x - min_x
                    box_i_h = max_y - min_y

        merged_box = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype=np.float32)
        merged.append((merged_box, (group_text.strip(), group_score)))

    merged_boxes = [m[0] for m in merged]
    merged_texts = [m[1] for m in merged]
    return merged_boxes, merged_texts

class OpenOCR(object):
    """
    A wrapper for OCR detection and recognition using 'OpenDetector' and 'OpenRecognizer'.

    Attributes:
        text_detector: The detection model.
        text_recognizer: The recognition model.
        det_box_type (str): Box type, e.g. 'quad' or 'poly'.
        drop_score (float): Score threshold for filtering recognition outputs.
    """
    def __init__(self, cfg_det_path, cfg_rec_path,
                 drop_score=0.5,
                 det_box_type='quad'):
        """
        Initialize the OCR system by loading detection and recognition configs.

        Args:
            cfg_det_path (str): Path to the detection config file (YAML).
            cfg_rec_path (str): Path to the recognition config file (YAML).
            drop_score (float): Score threshold to filter recognition results.
            det_box_type (str): Box type, 'quad' or 'poly'.
        """
        cfg_det = Config(cfg_det_path).cfg
        cfg_rec = Config(cfg_rec_path).cfg

        self.text_detector = OpenDetector(cfg_det)
        self.text_recognizer = OpenRecognizer(cfg_rec)
        self.det_box_type = det_box_type
        self.drop_score = drop_score

    def infer_single_image(self, img_numpy, rec_batch_num=6, crop_infer=False):
        """
        Perform text detection and recognition on a single image.

        Args:
            img_numpy (np.ndarray): The input image in BGR format.
            rec_batch_num (int): Batch size for recognition inference.
            crop_infer (bool): Whether to use a special 'crop_infer' approach
                               in the detection model (some models support it).

        Returns:
            tuple: (dt_boxes, rec_res, time_dict)
                - dt_boxes (list[np.ndarray]): The detected boxes.
                - rec_res (list[list]): The recognition results in [[text, score], ...].
                - time_dict (dict): Timing info, e.g. detection_time, recognition_time, etc.
        """
        if img_numpy is None:
            return None, None, None

        ori_img = img_numpy.copy()
        start = time.time()

        if crop_infer:
            det_result = self.text_detector.crop_infer(img_numpy=img_numpy)[0]
            dt_boxes = det_result['boxes']
        else:
            det_result = self.text_detector(img_numpy=img_numpy)[0]
            dt_boxes = det_result['boxes']

        det_time_cost = time.time() - start
        if dt_boxes is None or len(dt_boxes) == 0:
            return None, None, None

        dt_boxes = sorted_boxes(dt_boxes)

        # Crop sub-images for recognition
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = np.array(copy.deepcopy(dt_boxes[bno])).astype(np.float32)
            if self.det_box_type == 'quad':
                img_crop = get_rotate_crop_image(ori_img, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_img, tmp_box)
            img_crop_list.append(Image.fromarray(img_crop))

        start = time.time()
        rec_res = self.text_recognizer(img_numpy_list=img_crop_list, batch_num=rec_batch_num)
        rec_time_cost = time.time() - start

        # Filter results based on drop_score
        filter_boxes, filter_rec_res = [], []
        total_rec_time = 0.0
        for box, r in zip(dt_boxes, rec_res):
            text, score = r['text'], r['score']
            total_rec_time += r['elapse']
            # ADDED: skip empty text or text below threshold
            # i.e., text with length 0 after stripping or score < drop_score
            if score >= self.drop_score and len(text.strip()) > 0:
                filter_boxes.append(box)
                filter_rec_res.append([text, score])

        if len(filter_boxes) == 0:
            return None, None, None

        avg_rec_time = 0.0
        if len(dt_boxes) > 0:
            avg_rec_time = total_rec_time / len(dt_boxes)

        time_dict = {
            'time_cost': det_time_cost + rec_time_cost,
            'detection_time': det_time_cost,
            'recognition_time': rec_time_cost,
            'avg_rec_time_cost': avg_rec_time
        }
        return filter_boxes, filter_rec_res, time_dict

def process_image(ocr_engine,
                  image_path,
                  roi=None,
                  line_y_thresh=0.5,
                  line_x_gap=0.3,
                  do_merge=True):
    """
    Run the OCR pipeline on a single image file.

    Args:
        ocr_engine (OpenOCR): The OCR engine object with detection & recognition.
        image_path (str): Path to the input image file.
        roi (list[int] or None): ROI [x1,y1,x2,y2]. If provided, the image is cropped first.
        line_y_thresh (float): Vertical threshold ratio for line merging.
        line_x_gap (float): Horizontal gap ratio for line merging.
        do_merge (bool): Whether to apply same-line merging logic.

    Returns:
        list[dict]: A list of results, each item is a dict:
            {
              "text": str,
              "score": float,
              "box": [x, y, w, h]
            }
         in the coordinate space of the original image.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot read image: {image_path}")
        return []

    offset_x, offset_y = 0, 0
    if roi is not None:
        img, offset_x, offset_y = _crop_with_offset(img, roi)

    dt_boxes, rec_res, _ = ocr_engine.infer_single_image(img)
    if dt_boxes is None or rec_res is None:
        return []

    if do_merge:
        dt_boxes, rec_res = same_line_merge(dt_boxes, rec_res,
                                            line_y_thresh_ratio=line_y_thresh,
                                            line_x_gap_ratio=line_x_gap)

    results = []
    for box, (text, score) in zip(dt_boxes, rec_res):
        x, y, w, h = get_box_xywh(box)
        x += offset_x
        y += offset_y
        results.append({
            "text": text,
            "score": float(score),
            "box": [float(x), float(y), float(w), float(h)]
        })
    return results

def process_video(ocr_engine,
                  video_path,
                  roi=None,
                  line_y_thresh=0.5,
                  line_x_gap=0.3,
                  do_merge=True,
                  iou_threshold=0.5,
                  vanish_time=2.0,
                  min_interval=5.0,
                  sec_skip=1.0):
    """
    Run the OCR pipeline on a video, reading frames and skipping based on time intervals.

    Args:
        ocr_engine (OpenOCR): The OCR engine with detection & recognition.
        video_path (str): Path to the input video file.
        roi (list[int] or None): ROI [x1,y1,x2,y2]. Crop if not None.
        line_y_thresh (float): Vertical threshold ratio for line merging.
        line_x_gap (float): Horizontal gap ratio for line merging.
        do_merge (bool): Whether to apply same-line merging.
        iou_threshold (float): IoU threshold to match boxes across frames for repeated texts.
        vanish_time (float): Time in seconds after which inactive texts are removed from memory.
        min_interval (float): Minimum time in seconds to allow re-output of the same text.
        sec_skip (float): The minimum elapsed time in seconds between consecutive OCR attempts.

    Returns:
        list[dict]: A list where each element is:
            {
              "timestamp": float (seconds from video start),
              "texts": [
                  {
                     "text": str,
                     "score": float,
                     "box": [x, y, w, h]
                  },
                  ...
              ]
            }
        The 'box' is in the original video's coordinate space (considering ROI offset).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    video_results = []

    # List of active texts, each entry: 
    #  {
    #    'text': str,
    #    'box': (x, y, w, h),
    #    'first_seen': float,
    #    'last_seen': float,
    #    'last_output_time': float
    #  }
    active_texts = []

    last_ocr_time = 0.0  # keep track of the last second at which OCR was performed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_msec / 1000.0

        # Check if enough time has passed since last OCR attempt
        if (current_sec - last_ocr_time) < sec_skip:
            # Not enough elapsed time => skip OCR for this frame
            continue

        # Update the last OCR time for the next iteration
        last_ocr_time = current_sec

        # Crop ROI if needed
        offset_x, offset_y = 0, 0
        if roi is not None:
            frame_cropped, offset_x, offset_y = _crop_with_offset(frame, roi)
        else:
            frame_cropped = frame

        # OCR the cropped frame
        dt_boxes, rec_res, _ = ocr_engine.infer_single_image(frame_cropped)

        now_time = time.time()

        # Remove texts that have not been seen for more than vanish_time
        new_active_texts = []
        for t in active_texts:
            if (now_time - t['last_seen']) < vanish_time:
                new_active_texts.append(t)
        active_texts = new_active_texts

        # If no boxes found, move on
        if dt_boxes is None or rec_res is None or len(dt_boxes) == 0:
            continue

        # Merge lines if needed
        if do_merge:
            dt_boxes, rec_res = same_line_merge(
                dt_boxes, rec_res,
                line_y_thresh_ratio=line_y_thresh,
                line_x_gap_ratio=line_x_gap
            )

        frame_texts = []

        for box, (text, score) in zip(dt_boxes, rec_res):
            x, y, w, h = get_box_xywh(box)
            x += offset_x
            y += offset_y
            new_box_xywh = (x, y, w, h)

            # Check if the detected text matches an existing one in active_texts
            matched_idx = -1
            for i, atext in enumerate(active_texts):
                iou_val = iou(atext['box'], new_box_xywh)
                if iou_val > iou_threshold and atext['text'] == text:
                    matched_idx = i
                    break

            if matched_idx >= 0:
                # This text is already known
                active_texts[matched_idx]['last_seen'] = now_time

                # Decide if we want to "re-output" the text 
                if (now_time - active_texts[matched_idx]['last_output_time']) >= min_interval:
                    # We allow re-output after 'min_interval' 
                    frame_texts.append({
                        "text": text,
                        "score": float(score),
                        "box": [float(x), float(y), float(w), float(h)]
                    })
                    active_texts[matched_idx]['last_output_time'] = now_time
            else:
                # A new text
                frame_texts.append({
                    "text": text,
                    "score": float(score),
                    "box": [float(x), float(y), float(w), float(h)]
                })
                active_texts.append({
                    'text': text,
                    'box': new_box_xywh,
                    'first_seen': now_time,
                    'last_seen': now_time,
                    'last_output_time': now_time
                })

        # If we got new texts for this frame, append them to the results with a timestamp
        if len(frame_texts) > 0:
            video_results.append({
                "timestamp": current_sec,
                "texts": frame_texts
            })

    cap.release()
    return video_results

def main():
    """
    Main entry point. Parses command line arguments and runs either
    image or video OCR according to user inputs, then saves results to JSON.
    """
    parser = argparse.ArgumentParser(description='OpenOCR system skip by time (sec_skip)')
    parser.add_argument('--img_path', type=str, help='Path to an input image.')
    parser.add_argument('--video_path', type=str, help='Path to an input video.')
    parser.add_argument('--cfg_det_path', type=str, default="/var/account/ancv/OCR/OpenOCR/configs/det/dbnet/repvit_db.yml", help='Path to the detection config (YAML).')
    parser.add_argument('--cfg_rec_path', type=str, default="/var/account/ancv/OCR/OpenOCR/configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer.yml", help='Path to the recognition config (YAML).')
    parser.add_argument('--drop_score', type=float, default=0.9, help='Recognition score threshold.')
    parser.add_argument('--output_json', type=str, default='output_results.json', help='Path to output JSON file.')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI in "x1,y1,x2,y2" format. If not set, the full image/video is used.')

    # Merging ratio
    parser.add_argument('--line_y_thresh', type=float, default=0.5,
                        help='Vertical ratio threshold for line merging.')
    parser.add_argument('--line_x_gap', type=float, default=0.3,
                        help='Horizontal gap ratio threshold for line merging.')

    # Video-specific logic
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='IoU threshold for matching old vs new bounding boxes.')
    parser.add_argument('--vanish_time', type=float, default=2.0,
                        help='Time in seconds after which old, unseen texts are removed.')
    parser.add_argument('--min_interval', type=float, default=5.0,
                        help='Minimum time (seconds) for re-output of an old text.')
    parser.add_argument('--sec_skip', type=float, default=2.0,
                        help='Skip threshold in seconds for reading frames in video (time-based skipping).')

    args = parser.parse_args()

    # Ensure we have either an image or video path
    if (args.img_path is None) and (args.video_path is None):
        raise ValueError("You must provide --img_path or --video_path.")

    # Parse ROI
    roi = parse_roi(args.roi)

    # Initialize the OCR engine
    ocr_engine = OpenOCR(cfg_det_path=args.cfg_det_path,
                         cfg_rec_path=args.cfg_rec_path,
                         drop_score=args.drop_score)

    final_result = []
    if args.img_path:
        # Process a single image
        final_result = process_image(
            ocr_engine,
            image_path=args.img_path,
            roi=roi,
            line_y_thresh=args.line_y_thresh,
            line_x_gap=args.line_x_gap,
            do_merge=True
        )
    else:
        # Process a video
        final_result = process_video(
            ocr_engine,
            video_path=args.video_path,
            roi=roi,
            line_y_thresh=args.line_y_thresh,
            line_x_gap=args.line_x_gap,
            do_merge=True,
            iou_threshold=args.iou_thresh,
            vanish_time=args.vanish_time,
            min_interval=args.min_interval,
            sec_skip=args.sec_skip
        )

    # Save to JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Results have been saved to {args.output_json}.")

if __name__ == '__main__':
    main()
