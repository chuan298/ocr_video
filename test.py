# import cv2
# import os

# def extract_frames(video_path, output_folder="video_images", skip_frames=1):
#     """
#     Trích xuất frame từ video và lưu vào thư mục.

#     Args:
#         video_path (str): Đường dẫn đến video.
#         output_folder (str): Thư mục lưu ảnh.
#         skip_frames (int): Số frame cần bỏ qua trước khi lưu một frame.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     saved_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # Thoát vòng lặp nếu hết video

#         # Chỉ lưu frame nếu frame_count chia hết cho (skip_frames + 1)
#         if frame_count % (skip_frames + 1) == 0:
#             frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()
#     print(f"Đã trích xuất {saved_count} frames vào thư mục '{output_folder}'.")

# # Gọi hàm với tham số skip_frames
# video_path = r"C:\Users\chuva\Videos\2025-03-02 11-33-43.mp4"  # Thay bằng đường dẫn video của bạn
# extract_frames(video_path, skip_frames=3)



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
            print(len(image_paths))
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
    
    # Initialize OCR engine
    logger.info("Initializing OCR engine...")
    ocr_engine = OpenOCR(
        cfg_det_path=args.cfg_det_path,
        cfg_rec_path=args.cfg_rec_path,
        drop_score=args.drop_score,
        det_batch_size=args.det_batch_size,
        rec_batch_size=args.rec_batch_size
    )
    from PIL import Image
    import pandas as pd
    images = [Image.open(path) for path in image_paths]

    rec_res_full, total_rec_time = ocr_engine.infer_batch_image_rec(images)

    print(rec_res_full)
    print(total_rec_time)
    # Lưu kết quả OCR thành file CSV với các trường: image_file_name, ocr, score
    results = []
    for img_path, res in zip(image_paths, rec_res_full):
        results.append({
            "image_file_name": os.path.basename(img_path),
            "ocr": res[0],
            "score": res[1]
        })
    
    output_csv = args.output_json.replace('.json', '.csv') if args.output_json.endswith('.json') else args.output_json
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Results saved to {output_csv}")

main()