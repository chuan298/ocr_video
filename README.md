# OpenOCR Multiprocessing Video Pipeline

This project provides an end-to-end OCR pipeline specifically for video input. In this updated version, the video processing part leverages multiprocessing to read video segments concurrently—dramatically speeding up video input and preprocessing. The pipeline supports features such as ROI cropping, line merging, time-based frame skipping, and duplicate text suppression, and can optionally generate SRT subtitle files from the OCR results.

## Key Features

* **Multiprocessing Video Reading:**  
  The video is split into segments and read concurrently by multiple processes. This minimizes I/O bottlenecks and speeds up frame extraction, especially for high-resolution or long-duration videos.

* **Region of Interest (ROI) Cropping:**  
  Focus the OCR on a specific area by providing an ROI (`--roi "x1,y1,x2,y2"`) in either percentage or absolute values. Only the defined portion of the video is processed.

* **Line Merging:**  
  Bounding boxes that likely belong to the same line of text are merged based on vertical and horizontal thresholds (`--line_y_thresh` and `--line_x_gap`), resulting in cleaner OCR results.

* **Time-Based Frame Skipping:**  
  Only frames that are sufficiently apart in time (as specified by `--sec_skip`) are processed. This avoids redundant OCR on nearly identical consecutive frames.

* **Duplicate Text Suppression:**  
  The pipeline tracks recognized text across frames and suppresses duplicate outputs using Intersection over Union (IoU, `--iou_thresh`) and text similarity thresholds (`--text_sim_threshold`). Parameters like `--vanish_time` and `--min_interval` control how long a text remains in memory and when it can be output again.

* **Configurable Batch Sizes:**  
  Batch sizes for detection (`--det_batch_size`) and recognition (`--rec_batch_size`) are configurable to optimize performance.

* **SRT Generation:**  
  Optionally, a SubRip subtitle (SRT) file is generated from the OCR results for video content.

## File Overview

* **`main.py` :**  
  This is the main script that:
  - Implements the OCR pipeline for video input.
  - Uses multiprocessing to split and read video segments concurrently.
  - Contains the `OpenOCR` class that wraps the detection (`OpenDetector`) and recognition (`OpenRecognizer`) models.
  - Implements functions for ROI cropping, line merging (`same_line_merge`), and duplicate text suppression.
  - Generates an output JSON and (optionally) an SRT file from the OCR results.

* **`tools/` Directory:**  
  Contains auxiliary modules for detection, recognition, configuration, and logging. Adapt these modules to match your specific OCR models if needed.

## Installation and Requirements

```bash
pip install -r requirements.txt
```

## Usage

Since this pipeline is dedicated to video processing, you must provide the video path (using the `--video_path` argument). Here is an example:

Simple using default config:

```bash
python3 main.py --video_path input.mp4 --generate_srt
```

Using custom config:

```bash
python3 main.py --video_path input.mp4 --output_json results.json --sec_skip 2.0 --iou_thresh 0.5 --vanish_time 2.0 --min_interval 5.0 --num_processes 8 --roi "0.1,0.7,0.9,0.97" --det_batch_size 4 --rec_batch_size 16 --drop_score 0.9 --line_y_thresh 0.5 --line_x_gap 0.3
```

### Command-Line Arguments Overview

* **`--video_path`**:  
  Path to the input video file. (Required)

* **`--cfg_det_path`** and **`--cfg_rec_path`**:  
  Paths to the detection and recognition configuration files, respectively.

* **`--drop_score`**:  
  Recognition score threshold (default: 0.9).

* **`--det_batch_size`** and **`--rec_batch_size`**:  
  Batch sizes for detection and recognition (default: 4 and 16, respectively).

* **`--sec_skip`**:  
  Skip frames by seconds. OCR is only performed if the elapsed time since the last processed frame is at least this value (default: 2.0).

* **`--line_y_thresh`** and **`--line_x_gap`**:  
  Threshold parameters for merging bounding boxes into lines (defaults: 0.5 and 0.3).

* **`--iou_thresh`**:  
  IoU threshold for duplicate text suppression across frames (default: 0.5).

* **`--vanish_time`** and **`--min_interval`**:  
  Time parameters to manage duplicate text outputs (default: 2.0 and 5.0).

* **`--text_sim_threshold`**:  
  Text similarity threshold for duplicate suppression (default: 0.8).

* **`--roi`**:  
  Region of Interest for cropping (default: "0.1,0.7,0.9,0.97"). Coordinates may be percentages or absolute values.

* **`--output_json`**:  
  Path to the output JSON file (default: `output_results.json`).

* **`--output_srt`**:  
  Path to the output SRT file (default: `output_results.srt`).

* **`--generate_srt`**:  
  Flag to indicate if an SRT file should be generated (default: True).

* **`--remove_dup_iou_thresh`** and **`--remove_dup_text_sim_thresh`**:  
  Additional thresholds for duplicate text suppression.

* **`--debug_det_dir`** and **`--debug_box_dir`**:  
  Directories to save debug images for detection and recognition.

* **`--min_w_ratio`, `--min_h_ratio`, `--max_w_ratio`, `--max_h_ratio`**:  
  Parameters to filter out bounding boxes by size relative to the ROI.

* **`--num_processes`**:  
  Number of processes to use for reading the video concurrently (default: 8).

## Output Format

### Video Output

The output JSON for video processing is a list where each element represents a frame (or timestamp) where OCR was performed:

```json
[
  {
    "timestamp": 1.5,
    "texts": [
      {
        "text": "Some Text",
        "score": 0.9,
        "box": [50.0, 50.0, 100.0, 20.0]
      }
    ]
  },
  {
    "timestamp": 3.2,
    "texts": [
      {
        "text": "Other Text",
        "score": 0.85,
        "box": [60.0, 70.0, 120.0, 25.0]
      }
    ]
  }
]
```

* **`timestamp`**:  
  The time (in seconds) where OCR was performed.
  
* **`texts`**:  
  A list of OCR results for that frame, each containing:
  - **`text`**: Recognized text.
  - **`score`**: Recognition confidence.
  - **`box`**: Bounding box coordinates in the original video (format: `[x, y, w, h]`).

## Customization

* **Detection and Recognition Models:**  
  Replace the placeholder `OpenDetector` and `OpenRecognizer` classes with your actual models. Adapt the files in the `tools/` directory as necessary.

* **Configuration:**  
  Modify the YAML configuration files (`configs/det/...`, `configs/rec/...`) to adjust model parameters.

* **Line Merging:**  
  Adjust the `--line_y_thresh` and `--line_x_gap` parameters to control how bounding boxes are merged into lines.

* **Video Processing Settings:**  
  Tune `--sec_skip`, `--iou_thresh`, `--vanish_time`, `--min_interval`, and `--num_processes` to optimize performance on your hardware.

* **Batch Sizes:**  
  Experiment with `--det_batch_size` and `--rec_batch_size` to balance speed and accuracy.