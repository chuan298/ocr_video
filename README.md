
This project provides an example pipeline for Optical Character Recognition (OCR) on both images and video. It uses a custom detection model (`OpenDetector`) and recognition model (`OpenRecognizer`) from a hypothetical "OpenOCR" toolkit. 

## Features

1. **ROI Cropping**  
   - You can specify a rectangular region (`ROI`) in `x1,y1,x2,y2` format, so the OCR process will only run on that cropped area.  
   - The bounding boxes returned in the final JSON are offset back to the original coordinate system.

2. **Line Merging**  
   - If a single line of text is incorrectly split into multiple bounding boxes, the script can merge adjacent bounding boxes on the same line, based on user-specified vertical and horizontal ratios.

3. **Time-Based Frame Skipping**  
   - For video, instead of processing every frame or skipping by a certain number of frames, this system can skip frames based on **elapsed time (in seconds)**. For example, `--sec_skip 1.0` means that OCR is only run when at least 1 second has elapsed from the previous OCR run, regardless of frame rate.

4. **Avoid Re-Reading the Same Text**  
   - Maintains an internal list of `active_texts` to avoid repeatedly outputting the same text over consecutive frames. 
   - Uses bounding-box IoU (`--iou_thresh`) plus matching text to decide if a new detection is actually the same old text. 
   - Text that is not detected for `--vanish_time` seconds is removed from memory. 
   - Text can be re-output after a minimum interval (`--min_interval`).

## File Overview

- **main script**: The single Python file (e.g. `main.py`) containing:
  - **`OpenOCR` class** that wraps `OpenDetector` and `OpenRecognizer`.
  - Functions for **processing images** (`process_image`) and **processing video** (`process_video`).
  - **`same_line_merge`** to merge bounding boxes on the same line.
  - A **`main`** entry point that parses command-line arguments and saves output to JSON.

- **`tools/` directory** (not fully shown): Contains utility files for your detection and recognition logic.

## Installation and Requirements

- Python 3.7+ recommended.
- Dependencies (typical):
  - `numpy`
  - `opencv-python`
  - `Pillow`
  - `pyyaml` (if your config files are YAML)
  - Additional libraries for your detection/recognition models.

Install them with:
```bash
pip install -r requirements.txt
```
*(Create or update `requirements.txt` as needed.)*

## Usage

### 1. Process an Image

```bash
python infer_test.py \
  --img_path /path/to/image.jpg \
  --cfg_det_path /path/to/det_config.yml \
  --cfg_rec_path /path/to/rec_config.yml \
  --output_json /path/to/output.json \
  --roi "100,200,600,400" \
  --line_y_thresh 0.5 \
  --line_x_gap 0.3
```

- **`--img_path`**: Path to a single image.
- **`--roi "x1,y1,x2,y2"`**: Optional rectangular region for cropping the image. 
- **`--line_y_thresh`, `--line_x_gap`**: Parameters for line merging ratios.
- **`--output_json`**: Where the results are stored.

### 2. Process a Video

```bash
python infer_test.py \
  --video_path /path/to/video.mp4 \
  --cfg_det_path /path/to/det_config.yml \
  --cfg_rec_path /path/to/rec_config.yml \
  --output_json /path/to/output.json \
  --line_y_thresh 0.5 \
  --line_x_gap 0.3 \
  --iou_thresh 0.5 \
  --vanish_time 2.0 \
  --min_interval 5.0 \
  --sec_skip 1.0
```

- **`--video_path`**: Path to the input video.
- **`--sec_skip`**: Skip frames until at least X seconds have passed since the last OCR (time-based skipping).
- **`--iou_thresh`**: Intersection-over-Union threshold to decide whether a newly detected bounding box is the same as an old one.
- **`--vanish_time`**: How many seconds a text can go undetected before being removed from memory.
- **`--min_interval`**: Time in seconds to allow re-output of the same text.

## Output Format

- For images, the JSON output is a **list** of dictionary objects:
  ```json
  [
    {
      "text": "example",
      "score": 0.98,
      "box": [120.0, 180.0, 200.0, 40.0]
    },
    ...
  ]
  ```

- For videos, the JSON output is a **list** where each element corresponds to a timestamp:
  ```json
  [
    {
      "timestamp": 0.0,
      "texts": [
        {
          "text": "example",
          "score": 0.98,
          "box": [120.0, 180.0, 200.0, 40.0]
        },
        ...
      ]
    },
    {
      "timestamp": 1.03,
      "texts": [...]
    },
    ...
  ]
  ```

## Customization

- **Line merging**: Adjust `--line_y_thresh` (vertical tolerance) and `--line_x_gap` (horizontal gap) to get the desired merging behavior.
- **Time-based skipping**: Tweak `--sec_skip` to control how frequently OCR is performed in the video.
- **Text re-reading**: Modify `--min_interval` and the logic that handles `last_output_time` if you prefer to never re-output the same text or to re-output it more frequently.
