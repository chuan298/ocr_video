# OpenOCR Video Pipeline with Memory Optimization

This project provides a high-performance OCR pipeline specifically optimized for video processing with intelligent memory management. It supports configurable processing strategies for videos of all lengths while preventing memory overflow.

## Key Features

### Performance Optimizations
* **Memory-Optimized Processing:** Automatically adjusts batch sizes and segment durations based on available memory
* **Dynamic Resource Management:** Monitors and controls RAM usage to prevent out-of-memory errors
* **Video Length Optimization:** Special handling for short, medium and long videos
* **Segment-Based Processing:** Divides long videos into manageable chunks to optimize memory usage

### OCR Capabilities
* **Region of Interest (ROI) Filtering:** Focus OCR on specific areas (e.g., subtitle regions)
* **Text Line Merging:** Intelligently combines related text boxes into coherent lines
* **Frame Skipping:** Process only key frames to improve performance
* **Subtitle Tracking:** Tracks text across frames for consistent subtitle extraction
* **Duplicate Suppression:** Prevents repetitive text while maintaining temporal accuracy

### Output Options
* **JSON Export:** Detailed OCR results with timestamps and positions
* **SRT Generation:** Creates subtitle files compatible with video players
* **Visualization:** Optional debug visualizations of detected text regions
* **Performance Metrics:** Memory usage graphs and timing statistics

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/openocr-video.git
cd openocr-video

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage with memory optimization
python main.py --video_path input.mp4 --generate_srt --max_memory_gb 4.0

# For subtitle extraction with region of interest (bottom of screen)
python main.py --video_path input.mp4 --roi "0.1,0.7,0.9,0.97" --max_memory_gb 4.0

# For high-resolution videos on limited-memory systems
python main.py --video_path 4k_video.mp4 --resize_factor 0.4 --max_memory_gb 2.0
```

## Memory Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_memory_gb` | Maximum memory usage allowed in GB | 4.0 |
| `--resize_factor` | Frame size reduction factor (0-1) | 0.5 |
| `--initial_batch_size` | Starting batch size for processing | 8 |
| `--segment_duration_sec` | Duration of each video segment in seconds | 60 |
| `--dynamic_sizing` | Enable automatic batch and segment size adjustment | True |

## Understanding `segment_duration_sec`

The `segment_duration_sec` parameter is crucial for memory management and processing efficiency:

### What It Does
This parameter controls how video frames are grouped for processing. The system:
1. Divides the video into time segments (e.g., 60-second chunks)
2. Opens the video file and processes one segment
3. Closes the video file and releases all memory
4. Repeats for the next segment

### Impact on Memory Usage
- **Smaller Values (10-30 seconds)**:
  - Lower peak memory consumption
  - More frequent memory cleanup
  - Better for memory-constrained systems
  - Recommended for 4K videos or when processing with limited RAM

- **Larger Values (120-300 seconds)**:
  - Higher peak memory usage
  - Fewer memory cleanup operations
  - Better for systems with abundant RAM (16GB+)
  - May cause out-of-memory errors on long, high-resolution videos

### Impact on Processing Speed
- **Smaller Values (10-30 seconds)**:
  - More overhead from frequent video file open/close operations
  - Slightly slower overall processing due to segment transition overhead
  - Better for real-time monitoring of progress
  
- **Larger Values (120-300 seconds)**:
  - Less file operation overhead
  - Typically faster overall processing when memory is sufficient
  - May become slower if memory limits are reached and system starts swapping

### Recommended Settings by Video Length

| Video Length | Recommended Setting | Impact on Speed |
|--------------|---------------------|-----------------|
| < 3 minutes  | Set to video length | 2-5x faster     |
| 3-10 minutes | 1/3 of video length | 1.5-3x faster   |
| > 10 minutes | 60-120 seconds      | Minimal impact  |

**Note:** With `dynamic_sizing=True` (default), the system will automatically optimize this parameter based on video length.

## Optimizing for Video Length

The system automatically adjusts its processing strategy based on video duration:

### Short Video Optimization (< 3 minutes)
- **Single Segment Processing**: Processes the entire video in one pass
- **Larger Batch Sizes**: Uses up to 2x larger batch sizes for faster processing
- **Reduced Overhead**: Eliminates file opening/closing operations
- **Maximum Memory Utilization**: Uses more available memory to speed up processing

### Medium Video Optimization (3-30 minutes)
- **Reduced Segmentation**: Uses fewer segments to minimize overhead
- **Balanced Memory Usage**: Adjusts segment size proportionally to video length
- **Adaptive Batch Sizing**: Maintains larger batch sizes when memory permits

### Long Video Optimization (> 30 minutes)
- **Full Memory Management**: Applies complete memory optimization strategy
- **Controlled Segmentation**: Keeps segment size between 30-300 seconds based on available memory
- **Conservative Batch Sizing**: Prioritizes stable processing over speed

### Memory vs Speed Trade-off:
- With `dynamic_sizing=True`, the system automatically optimizes for your video length
- For short videos on high-memory systems, set `max_memory_gb` higher (e.g., 8.0-16.0) to maximize speed
- For short videos on limited memory, keep `resize_factor` low (e.g., 0.3-0.4) to allow larger batch sizes

## OCR Processing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--sec_skip` | Process frames every N seconds | 0.5 |
| `--roi` | Region of interest (format: "x1,y1,x2,y2") | "0.1,0.6,0.9,0.97" |
| `--drop_score` | Minimum confidence score | 0.9 |
| `--line_y_thresh` | Vertical threshold for line merging | 0.5 |
| `--line_x_gap` | Horizontal gap threshold for line merging | 0.3 |
| `--iou_thresh` | IoU threshold for duplicate detection | 0.5 |
| `--min_interval` | Minimum time between duplicate text | 5.0 |
| `--text_sim_threshold` | Text similarity threshold | 0.8 |

## Output Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output_json` | Path to save JSON results | "output_results.json" |
| `--output_srt` | Path to save SRT subtitle file | "output_results.srt" |
| `--generate_srt` | Generate SRT file | True |
| `--debug_det_dir` | Directory to save detection visualizations | None |

## Advanced Usage Examples

### Processing a Long HD Movie

```bash
python main.py --video_path movie.mp4 --max_memory_gb 6.0 --segment_duration_sec 120 \
  --roi "0.1,0.75,0.9,0.95" --sec_skip 1.0 --resize_factor 0.5
```

### Processing a 4K Video on Limited Memory

```bash
python main.py --video_path 4k_concert.mp4 --max_memory_gb 2.0 --segment_duration_sec 30 \
  --resize_factor 0.3 --initial_batch_size 4
```

### Maximum Speed for Short Videos

```bash
python main.py --video_path short_clip.mp4 --max_memory_gb 16.0 \
  --dynamic_sizing --initial_batch_size 16
```

### Extracting Subtitles from an Anime

```bash
python main.py --video_path anime_episode.mp4 --roi "0.1,0.8,0.9,0.95" \
  --text_sim_threshold 0.95 --min_interval 3.0
```

## Performance Logging

The system automatically logs performance metrics to the `logs/` directory:
- Memory usage graphs
- Processing speed statistics
- Segment and batch timing details

## JSON Output Format

```json
[
  {
    "start_time": 10.5,
    "end_time": 13.2,
    "timestamp": 10.5,
    "texts": [
      {
        "text": "Example subtitle text",
        "score": 0.97,
        "box": [250, 400, 800, 50]
      }
    ]
  }
]
```

## Understanding Memory Optimization

The system uses a multi-level memory optimization approach:

1. **Frame-Level**: Resizes frames to reduce memory footprint
2. **Batch-Level**: Dynamically adjusts batch sizes based on available memory
3. **Segment-Level**: Processes the video in time-based chunks (segment_duration_sec)
4. **Video-Length Aware**: Uses different strategies for short, medium, and long videos
5. **Monitoring**: Continuously tracks memory usage and adjusts accordingly

## Troubleshooting

### Out of Memory Errors
- Decrease `--max_memory_gb` to set a lower limit
- Reduce `--resize_factor` to work with smaller frames
- Decrease `--segment_duration_sec` to process smaller video chunks
- Lower `--initial_batch_size` to reduce peak memory usage

### Slow Processing
- Increase `--sec_skip` to process fewer frames
- Set a specific ROI to process smaller image regions
- Increase `--max_memory_gb` if you have available RAM
- Decrease `--segment_duration_sec` for short videos to avoid unnecessary segmentation
- Increase `--initial_batch_size` for faster batch processing when memory allows

### Poor Text Recognition
- Decrease `--resize_factor` to maintain higher image quality
- Adjust `--line_y_thresh` and `--line_x_gap` for better line merging
- Increase `--drop_score` threshold for higher confidence results
- Reduce `--text_sim_threshold` to capture more variations in text

## License

[MIT License](LICENSE)