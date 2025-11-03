# TreeInventorization

A comprehensive street-level tree detection and inventory system that processes Google Street View panoramas to identify, segment, and geolocate trees in urban environments.

## Overview

This project provides an end-to-end pipeline for automated tree detection from street-level imagery. It combines computer vision models for tree segmentation, depth estimation, and quality assessment to create accurate tree inventories with precise geolocation data.

## Features

- **Tree Detection**: YOLO-based tree segmentation from street view panoramas
- **Depth Estimation**: DepthAnything V2 model for accurate depth mapping
- **Quality Assessment**: Mask quality verification to filter false positives
- **Geolocation**: Precise tree positioning using depth maps and camera parameters
- **Batch Processing**: Asynchronous processing of multiple panoramas
- **Evaluation Tools**: Comprehensive evaluation against ground truth data
- **MRF Triangulation**: Physics-based triangulation for high-precision tree positioning

## Project Structure

```
TreeInventorization/
├── main.py                # Main pipeline entry point  
├── config.py              # Configuration management
├── cli.py                 # Command-line interface
├── src/                   # Source code
│   ├── inference/         # Model inference modules
│   │   ├── depth.py       # Depth estimation
│   │   ├── mask.py        # Mask quality verification
│   │   └── segment.py     # Tree segmentation
│   ├── pipeline/          # Main processing pipeline
│   │   └── pano_async.py  # Asynchronous panorama processing
│   ├── utils/             # Utility functions
│   │   ├── depth_calibration.py
│   │   ├── geodesic.py
│   │   ├── masks.py
│   │   ├── transformation.py
│   │   └── unwrap.py
│   └── notebooks/         # Testing notebooks (not used in pipeline)
├── models/                # Model weights (gitignored - must be downloaded)
│   ├── TreeModelV3/       # Trunk segmentation model
│   ├── TreeModel/         # Tree segmentation model
│   ├── DepthAnything/     # Depth estimation model
│   ├── CalibrateDepth/    # Depth calibration model
│   └── MaskQuality/       # Mask quality assessment model
├── data/                  # Data storage
│   ├── full/              # Full panorama images
│   ├── views/             # Perspective views
│   ├── depth_maps/        # Generated depth maps
│   ├── masks/             # Generated masks
│   └── logs/              # Processing logs
├── streetviews/           # Panorama metadata
│   └── *.csv              # Panorama ID lists
├── outputs/               # Generated outputs
│   └── *.csv              # Tree detection results
├── eval/                  # Evaluation tools
│   └── eval.py            # Model evaluation script
├── annotations/           # Training annotations (ignored)
└── old/                   # Legacy code (ignored)
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utk984/TreeInventorization
   cd TreeInventorization
   ```

2. **Download model weights:**
   The `models/` directory is gitignored and must be populated with pre-trained model weights:
   - TreeModelV3 weights
   - DepthAnything V2 checkpoints
   - CalibrateDepth model weights
   - MaskQuality model weights

   Place the model files in their respective directories as specified in `config.py`.

## Usage

### Quick Start

Run the main tree detection pipeline:

```bash
# Run the tree detection pipeline
python3 main.py
```

### Detailed Usage

#### Configuration

All pipeline settings are configured through `config.py`. Key settings include:

- **Model paths**: Locations of pre-trained model weights  
- **Data directories**: Input/output folder paths
- **Processing parameters**: Image dimensions, batch size, FOV
- **Device settings**: CUDA/CPU selection
- **Triangulation settings**: MRF triangulation parameters
- **Logging**: Log file configuration

#### Evaluation

Evaluate model predictions against ground truth:

```bash
python3 eval/eval.py path/to/predictions.csv
```


## Data Format

### Input CSV Format
Panorama ID CSV should contain:
- `pano_id`: Google Street View panorama identifier
- `lat`, `lng`: Panorama coordinates

### Output CSV Format
Generated tree data CSV contains:
- `pano_id`: Source panorama identifier
- `tree_lat`, `tree_lng`: Tree coordinates
- `conf`: Detection confidence score
- `image_path`: Path to source image
- `stview_lat`, `stview_lng`: Street view coordinates
- Additional metadata fields

### Ground Truth Format
Ground truth CSV should contain:
- `tree_lat`, `tree_lng`: Ground truth tree coordinates
- `pano_id`: Associated panorama identifier

## Pipeline Workflow

1. **Panorama Fetching**: Download street view panoramas and depth maps
2. **Perspective Generation**: Create multiple perspective views from panoramas
3. **Tree Detection**: Apply YOLO model for tree segmentation
4. **Depth Estimation**: Generate depth maps using DepthAnything V2
5. **Quality Assessment**: Filter detections using mask quality model
6. **Geolocation**: Calculate precise tree coordinates using depth and camera parameters
7. **MRF Triangulation**: Advanced post-processing for high-precision tree positioning
8. **Duplicate Removal**: Remove duplicate trees using physics-based triangulation

## Models

- **TreeModelV3**: YOLO-based trunk segmentation model
- **TreeModel**: YOLO-based tree segmentation model
- **DepthAnything V2**: Vision Transformer for depth estimation
- **CalibrateDepth**: Random Forest model for depth calibration
- **MaskQuality**: Quality assessment for segmentation masks

## Evaluation Metrics

The evaluation system provides:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Distance thresholds**: Configurable matching distances (3m, 5m)
- **Duplicate removal**: Configurable duplicate detection (2m, 5m)

## TODO

Future improvements and planned features:

- [x] **Flag-based saving for depth maps and segmentation masks** - Add config flags to optionally save intermediate processing results (depth maps, segmentation masks) for debugging and analysis purposes.

- [x] **Interactive visualization system** - Real-time interactive maps with tree and street view markers, connection lines, and image popups.

- [x] **Unified command-line interface** - Single script for pipeline, evaluation, and visualization operations.

- [ ] **Multiview triangulation** - Implement triangulation algorithms to improve tree localization accuracy by combining detections from multiple panorama viewpoints.

- [ ] **Improve tree detection model** - Enhance the current tree segmentation model with better training data, architecture improvements, or ensemble methods for higher accuracy.

- [ ] **Redo depth anything + regression pipeline** - Refactor the depth estimation pipeline to improve the integration between DepthAnything model and the regression-based depth calibration system.

## Acknowledgment

Developers: [Utkarsh Agarwal](https://github.com/Utk984), [Malhar Bhise](https://github.com/coolperson111)

This project was undertaken in collaboration with the [Geospatial Computer Vision Group](https://anupamsobti.github.io/geospatial-computer-vision/) led by [Dr. Anupam Sobti](https://anupamsobti.github.io/). We are grateful for the support and guidance provided throughout the development of this project.