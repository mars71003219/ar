# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MMPose is an open-source toolbox for pose estimation based on PyTorch, part of the OpenMMLab project. It supports diverse pose analysis tasks including 2D/3D human pose estimation, hand/face keypoint detection, whole-body pose estimation, and animal pose estimation.

## Development Commands

### Installation & Setup
```bash
# Install from source (recommended for development)
pip install -e .

# Install with all dependencies
pip install -e .[all]

# Install for testing only
pip install -e .[tests]
```

### Testing
```bash
# Run all tests with pytest
pytest

# Run tests for a specific module
pytest tests/test_models/

# Run tests with coverage
pytest --cov=mmpose

# Run with xdoctest (as configured in pytest.ini)
pytest --xdoctest --xdoctest-style=auto
```

### Code Quality & Linting
```bash
# Format code with yapf
yapf -i **/*.py

# Check code style with flake8
flake8 mmpose tests

# Sort imports with isort
isort mmpose tests --check-diff

# Check test coverage
coverage run -m pytest
coverage report
```

### Model Training & Evaluation
```bash
# Training
python tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py

# Distributed training
bash tools/dist_train.sh configs/path/to/config.py 8

# Testing/Evaluation
python tools/test.py configs/path/to/config.py checkpoints/model.pth

# Distributed testing
bash tools/dist_test.sh configs/path/to/config.py checkpoints/model.pth 8
```

### Demo & Inference
```bash
# Image demo with pose detection
python demo/image_demo.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py \
    checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input tests/data/coco/000000000785.jpg \
    --output-root vis_results

# Video demo
python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py \
    checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input demo/demo.mp4 \
    --output-root vis_results
```

## Architecture Overview

### Core Components

1. **mmpose/models/**: Model implementations
   - `backbones/`: Feature extractors (ResNet, HRNet, Swin, etc.)
   - `heads/`: Task-specific prediction heads (heatmap, coordinate classification, regression)
   - `pose_estimators/`: Top-down, bottom-up, and pose lifting models
   - `necks/`: Feature processing modules (FPN, PAFPn)
   - `losses/`: Various loss functions for pose estimation

2. **mmpose/datasets/**: Dataset handling
   - `datasets/`: Dataset classes for different pose estimation tasks
   - `transforms/`: Data augmentation and preprocessing pipelines
   - Supports COCO, MPII, Human3.6M, and many specialized datasets

3. **mmpose/codecs/**: Label encoding/decoding
   - Handles conversion between annotations and model targets
   - Supports heatmaps, coordinate classification, regression labels

4. **mmpose/evaluation/**: Evaluation metrics
   - COCO-style AP metrics, PCK, AUC, EPE for different tasks
   - Custom metrics for specialized datasets

### Key Architectural Patterns

- **Registry System**: Uses OpenMMLab's registry pattern for component registration
- **Config System**: YAML/Python configs define model architecture and training settings
- **Hook System**: Training hooks for logging, checkpointing, and custom callbacks
- **Data Pipeline**: Flexible transform pipelines for different input modalities

### Model Categories

1. **Top-down**: Detect persons first, then estimate pose (RTMPose, HRNet)
2. **Bottom-up**: Detect keypoints first, then group into persons (Associative Embedding)
3. **One-stage**: End-to-end detection and pose estimation (RTMO, YOLOX-Pose)
4. **3D Pose**: Lifting 2D poses to 3D or direct 3D estimation

### Recent Key Models

- **RTMPose**: Real-time pose estimation with SimCC (coordinate classification)
- **RTMO**: Real-time multi-person pose estimation (one-stage)
- **RTMW**: Whole-body pose estimation (133 keypoints)

## Configuration System

Configs are located in `configs/` and organized by task:
- `body_2d_keypoint/`: Human pose estimation
- `wholebody_2d_keypoint/`: Whole-body pose (body + face + hands)
- `hand_2d_keypoint/`: Hand keypoint detection
- `face_2d_keypoint/`: Facial landmark detection
- `body_3d_keypoint/`: 3D human pose estimation

## Dataset Structure

Place datasets in `data/` directory following MMPose format. Use tools in `tools/dataset_converters/` to convert from original formats to COCO-style annotations.

## Important Files

- `mmpose/registry.py`: Central registry for all components
- `mmpose/structures/pose_data_sample.py`: Core data structure
- `configs/_base_/default_runtime.py`: Default training configuration
- `tools/train.py` & `tools/test.py`: Main training/testing scripts

## Development Tips

- When adding new models, register them in the appropriate registry
- Use existing transform pipelines when possible
- Follow OpenMMLab coding standards and patterns
- Always test with both single-GPU and multi-GPU setups
- Check that configs work with both training and testing modes