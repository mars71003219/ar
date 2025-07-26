# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a machine learning research repository focused on computer vision tasks, specifically combining **MMPose** (pose estimation) and **MMAction2** (action recognition) frameworks from OpenMMLab. The repository contains:

1. **MMPose** - OpenMMLab pose estimation toolbox supporting 2D/3D human pose estimation, hand/face keypoint detection, and whole-body pose estimation
2. **MMAction2** - OpenMMLab video understanding toolbox for action recognition, detection, and localization
3. **RTMO GCN Test** - A custom violence detection application combining RTMO pose detection with GCN-based action classification

## Development Commands

### Environment Setup
```bash
# Install MMPose from source
cd mmpose
pip install -e .

# Install MMAction2 from source  
cd mmaction2
pip install -e .

# Install dependencies
pip install -r mmpose/requirements.txt
pip install -r mmaction2/requirements.txt
```

### Testing
```bash
# Run MMPose tests
cd mmpose
pytest tests/

# Run MMAction2 tests
cd mmaction2
pytest tests/

# Run specific test modules
pytest tests/test_models/
pytest tests/test_datasets/
```

### Training & Evaluation

#### MMPose
```bash
# Train pose estimation model
python mmpose/tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py

# Test pose estimation model
python mmpose/tools/test.py configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py checkpoints/model.pth

# Distributed training/testing
bash mmpose/tools/dist_train.sh configs/path/to/config.py 8
bash mmpose/tools/dist_test.sh configs/path/to/config.py checkpoints/model.pth 8
```

#### MMAction2
```bash
# Train action recognition model
python mmaction2/tools/train.py configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py

# Test action recognition model
python mmaction2/tools/test.py configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py checkpoints/model.pth

# Distributed training/testing
bash mmaction2/tools/dist_train.sh configs/path/to/config.py 8
bash mmaction2/tools/dist_test.sh configs/path/to/config.py checkpoints/model.pth 8
```

### Demo Applications

#### MMPose Demos
```bash
# Image demo with pose detection
python mmpose/demo/image_demo.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py \
    checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input tests/data/coco/000000000785.jpg \
    --output-root vis_results

# Video demo
python mmpose/demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py \
    checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input demo/demo.mp4 \
    --output-root vis_results
```

#### Violence Detection Application
```bash
# Run the violence detection web interface
cd rtmo_gcn_test
python main.py --server-name 0.0.0.0 --server-port 7862
```

## Architecture Overview

### Project Structure

```
mmlabs/
├── mmpose/                    # Pose estimation framework
│   ├── mmpose/               # Core package
│   │   ├── apis/            # High-level inference APIs
│   │   ├── models/          # Model implementations
│   │   │   ├── backbones/   # Feature extractors (ResNet, HRNet, Swin)
│   │   │   ├── heads/       # Task-specific heads (heatmap, regression)
│   │   │   ├── pose_estimators/ # Top-down, bottom-up models
│   │   │   └── losses/      # Loss functions
│   │   ├── datasets/        # Dataset implementations
│   │   ├── codecs/          # Label encoding/decoding
│   │   └── evaluation/      # Metrics and evaluation
│   ├── configs/             # Model configurations
│   ├── tools/               # Training/testing scripts
│   └── checkpoints/         # Pre-trained models
├── mmaction2/                # Action recognition framework
│   ├── mmaction/            # Core package
│   │   ├── apis/            # High-level inference APIs
│   │   ├── models/          # Model implementations
│   │   │   ├── backbones/   # 2D/3D CNN backbones
│   │   │   ├── heads/       # Classification heads
│   │   │   ├── recognizers/ # Action recognition models
│   │   │   └── losses/      # Loss functions
│   │   ├── datasets/        # Dataset implementations
│   │   └── evaluation/      # Metrics and evaluation
│   ├── configs/             # Model configurations
│   ├── tools/               # Training/testing scripts
│   └── work_dirs/           # Training outputs
└── rtmo_gcn_test/           # Violence detection application
    ├── main.py              # Gradio web interface
    ├── pose_detector.py     # RTMO pose detection wrapper
    └── action_classifier.py # GCN action classification wrapper
```

### Key Components

#### MMPose Architecture
- **Registry System**: Component registration using OpenMMLab registry pattern
- **Config System**: YAML/Python configs for reproducible experiments
- **Data Pipeline**: Flexible transform pipelines for different modalities
- **Model Types**: Top-down (person detection + pose), bottom-up (keypoint detection + grouping), one-stage (end-to-end)
- **Key Models**: RTMPose (real-time), RTMO (one-stage multi-person), RTMW (whole-body)

#### MMAction2 Architecture
- **Modular Design**: Separate components for different video understanding tasks
- **Supported Tasks**: Action recognition, temporal action localization, spatio-temporal action detection, skeleton-based action recognition
- **Data Formats**: Video clips, extracted frames, skeleton sequences, audio features
- **Key Models**: SlowFast, I3D, TSN/TSM, ST-GCN, PoseC3D

#### Violence Detection Application
- **Real-time Pipeline**: RTMO pose detection → sequence buffering → GCN classification
- **Web Interface**: Gradio-based UI for model configuration and inference
- **Output Formats**: CSV logging, confusion matrices, real-time visualization
- **Multi-threading**: Separate threads for processing and UI updates

### Configuration System

Both frameworks use hierarchical configs:
- `_base_/`: Base configurations for models, datasets, schedules
- Task-specific configs inherit from base configs
- Runtime configs for training/testing settings
- Easy experimentation through config composition

### Model Deployment Patterns

1. **Single Model Inference**: Direct API calls for individual predictions
2. **Batch Processing**: Efficient processing of multiple inputs
3. **Real-time Applications**: Streaming inference with optimized models
4. **Multi-model Pipelines**: Chaining different models (pose → action)

## Development Guidelines

### Adding New Models
1. Implement model in appropriate package (`mmpose/models/` or `mmaction/models/`)
2. Register model using `@MODELS.register_module()`
3. Create configuration file following existing patterns
4. Add unit tests and documentation

### Dataset Integration
1. Implement dataset class inheriting from base dataset
2. Create data transforms for preprocessing
3. Register dataset and transforms
4. Convert annotations to expected format (COCO-style for pose, pickle for action)

### Custom Applications
- Use high-level APIs (`init_model`, `inference_*`) for quick prototyping
- Leverage existing transforms and data structures
- Follow OpenMMLab coding standards and patterns
- Test with both single-GPU and distributed setups

## Important Notes

- **Model Checkpoints**: Store in `checkpoints/` directories, download via OpenMMLab model zoo
- **Data Preparation**: Follow dataset-specific preparation guides in `tools/data/`
- **GPU Memory**: Monitor memory usage, adjust batch sizes and model sizes as needed
- **Version Compatibility**: Ensure MMEngine, MMCV versions are compatible with MMPose/MMAction2
- **Registry**: Always register new components in appropriate registries for proper discovery

## Development Memories

- 코드를 작성할때 항상 주석을 작성하고, 로그에 이모지 아이콘은 사용하지 말아죠
- 클로드 너의 위치는 우분투 호스트 경로이고, 코드 디버깅과 개발환경은 도커 컨테이너야 
  - /home/gaonpf/hsnam/mmlabs 는 /workspace 와 매핑돼 
  - docker exec -it mmlabs bash로 컨테이너 내부 진입이 가능해. 다만 디버깅은 토큰 비용이 많이 발생하니까 내가 직접 실행할꺼야 