# Recognizer

**Real-time Human Action Recognition and Analysis System**

MMPose 기반의 실시간 동작 인식 및 분석 시스템으로, 포즈 추정, 객체 추적, 동작 분류, 이벤트 감지 기능을 통합 제공합니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 주요 기능

### 실시간 동작 감지
- **Fight Detection**: 폭력 행동 실시간 감지
- **Falldown Detection**: 낙상 상황 즉시 감지
- **Multi-Service**: 여러 동작 동시 모니터링

### 고성능 처리
- **30 FPS**: 실시간 비디오 처리 (640x640)
- **<100ms**: 초저지연 이벤트 감지
- **Multi-GPU**: 대용량 데이터 병렬 처리

### 다양한 실행 모드
- **Realtime**: 실시간 스트림 분석
- **Analysis**: 배치 비디오 분석
- **Annotation**: 학습 데이터 자동 생성
- **Visualization**: 결과 시각화

## 시스템 아키텍처

```
Video Input → Pose Estimation → Object Tracking → Action Classification → Event Detection
     RTMO              ByteTracker           STGCN++              Real-time Alert
```

### 핵심 구성 요소
- **RTMO**: 실시간 다중인물 포즈 추정 (ONNX/TensorRT 최적화)
- **ByteTracker**: 안정적인 다중 객체 추적
- **STGCN++**: 시공간 그래프 기반 동작 분류
- **Event Manager**: 지능적 이벤트 감지 및 관리

## 빠른 시작

### 전제 조건
- NVIDIA GPU (RTX 3090+ 권장)
- Docker with NVIDIA Container Toolkit
- CUDA 11.8+

### 1. 실시간 처리 시작
```bash
# Docker 컨테이너 접속
docker exec -it mmlabs bash

# Recognizer 디렉토리로 이동
cd /workspace/recognizer

# 실시간 모드 실행
python3 main.py --mode inference.realtime
```

### 2. 배치 분석 실행
```bash
# config.yaml에서 입력 경로 설정
vim configs/config.yaml

# 분석 모드 실행
python3 main.py --mode inference.analysis
```

### 3. 학습 데이터 생성
```bash
# 3단계 파이프라인 실행
python3 main.py --mode annotation.stage1  # 포즈 추정
python3 main.py --mode annotation.stage2  # 객체 추적
python3 main.py --mode annotation.stage3  # 데이터셋 생성
```

## 성능 벤치마크

| 모델 | 백엔드 | FPS | 지연시간 | GPU 메모리 |
|------|--------|-----|----------|-----------|
| RTMO-L | PyTorch | 15 | ~150ms | 8GB |
| RTMO-L | ONNX | 25 | ~100ms | 6GB |
| RTMO-L | TensorRT | 35 | ~80ms | 4GB |

## 설정 가이드

### 기본 설정 (`configs/config.yaml`)
```yaml
# 실행 모드 선택
mode: inference.realtime

# 듀얼 서비스 설정
dual_service:
  enabled: true
  services: [fight, falldown]

# 성능 최적화
models:
  pose_estimation:
    inference_mode: onnx  # onnx | tensorrt | pth
```

### 모드별 설정
- **개발/테스트**: `inference_mode: pth`
- **운영환경**: `inference_mode: onnx` 또는 `tensorrt`
- **대용량 처리**: 멀티프로세싱 활성화

## 프로젝트 구조

```
recognizer/
├── action_classification/    # 동작 분류 모듈
├── configs/                 # 설정 파일
├── core/                    # 핵심 추론 모드
├── docs/                    # 📚 상세 문서
├── events/                  # 이벤트 관리
├── main.py                  # 🚀 메인 실행 파일
├── pipelines/              # 파이프라인 구현
├── pose_estimation/        # 포즈 추정 모듈
├── scoring/                # 점수 계산
├── tracking/               # 객체 추적
├── utils/                  # 유틸리티
└── visualization/          # 시각화
```

## 상세 문서

프로젝트의 상세한 설계와 사용법은 [`docs/`](docs/) 디렉토리를 참조하세요:

| 문서 | 설명 |
|------|------|
| [README](docs/README.md) | 문서 가이드 및 읽기 순서 |
| [Folder Structure](docs/01_folder_structure_guide.md) | 폴더 구조별 기능 설명 |
| [Data Structure](docs/02_pkl_data_structure_guide.md) | PKL 데이터 구조 상세 |
| [Pipeline Architecture](docs/03_pipeline_architecture_guide.md) | 파이프라인 아키텍처 |
| [UML Diagrams](docs/04_uml_diagrams.md) | 시스템 설계 다이어그램 |
| [Config Guide](docs/05_config_settings_guide.md) | 설정 가이드 |
| [Training Guide](docs/06_pose_stgcn_training_guide.md) | 모델 학습 가이드 |
| [Design Document](docs/07_software_design_document.md) | 소프트웨어 설계서 |

## 개발 가이드

### 새로운 동작 타입 추가
1. **분류기 구현**: `action_classification/` 에 새 분류기 추가
2. **스코어러 구현**: `scoring/` 에 새 점수 계산기 추가
3. **설정 추가**: `config.yaml` 에 모델 및 이벤트 설정
4. **팩토리 등록**: `main.py` 에서 모듈 등록

### 새로운 백엔드 추가
1. **추정기 구현**: `pose_estimation/` 에 새 백엔드 구현
2. **베이스 클래스 상속**: `BasePoseEstimator` 상속
3. **팩토리 등록**: 모듈 팩토리에 등록

## 실험 결과

### Fight Detection (RWF-2000 Dataset)
- **정확도**: 94.2%
- **정밀도**: 92.8%
- **재현율**: 95.1%
- **F1-Score**: 93.9%

### Falldown Detection (AI-Hub Dataset)
- **정확도**: 96.7%
- **정밀도**: 95.3%
- **재현율**: 97.8%
- **F1-Score**: 96.5%

## 시스템 요구사항

### 최소 요구사항
- **GPU**: NVIDIA GTX 1080 Ti (11GB)
- **CPU**: Intel i7-8700K or AMD Ryzen 7 2700X
- **RAM**: 16GB
- **Storage**: 500GB SSD

### 권장 요구사항
- **GPU**: NVIDIA RTX 3090 (24GB) 이상
- **CPU**: Intel i9-12900K or AMD Ryzen 9 5900X
- **RAM**: 32GB
- **Storage**: 1TB NVMe SSD

## 문제 해결

### 일반적인 문제
- **GPU 메모리 부족**: 배치 크기 감소 또는 ONNX/TensorRT 사용
- **느린 처리 속도**: TensorRT 백엔드 사용 및 멀티프로세싱 활성화
- **학습 안 됨**: 학습률 조정 및 배치 크기 증가

### 로그 확인
```bash
# 디버그 모드로 실행
python3 main.py --log-level DEBUG

# 로그 파일 확인
tail -f output/event_logs/events.log
```

