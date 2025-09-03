# Recognizer 사용자 가이드

**완전 모듈화된 비디오 분석 시스템 종합 사용법**

## 📚 목차

1. [빠른 시작](#빠른-시작)
2. [시스템 개요](#시스템-개요)
3. [모드별 상세 가이드](#모드별-상세-가이드)
4. [설정 파일 커스터마이징](#설정-파일-커스터마이징)
5. [ONNX 모델 사용법](#onnx-모델-사용법)
6. [멀티프로세스 처리](#멀티프로세스-처리)
7. [고급 워크플로우](#고급-워크플로우)
8. [문제 해결](#문제-해결)

## 🚀 빠른 시작

### 1. 환경 확인
```bash
# Docker 컨테이너 환경에서 실행 (필수)
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --list-modes"

# 시스템 상태 확인
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis --log-level INFO"
```

### 2. 첫 번째 분석
```bash
# 기본 분석 실행
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py"

# 결과 확인
ls /workspace/recognizer/output/analysis/
```

## 🏗️ 시스템 개요

### 아키텍처
Recognizer는 4단계 파이프라인으로 구성된 모듈화된 비디오 분석 시스템입니다:

1. **포즈 추정** (Pose Estimation)
   - RTMO PyTorch/ONNX/TensorRT 지원
   - 실시간 인체 키포인트 탐지

2. **객체 추적** (Object Tracking)  
   - ByteTracker 기반
   - 프레임 간 인물 ID 유지

3. **복합 점수 계산** (Composite Scoring)
   - 움직임 기반 점수 계산
   - 영역 기반 상호작용 분석

4. **행동 분류** (Action Classification)
   - ST-GCN++ PyTorch/ONNX 지원
   - 폭력/비폭력 분류

### 지원 모델
- **RTMO**: PyTorch, ONNX, TensorRT
- **ST-GCN++**: PyTorch, ONNX (온도 스케일링 적용)
- **ByteTracker**: 객체 추적

## 🎯 모드별 상세 가이드

### Inference 모드

#### 📊 inference.analysis - 분석 모드

**용도**: 비디오를 완전히 분석하여 JSON/PKL 파일 생성

**특징**:
- 전체 비디오 완전 처리
- 시각화 없이 데이터만 생성  
- 배치 처리 지원
- 평가 모드 내장 (차트, 혼동행렬, 보고서 생성)

**실행 방법**:
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis"
```

**설정 예시**:
```yaml
inference:
  analysis:
    input: "/path/to/video.mp4"           # 단일 파일
    input: "/path/to/videos/"             # 폴더 (모든 비디오)
    output_dir: "output/analysis"
    
    # 평가 모드 활성화
    evaluation:
      enabled: true
      ground_truth_dir: "/path/to/labels"
      charts:
        enabled: true
        confidence_histogram: true
        temporal_analysis: true
      confusion_matrix:
        enabled: true
        normalize: true
      report:
        enabled: true
        format: "html"  # html, pdf
```

#### 🎬 inference.realtime - 실시간 모드

**용도**: 실시간 비디오 스트림 처리 및 시각화

**특징**:
- 실시간 디스플레이
- 이벤트 감지 및 알림
- 성능 모니터링
- 웹캠/비디오 파일 지원

**실행 방법**:
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.realtime"
```

**설정 예시**:
```yaml
inference:
  realtime:
    input: 0                              # 웹캠 (0, 1, 2...)
    input: "/path/to/video.mp4"           # 비디오 파일
    
    # 이벤트 설정
    event_manager:
      alert_threshold: 0.8
      min_consecutive_detections: 3
      cooldown_duration: 5.0
      
    # 시각화 설정
    visualize:
      display: true
      save_video: true
      overlay_keypoints: true
      overlay_tracks: true
```

#### 🎨 inference.visualize - 시각화 모드

**용도**: 기존 PKL 파일을 이용한 오버레이 비디오 생성

**특징**:
- PKL 파일 기반 시각화
- 고품질 오버레이 생성
- 프레임별 상세 정보 표시

**실행 방법**:
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.visualize"
```

### Annotation 모드

#### 🏗️ annotation.pipeline - 통합 파이프라인

**용도**: 전체 어노테이션 과정을 한번에 실행

**특징**:
- Stage 1-3 자동 연결
- 최적화된 메모리 사용
- 진행상황 모니터링

**실행 방법**:
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.pipeline"
```

#### 📝 annotation.stage1 - 포즈 추정

**용도**: 원본 비디오에서 포즈 추정 결과 생성

**실행 방법**:
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.stage1"
```

#### 🎯 annotation.stage2 - 윈도우 생성

**용도**: 포즈 데이터를 분류용 윈도우로 변환

#### 🔍 annotation.stage3 - 행동 분류

**용도**: 윈도우 데이터를 이용한 최종 분류 수행

#### 🎨 annotation.visualize - 어노테이션 시각화

**용도**: 어노테이션 결과 시각화

## ⚙️ 설정 파일 커스터마이징

### 기본 구조
```yaml
# config.yaml
models:
  # 포즈 추정 모델
  pose_estimation:
    inference_mode: onnx        # pth, onnx, tensorrt
    model_name: rtmo_l          # rtmo_s, rtmo_m, rtmo_l
    device: cuda:0
    score_threshold: 0.3
    
    # ONNX 설정
    onnx:
      model_path: "/path/to/rtmo.onnx"
    
    # PyTorch 설정  
    pth:
      config_file: "/path/to/config.py"
      checkpoint_path: "/path/to/model.pth"

  # 행동 분류 모델  
  action_classification:
    model_name: stgcn_onnx      # stgcn, stgcn_onnx
    checkpoint_path: "/path/to/stgcn.onnx"
    confidence_threshold: 0.4
    window_size: 100
    max_persons: 4
    
  # 추적 모델
  tracking:
    model_name: bytetrack
    track_thresh: 0.5
    min_box_area: 200

  # 점수 계산
  scoring:
    model_name: movement_based
    distance_threshold: 100.0
```

### 모드별 설정

#### Inference 설정
```yaml
inference:
  analysis:
    input: "/data/videos/"
    output_dir: "output/analysis"
    batch_size: 1
    
    evaluation:
      enabled: true
      ground_truth_dir: "/data/labels"
      
  realtime:
    input: 0  # 웹캠 또는 비디오 경로
    display_window_size: [1280, 720]
    fps_limit: 30
    
  visualize:
    input_pkl: "output/analysis/results.pkl"  
    input_video: "/data/input.mp4"
    output_video: "output/visualized.mp4"
```

#### Annotation 설정
```yaml
annotation:
  input: "/data/training_videos/"
  output_dir: "output/annotations"
  
  # 멀티프로세스 설정
  multi_process:
    enabled: false
    num_processes: 4
    gpus: [0, 1]
```

## 🔧 ONNX 모델 사용법

### ONNX 모델 장점
- **빠른 추론 속도**: PyTorch 대비 2-3배 빠름
- **메모리 효율성**: 더 적은 VRAM 사용
- **배포 편의성**: 의존성 최소화

### ST-GCN++ ONNX 변환

1. **모델 변환**:
```bash
cd /workspace/mmaction2/tools/deployment
python export_onnx_gcn.py \
    --config /workspace/mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py \
    --checkpoint /workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-l_RWF2000plus_stable/best_acc_top1_epoch_30.pth \
    --output-file /workspace/mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.onnx \
    --num_frames 100 \
    --num_person 4
```

2. **설정 파일 수정**:
```yaml
models:
  action_classification:
    model_name: stgcn_onnx
    checkpoint_path: /workspace/mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.onnx
    input_format: stgcn_onnx
```

### Temperature Scaling

ONNX 모델은 raw logits를 출력하므로 온도 스케일링이 자동 적용됩니다:
- **Temperature**: 0.005 (자동 조정)
- **출력 범위**: 0-1 확률값
- **PyTorch 호환**: 동일한 확률 분포

## ⚡ 멀티프로세스 처리

### 자동 활성화 조건
```yaml
annotation:
  multi_process:
    enabled: true           # 명시적 활성화
    num_processes: 8        # 프로세스 수
    gpus: [0, 1, 2, 3]     # GPU 할당 (라운드 로빈)
```

### 명령줄 실행
```bash
# 멀티프로세스 어노테이션
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --multi-process --num-processes 8 --gpus 0,1,2,3"

# GPU별 부하 모니터링
nvidia-smi -l 1
```

### 성능 최적화
- **프로세스 수**: CPU 코어 수와 동일 권장
- **GPU 할당**: 라운드 로빈 방식 자동 분배
- **메모리 관리**: 프로세스별 독립적 메모리 사용

## 🔬 고급 워크플로우

### 1. 대규모 데이터셋 처리
```bash
# 1단계: 멀티프로세스 어노테이션
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode annotation.pipeline --multi-process --num-processes 8"

# 2단계: 결과 분석
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis"

# 3단계: 시각화 생성  
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.visualize"
```

### 2. 모델 성능 비교
```yaml
# PyTorch vs ONNX 성능 비교
models:
  action_classification:
    model_name: stgcn        # 첫 번째 실행
    # model_name: stgcn_onnx # 두 번째 실행
```

### 3. 커스텀 파이프라인 구축
```python
# custom_pipeline.py
from core.inference_modes import AnalysisMode
from utils.factory import ModuleFactory

# 커스텀 모델 등록
ModuleFactory.register_classifier(
    name='custom_model',
    classifier_class=CustomClassifier,
    default_config={'param1': 'value1'}
)
```

## 🛠️ 문제 해결

### 일반적인 문제

#### 1. MMCV 호환성 오류
```bash
# 해결책: Docker 컨테이너 사용
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py"
```

#### 2. ONNX 모델 로딩 실패
```bash
# 모델 경로 확인
ls -la /workspace/mmaction2/checkpoints/

# 권한 확인
chmod 644 /workspace/mmaction2/checkpoints/*.onnx
```

#### 3. GPU 메모리 부족
```yaml
# 설정 최적화
models:
  action_classification:
    max_persons: 2        # 기본값 4에서 감소
  pose_estimation:
    input_size: [256, 192] # 입력 크기 감소
```

#### 4. 실시간 모드 성능 문제
```yaml
inference:
  realtime:
    fps_limit: 15         # FPS 제한
    skip_frames: 2        # 프레임 스킵
```

### 로그 분석

#### 디버그 모드 실행
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --log-level DEBUG"
```

#### 핵심 로그 메시지
- `STGCN ONNX RESULT`: 분류 결과 확인
- `raw_scores`: ONNX 원본 출력 확인
- `Probabilities after softmax`: 최종 확률값
- `Processing inference result`: PyTorch 결과 확인

### 성능 모니터링

#### 시스템 리소스
```bash
# GPU 사용량
nvidia-smi

# 메모리 사용량
free -h

# CPU 사용량  
htop
```

#### 파이프라인 성능
- **Pose Estimation FPS**: 포즈 추정 속도
- **Classification FPS**: 행동 분류 속도  
- **Overall Processing FPS**: 전체 파이프라인 속도
- **Memory Usage**: 메모리 사용량

## 📚 추가 문서

- [API 참조](docs/API_REFERENCE.md) - 상세 API 문서
- [데이터 구조](docs/DATA_STRUCTURE.md) - 내부 데이터 구조
- [실시간 아키텍처](docs/REALTIME_INFERENCE_ARCHITECTURE.md) - 실시간 처리 구조
- [ONNX 최적화 가이드](docs/ONNX_OPTIMIZATION_GUIDE.md) - ONNX 성능 최적화
- [평가 가이드](docs/EVALUATION_GUIDE.md) - 모델 평가 방법

## 🔧 도움 받기

### 이슈 리포트
문제가 발생하면 다음 정보와 함께 이슈를 등록해주세요:
1. 실행 명령어
2. 에러 로그 (--log-level DEBUG)
3. 시스템 환경 (GPU, CUDA 버전)
4. 설정 파일 내용

### 연락처
- 기술 지원: [기술 지원 연락처]
- 문서 개선: [문서 개선 요청]

---

*이 가이드는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.*