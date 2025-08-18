# Recognizer 사용자 가이드

**완전 모듈화된 비디오 분석 시스템 상세 사용법**

## 📚 목차

1. [빠른 시작](#빠른-시작)
2. [모드별 상세 가이드](#모드별-상세-가이드)
3. [설정 파일 커스터마이징](#설정-파일-커스터마이징)
4. [고급 워크플로우](#고급-워크플로우)
5. [문제 해결](#문제-해결)

## 🚀 빠른 시작

### 1. 시스템 확인
```bash
# 모든 모드 확인
python main.py --list-modes

# 기본 설정 확인
python main.py --mode inference.analysis --log-level DEBUG
```

### 2. 첫 번째 분석
```bash
# 기본 분석 실행
python main.py

# 결과 확인
ls output/analysis/
```

## 🎯 모드별 상세 가이드

### Inference 모드

#### 📊 inference.analysis - 분석 모드

**용도**: 비디오를 완전히 분석하여 JSON/PKL 파일 생성

**특징**:
- 전체 비디오 완전 처리 (20초 제한 없음)
- 시각화 없이 데이터만 생성
- 배치 처리 지원

**설정 예시**:
```yaml
inference:
  analysis:
    input: "/path/to/video.mp4"           # 단일 파일
    # input_dir: "/path/to/videos/"       # 폴더 처리
    output_dir: "output/analysis"
```

**실행**:
```bash
# 단일 파일
python main.py --mode inference.analysis

# 커스텀 설정
python main.py --config custom.yaml --mode inference.analysis
```

**출력**:
```
output/analysis/
├── json/
│   └── video_results.json
└── pkl/
    ├── video_frame_poses.pkl
    └── video_rtmo_poses.pkl
```

#### 🎥 inference.realtime - 실시간 모드

**용도**: 실시간 비디오 스트림 처리 및 라이브 디스플레이

**특징**:
- 실시간 오버레이 표시
- 선택적 결과 비디오 저장
- 키보드 인터랙션 지원

**설정 예시**:
```yaml
inference:
  realtime:
    input: "/path/to/video.mp4"
    save_output: true                     # 비디오 저장 여부
    output_path: "output/realtime.mp4"
    display_width: 1280
    display_height: 720
```

**실행**:
```bash
python main.py --mode inference.realtime
```

**키보드 조작**:
- `q`: 종료
- `space`: 일시정지/재생
- `s`: 스크린샷 저장

#### 🎨 inference.visualize - 시각화 모드

**용도**: 기존 분석 결과를 고품질 오버레이로 시각화

**특징**:
- PKL 파일 기반 정확한 재현
- 고품질 오버레이 생성
- 실시간 표시 또는 파일 저장

**설정 예시**:
```yaml
inference:
  visualize:
    results_dir: "output/analysis"        # 분석 결과 디렉토리
    video_file: "/path/to/video.mp4"      # 원본 비디오
    # video_dir: "/path/to/videos/"       # 폴더 시각화
    save_mode: true                       # true: 저장, false: 실시간 표시
    save_dir: "output/overlay"
```

**실행**:
```bash
# 실시간 표시
python main.py --mode inference.visualize

# 파일 저장 (config.yaml에서 save_mode: true 설정)
python main.py --mode inference.visualize
```

### Annotation 모드

#### 🎯 annotation.stage1 - 포즈 추정

**용도**: 비디오에서 RTMO 포즈 추정만 수행

**특징**:
- 17개 키포인트 추출
- 다중 객체 지원
- 원시 포즈 데이터 저장

**설정 예시**:
```yaml
annotation:
  stage1:
    input_dir: "/workspace/raw_videos"
    output_dir: "output/stage1_poses"
```

**실행**:
```bash
python main.py --mode annotation.stage1
```

**출력**:
```
output/stage1_poses/
├── video1_poses.pkl
├── video2_poses.pkl
└── ...
```

#### 🔗 annotation.stage2 - 트래킹 및 정렬

**용도**: Stage1 결과를 바탕으로 객체 추적 및 정렬

**특징**:
- ByteTrack 기반 추적
- 복합 점수 계산
- 하이브리드 매칭 (IoU + 키포인트)

**설정 예시**:
```yaml
annotation:
  stage2:
    poses_dir: "output/stage1_poses"      # Stage1 결과
    output_dir: "output/stage2_tracking"
```

**실행**:
```bash
python main.py --mode annotation.stage2
```

**출력**:
```
output/stage2_tracking/
├── video1_tracking.pkl
├── video2_tracking.pkl
└── ...
```

#### 🗃️ annotation.stage3 - 데이터셋 통합

**용도**: 개별 비디오 결과를 학습용 데이터셋으로 통합

**특징**:
- train/val/test 분할 (7:1.5:1.5)
- 메타데이터 생성
- 모델 학습 형식 변환

**설정 예시**:
```yaml
annotation:
  stage3:
    tracking_dir: "output/stage2_tracking"
    output_dir: "output/stage3_dataset"
    split_ratios:
      train: 0.7
      val: 0.15
      test: 0.15
```

**실행**:
```bash
python main.py --mode annotation.stage3
```

**출력**:
```
output/stage3_dataset/
├── train.pkl
├── val.pkl
├── test.pkl
└── metadata.json
```

#### 👁️ annotation.visualize - 어노테이션 시각화

**용도**: 각 stage별 중간 결과 시각화

**특징**:
- stage별 맞춤 시각화
- 디버깅 및 검증 용도
- 실시간 또는 저장 모드

**설정 예시**:
```yaml
annotation:
  visualize:
    stage: "stage2"                       # stage1, stage2, stage3
    results_dir: "output/stage2_tracking"
    video_dir: "/workspace/raw_videos"
    save_mode: false
    save_dir: "output/annotation_overlay"
```

**실행**:
```bash
# Stage2 결과 시각화
python main.py --mode annotation.visualize

# Stage1 결과 시각화 (config.yaml에서 stage: "stage1" 설정)
python main.py --mode annotation.visualize
```

## ⚙️ 설정 파일 커스터마이징

### 기본 설정 구조

```yaml
# 기본 실행 모드
mode: "inference.analysis"

# 추론 모드 설정
inference:
  analysis: {...}
  realtime: {...}
  visualize: {...}

# 어노테이션 모드 설정  
annotation:
  stage1: {...}
  stage2: {...}
  stage3: {...}
  visualize: {...}

# 모델 설정 (모든 모드 공통)
models:
  pose_estimation: {...}
  tracking: {...}
  scoring: {...}
  action_classification: {...}

# 성능 설정
performance:
  device: "cuda:0"
  window_size: 100
  window_stride: 50
  batch_size: 8
```

### 모델 세부 설정

```yaml
models:
  pose_estimation:
    model_name: "rtmo"
    config_file: "/workspace/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
    checkpoint_path: "/workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
    device: "cuda:0"
    score_threshold: 0.2
    input_size: [640, 640]

  tracking:
    tracker_name: "bytetrack"
    frame_rate: 30
    track_thresh: 0.2
    track_buffer: 120
    match_thresh: 0.5

  action_classification:
    model_name: "stgcn"
    config_file: "/workspace/mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py"
    checkpoint_path: "/workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable/best_acc_top1_epoch_14.pth"
    device: "cuda:0"
    window_size: 100
    class_names: ["NonFight", "Fight"]
    confidence_threshold: 0.4
```

### 성능 최적화 설정

```yaml
performance:
  device: "cuda:0"
  window_size: 100                 # 분류 윈도우 크기
  window_stride: 50                # 윈도우 간격
  batch_size: 8                    # 배치 크기
  
  # 메모리 관리
  max_cache_size: 1000
  gc_interval: 100
  enable_garbage_collection: true

# 오류 처리
error_handling:
  continue_on_error: true
  max_consecutive_errors: 10
  error_recovery_strategy: "skip"
```

## 🔄 고급 워크플로우

### 1. 완전한 분석 파이프라인

```bash
#!/bin/bash
# complete_analysis.sh

echo "=== 1단계: 분석 수행 ==="
python main.py --mode inference.analysis

echo "=== 2단계: 결과 시각화 ==="
python main.py --mode inference.visualize

echo "=== 분석 완료 ==="
ls -la output/analysis/
ls -la output/overlay/
```

### 2. 어노테이션 파이프라인

```bash
#!/bin/bash
# annotation_pipeline.sh

echo "=== Stage 1: 포즈 추정 ==="
python main.py --mode annotation.stage1

echo "=== Stage 2: 트래킹 및 정렬 ==="
python main.py --mode annotation.stage2

echo "=== Stage 3: 데이터셋 통합 ==="
python main.py --mode annotation.stage3

echo "=== 결과 확인 ==="
python main.py --mode annotation.visualize

echo "=== 어노테이션 완료 ==="
ls -la output/stage3_dataset/
```

### 3. 배치 처리

```yaml
# batch_config.yaml
inference:
  analysis:
    input_dir: "/workspace/batch_videos"   # 폴더 지정
    output_dir: "output/batch_analysis"

performance:
  batch_size: 4                           # 메모리 절약
  window_stride: 25                       # 더 세밀한 분석
```

```bash
# 배치 처리 실행
python main.py --config batch_config.yaml --mode inference.analysis
```

### 4. 커스텀 모델 설정

```yaml
# custom_model_config.yaml
models:
  pose_estimation:
    score_threshold: 0.1               # 더 민감한 탐지
  
  action_classification:
    confidence_threshold: 0.3          # 더 보수적인 분류
    window_size: 150                   # 더 긴 컨텍스트
```

## 🛠️ 문제 해결

### 일반적인 오류

#### 1. 모델 로딩 실패
```
ERROR: Failed to create pose_estimator module rtmo
```

**해결책**:
```bash
# 체크포인트 파일 확인
ls /workspace/mmpose/checkpoints/

# 설정에서 올바른 경로 지정
# config.yaml에서 checkpoint_path 수정
```

#### 2. CUDA 메모리 부족
```
CUDA out of memory
```

**해결책**:
```yaml
# config.yaml 수정
performance:
  batch_size: 4        # 8에서 4로 감소
  device: "cpu"        # GPU 대신 CPU 사용
```

#### 3. 입력 파일 없음
```
Input directory does not exist
```

**해결책**:
```bash
# 경로 확인
ls -la /path/to/input/

# 절대 경로 사용
# config.yaml에서 전체 경로 지정
```

#### 4. 권한 문제
```
Permission denied
```

**해결책**:
```bash
# 출력 디렉토리 권한 확인
chmod 755 output/

# Docker 컨테이너에서 실행
docker exec mmlabs bash -c "cd /workspace && python recognizer/main.py"
```

### 디버깅 가이드

#### 상세 로그 확인
```bash
python main.py --mode [MODE] --log-level DEBUG
```

#### 설정 검증
```bash
# 설정 파일 구문 검사
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# 모드 목록 확인
python main.py --list-modes
```

#### 성능 모니터링
```bash
# GPU 사용량 확인
nvidia-smi

# 메모리 사용량 확인
htop
```

### 성능 최적화 팁

1. **GPU 메모리 최적화**
   - batch_size 조정
   - window_size 감소
   - 가비지 컬렉션 활성화

2. **처리 속도 향상**
   - window_stride 증가
   - score_threshold 증가
   - 병렬 처리 활용

3. **정확도 향상**
   - window_size 증가
   - score_threshold 감소
   - confidence_threshold 조정

## 📋 체크리스트

### 설치 확인
- [ ] MMPose 설치 완료
- [ ] MMAction2 설치 완료
- [ ] CUDA 환경 설정
- [ ] 모델 체크포인트 다운로드

### 실행 전 확인
- [ ] 입력 파일/디렉토리 존재
- [ ] 출력 디렉토리 권한
- [ ] GPU/CPU 자원 충분
- [ ] 설정 파일 구문 정확

### 결과 검증
- [ ] JSON 파일 생성
- [ ] PKL 파일 생성  
- [ ] 오버레이 비디오 품질
- [ ] 로그 메시지 확인

---

**이 가이드를 통해 Recognizer의 모든 기능을 효과적으로 활용하세요!** 🚀