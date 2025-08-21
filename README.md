# Recognizer - 실시간 폭력 감지 시스템

**MMPose + MMAction2 기반 통합 행동 분석 프레임워크**

RTMO 포즈 추정, ByteTrack 추적, ST-GCN 행동 분류를 통합한 실시간 폭력 감지 및 비디오 분석 시스템

##  주요 특징

- **실시간 폭력 감지**: RTMO + ST-GCN 기반 고속 추론
- **다중 추론 백엔드**: PyTorch, ONNX, TensorRT 지원
- **완전 모듈화**: 8개 독립 모드 (추론 3개 + 어노테이션 5개)
- **이벤트 관리 시스템**: 실시간 이벤트 탐지 및 로깅
- **고성능 최적화**: ONNX/TensorRT 최적화 지원
- **확장 가능**: 새로운 모드 및 모델 추가 용이

##  빠른 시작

### 환경 요구사항
- Python 3.8+
- CUDA 12.1+ (GPU 추론용)
- Docker (권장)

### Docker 환경 설정
```bash
# 컨테이너 진입
docker exec -it mmlabs bash

# 작업 디렉토리 이동
cd /workspace/recognizer
```

### 의존성 설치
```bash
# MMPose 설치
cd /workspace/mmpose
pip install -e .

# MMAction2 설치
cd /workspace/mmaction2
pip install -e .

# 추가 의존성
pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### 기본 사용법
```bash
# 모드 목록 확인
python main.py --list-modes

# 기본 실행 (실시간 모드)
python main.py

# 특정 모드 실행
python main.py --mode inference.analysis
python main.py --mode annotation.stage1
```

##  가중치 및 데이터셋 (https://192.168.190.100:5001/)

### 가중치 파일 위치

**RTMO 포즈 추정 모델:**
```bash
# 체크포인트(pytorch, onnx, tensorrt) & 설정파일
/aivanas/gaon/weights/action_recognition/1. 3b665fe5aa/rtmo

```

**ST-GCN 행동 분류 모델:**
```bash
# 체크포인트 & 설정파일
/aivanas/gaon/weights/action_recognition/1. 3b665fe5aa/stgcnpp
```

### 데이터셋 위치

**UBI-FIGHTS 데이터셋:**
```bash
# 폭력 영상
/aivanas/raw/surveillance/action/violence/action_recognition/data/UBI_FIGHTS/videos/fight/

# 정상 영상
/aivanas/raw/surveillance/action/violence/action_recognition/data/UBI_FIGHTS/videos/normal/
```

**RWF-2000 데이터셋:**
```bash
# 학습용 데이터셋
/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000/

# 어노테이션 파일
/aivanas/annotations/RWF-2000/
```

### 로컬 설정

로컬 환경에서 사용할 경우 `configs/config.yaml`에서 경로를 다음과 같이 수정:

```yaml
models:
  pose_estimation:
    pth:
      checkpoint_path: /workspace/mmpose/checkpoints/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth
    onnx:
      model_path: /workspace/mmpose/checkpoints/end2end.onnx
    tensorrt:
      model_path: /workspace/mmpose/checkpoints/rtmo.trt
      
  action_classification:
    checkpoint_path: /workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable/best_acc_top1_epoch_14.pth
```

##  시스템 구조

```
recognizer/
├── main.py                     # 통합 실행기
├── configs/config.yaml         # 통합 설정 파일
├── core/                       # 모드 관리 엔진
│   ├── mode_manager.py         # 통합 모드 매니저
│   ├── inference_modes.py      # 추론 모드들
│   └── annotation_modes.py     # 어노테이션 모드들
├── pose_estimation/            # 포즈 추정 모듈
│   └── rtmo/                   # RTMO 구현체
├── action_classification/      # 행동 분류 모듈
│   └── stgcn/                  # ST-GCN 구현체
├── tracking/                   # 객체 추적 모듈
│   └── bytetrack/             # ByteTracker 구현체
├── events/                     # 이벤트 관리 시스템
├── visualization/              # 시각화 모듈
├── utils/                      # 공통 유틸리티
└── docs/                       # API 문서
```

##  8개 실행 모드

### 추론 모드 (Inference)

#### 1. `inference.analysis` - 분석 모드
**목적**: 영상 → JSON/PKL 파일 생성 (백그라운드 분석)
```bash
python main.py --mode inference.analysis
```
- 전체 영상 완전 분석
- 폭력/비폭력 분류 결과 저장
- JSON 결과 + PKL 데이터 출력

#### 2. `inference.realtime` - 실시간 모드  
**목적**: 실시간 폭력 감지 + 이벤트 알림
```bash
python main.py --mode inference.realtime
```
- 실시간 영상 스트림 처리
- 라이브 오버레이 표시
- 폭력 이벤트 실시간 탐지
- 선택적 결과 영상 저장

#### 3. `inference.visualize` - 시각화 모드
**목적**: PKL 파일 + 원본 영상 → 오버레이 영상
```bash
python main.py --mode inference.visualize
```
- 기존 분석 결과 시각화
- 고품질 오버레이 생성
- 배치 처리 지원

### 어노테이션 모드 (Annotation)

#### 4. `annotation.stage1` - 포즈 추정
**목적**: 비디오 → 포즈 추정 PKL 파일
```bash
python main.py --mode annotation.stage1
```
- RTMO 포즈 추정만 수행
- 키포인트 데이터 저장
- 다음 단계 준비

#### 5. `annotation.stage2` - 트래킹 및 정렬
**목적**: 포즈 PKL → 트래킹/정렬 PKL 파일
```bash
python main.py --mode annotation.stage2
```
- ByteTrack 객체 추적
- 복합 점수 기반 정렬
- 고품질 추적 데이터

#### 6. `annotation.stage3` - 데이터셋 통합
**목적**: 비디오별 PKL → train/val/test 통합 PKL
```bash
python main.py --mode annotation.stage3
```
- 데이터셋 분할 (7:1.5:1.5)
- 모델 학습용 형식 변환
- 메타데이터 생성

#### 7. `annotation.visualize` - 어노테이션 시각화
**목적**: 각 stage별 결과 시각화
```bash
python main.py --mode annotation.visualize
```
- stage1: 포즈 키포인트 표시
- stage2: 추적 ID + 정렬 순위
- stage3: 데이터셋 통계

##  설정 관리

### 통합 설정 파일 (`config.yaml`)

```yaml
# 기본 실행 모드
mode: "inference.analysis"

# 추론 모드 설정
inference:
  analysis:
    input: "video.mp4"
    output_dir: "output/analysis"
  
  realtime:
    input: "video.mp4"
    save_output: false
    display_width: 1280
    display_height: 720
  
  visualize:
    results_dir: "output/analysis"
    video_file: "video.mp4"
    save_mode: false

# 어노테이션 모드 설정
annotation:
  stage1:
    input_dir: "/workspace/videos"
    output_dir: "output/stage1"
  
  stage2:
    poses_dir: "output/stage1"
    output_dir: "output/stage2"
  
  stage3:
    tracking_dir: "output/stage2"
    output_dir: "output/stage3"
    split_ratios: {train: 0.7, val: 0.15, test: 0.15}
  
  visualize:
    stage: "stage2"
    results_dir: "output/stage2"
    video_dir: "/workspace/videos"

# 모델 설정 (모든 모드 공통)
models:
  pose_estimation: {...}
  tracking: {...}
  action_classification: {...}
```

### 실행 인자 (최소화)

```bash
python main.py [OPTIONS]

OPTIONS:
  --config FILE        설정 파일 경로 (기본: config.yaml)
  --mode MODE         실행 모드 오버라이드
  --log-level LEVEL   로그 레벨 (DEBUG/INFO/WARNING/ERROR)
  --list-modes        사용 가능한 모드 목록
```

##  워크플로우 예시

### 완전한 분석 워크플로우
```bash
# 1단계: 분석 수행
python main.py --mode inference.analysis
# → output/analysis/json/, pkl/ 생성

# 2단계: 결과 시각화
python main.py --mode inference.visualize
# → 실시간 오버레이 또는 비디오 저장
```

### 어노테이션 파이프라인
```bash
# 1단계: 포즈 추정
python main.py --mode annotation.stage1
# → output/stage1/*.pkl

# 2단계: 트래킹 및 정렬
python main.py --mode annotation.stage2
# → output/stage2/*.pkl

# 3단계: 데이터셋 통합
python main.py --mode annotation.stage3
# → output/stage3/train.pkl, val.pkl, test.pkl

# 4단계: 결과 확인
python main.py --mode annotation.visualize
# → stage별 시각화
```

##  고급 사용법

### 사용자 정의 설정
```bash
# 커스텀 설정 파일
python main.py --config my_config.yaml --mode inference.analysis

# 특정 로그 레벨
python main.py --mode annotation.stage1 --log-level DEBUG
```

### 배치 처리
```yaml
# config.yaml에서 폴더 처리 설정
inference:
  analysis:
    input_dir: "/workspace/videos"  # 폴더 처리
    output_dir: "output/batch"
```

### 성능 최적화
```yaml
# 성능 설정
performance:
  device: "cuda:0"
  window_size: 100
  window_stride: 50
  batch_size: 8
```

##  AI 모델 아키텍처

### 포즈 추정 (RTMO)
- **입력**: 비디오 프레임
- **출력**: 17개 키포인트 좌표
- **특징**: 실시간 다중 객체 지원

### 객체 추적 (ByteTrack)
- **입력**: 프레임별 포즈 박스
- **출력**: 추적 ID + 궤적
- **특징**: 하이브리드 매칭 (IoU + 키포인트)

### 행동 분류 (STGCN)
- **입력**: 100프레임 키포인트 시퀀스
- **출력**: Fight/NonFight 확률
- **특징**: 시공간 그래프 컨볼루션

##  출력 형식

### JSON 결과 (`results.json`)
```json
{
  "input_video": "video.mp4",
  "total_frames": 3000,
  "total_windows": 59,
  "classification_results": [
    {
      "window_id": 0,
      "window_start": 0,
      "window_end": 100,
      "predicted_class": "NonFight",
      "confidence": 0.823,
      "probabilities": [0.823, 0.177]
    }
  ]
}
```

### PKL 데이터 구조
- **프레임 포즈**: `{frame_id: [(x,y,score), ...], ...}`
- **윈도우 어노테이션**: `{keypoints, scores, tracking_ids, ...}`
- **분류 결과**: `{window_data, predictions, metadata}`

##  문제 해결

### 자주 발생하는 오류

1. **"Failed to create pose_estimator module"**
   ```bash
   # 모델 경로 확인
   ls /workspace/mmpose/checkpoints/
   ```

2. **"Input directory does not exist"**
   ```bash
   # 입력 경로 확인
   python main.py --mode inference.analysis --log-level DEBUG
   ```

3. **"CUDA out of memory"**
   ```yaml
   # config.yaml에서 배치 크기 조정
   performance:
     batch_size: 4  # 기본값 8에서 감소
   ```

### 디버깅 모드
```bash
# 상세 로그로 문제 진단
python main.py --mode [MODE] --log-level DEBUG
```

##  관련 프로젝트

- **MMPose**: 포즈 추정 프레임워크
- **MMAction2**: 행동 인식 프레임워크  
- **ByteTrack**: 다중 객체 추적