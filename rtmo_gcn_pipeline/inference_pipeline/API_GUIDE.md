# STGCN++ Violence Detection API Guide

## 개요

STGCN++ 폭력 검출 시스템은 RTMO 포즈 추정과 STGCN++ 그래프 신경망을 결합하여 비디오에서 폭력 행동을 실시간으로 검출하는 엔드투엔드 파이프라인입니다.

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [API 모듈](#api-모듈)
3. [사용법](#사용법)
4. [설정 가이드](#설정-가이드)
5. [성능 메트릭](#성능-메트릭)
6. [문제 해결](#문제-해결)

---

## 시스템 아키텍처

### 전체 파이프라인 흐름

```
[비디오 입력] → [RTMO 포즈 추정] → [Fight-우선 트래킹] → [STGCN++ 분류] → [성능 평가] → [오버레이 생성]
```

### 핵심 구성 요소

1. **RTMOPoseEstimator**: 실시간 다중 인물 포즈 추정
2. **FightPrioritizedTracker**: Fight-우선 5영역 분할 트래킹 시스템
3. **STGCNActionClassifier**: 시공간 그래프 신경망 기반 행동 분류
4. **MetricsCalculator**: 종합적인 성능 평가 도구
5. **VideoOverlayGenerator**: 시각화 및 오버레이 비디오 생성

---

## API 모듈

### 1. EndToEndPipeline 클래스

메인 파이프라인을 관리하는 핵심 클래스입니다.

#### 초기화

```python
from main_pipeline import EndToEndPipeline

pipeline = EndToEndPipeline(
    pose_config="/path/to/rtmo_config.py",
    pose_checkpoint="/path/to/rtmo_checkpoint.pth",
    gcn_config="/path/to/stgcn_config.py", 
    gcn_checkpoint="/path/to/stgcn_checkpoint.pth",
    device="cuda:0"
)
```

**매개변수:**
- `pose_config`: RTMO 모델 설정 파일 경로
- `pose_checkpoint`: RTMO 체크포인트 파일 경로
- `gcn_config`: STGCN++ 모델 설정 파일 경로
- `gcn_checkpoint`: STGCN++ 체크포인트 파일 경로
- `device`: 추론 디바이스 (`cuda:0`, `cpu`)

#### 주요 메서드

##### process_single_video()

단일 비디오 파일을 처리합니다.

```python
result = pipeline.process_single_video(
    video_path="/path/to/video.mp4",
    ground_truth_label=1,  # 0: NonFight, 1: Fight
    generate_overlay=True
)
```

**반환값:**
```python
{
    'video_path': str,
    'video_name': str,
    'ground_truth_label': int,
    'pose_estimation': {
        'total_frames': int,
        'valid_frames': int
    },
    'tracking': {
        'sequence_length': int,
        'selected_keypoints_shape': tuple,
        'selected_scores_shape': tuple
    },
    'classification': {
        'prediction': int,           # 0 또는 1
        'confidence': float,         # 0.0 ~ 1.0
        'prediction_label': str,     # 'Fight' 또는 'NonFight'
        'window_predictions': list,
        'window_confidences': list
    },
    'confidence_analysis': dict,
    'overlay_path': str,
    'processing_time': float,
    'status': str                   # 'success', 'failed', 'empty'
}
```

##### process_batch_videos()

다중 비디오 파일을 배치로 처리합니다.

```python
batch_result = pipeline.process_batch_videos(
    video_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"],
    ground_truth_labels=[1, 0],
    generate_overlay=True,
    output_dir="./results"
)
```

**반환값:**
```python
{
    'summary': {
        'total_videos': int,
        'successful': int,
        'failed': int,
        'success_rate': float,
        'total_processing_time': float,
        'average_time_per_video': float
    },
    'performance_metrics': dict,      # 성능 메트릭 상세
    'individual_results': list,       # 개별 비디오 결과
    'failed_videos': list,           # 실패한 비디오 목록
    'config': dict                   # 사용된 설정
}
```

### 2. RTMOPoseEstimator 클래스

RTMO 모델을 사용한 포즈 추정을 담당합니다.

#### 초기화 및 사용

```python
from pose_estimator import RTMOPoseEstimator

estimator = RTMOPoseEstimator(
    config_path="/path/to/rtmo_config.py",
    checkpoint_path="/path/to/rtmo_checkpoint.pth",
    device="cuda:0"
)

# 비디오에서 포즈 추정
pose_results = estimator.estimate_poses_from_video(
    video_path="/path/to/video.mp4",
    max_frames=900
)
```

**주요 메서드:**
- `estimate_poses_single_frame(frame)`: 단일 프레임 포즈 추정
- `estimate_poses_batch(frames)`: 배치 프레임 포즈 추정
- `estimate_poses_from_video(video_path)`: 비디오 파일 포즈 추정
- `get_valid_poses_count(pose_results)`: 유효한 포즈 수 반환

### 3. FightPrioritizedTracker 클래스

Fight-우선 트래킹 시스템으로 5영역 분할과 복합 점수를 활용합니다.

#### 5영역 분할 시스템

```python
from fight_tracker import FightPrioritizedTracker

tracker = FightPrioritizedTracker(
    frame_width=640,
    frame_height=480,
    region_weights={
        'center': 1.0,         # 중앙 영역 (최고 우선순위)
        'top_left': 0.7,       # 좌상단
        'top_right': 0.7,      # 우상단
        'bottom_left': 0.6,    # 좌하단
        'bottom_right': 0.6    # 우하단
    },
    composite_weights={
        'position': 0.3,       # 위치 점수 (30%)
        'movement': 0.25,      # 움직임 점수 (25%)
        'interaction': 0.25,   # 상호작용 점수 (25%)
        'detection': 0.1,      # 검출 신뢰도 (10%)
        'consistency': 0.1     # 시간적 일관성 (10%)
    }
)
```

**핵심 알고리즘:**

1. **위치 점수**: 인물의 중심점이 어느 영역에 위치하는지에 따른 가중치
2. **움직임 점수**: 연속 프레임 간 위치 변화량으로 동작의 격렬함 측정
3. **상호작용 점수**: 다른 인물과의 거리를 기반으로 상호작용 정도 계산
4. **검출 신뢰도**: 포즈 추정 키포인트의 평균 신뢰도
5. **시간적 일관성**: 최근 프레임들에서의 점수 일관성

**주요 메서드:**
- `calculate_composite_scores(keypoints_list, scores_list)`: 복합 점수 계산
- `get_fight_prioritized_order(keypoints_list, scores_list)`: Fight-우선 정렬
- `select_top_person(keypoints_list, scores_list)`: 최고 점수 인물 선택
- `process_video_sequence(pose_results, sequence_length)`: 비디오 시퀀스 처리

### 4. STGCNActionClassifier 클래스

STGCN++ 모델을 사용한 행동 분류를 담당합니다.

#### 사용법

```python
from action_classifier import STGCNActionClassifier

classifier = STGCNActionClassifier(
    config_path="/path/to/stgcn_config.py",
    checkpoint_path="/path/to/stgcn_checkpoint.pth",
    device="cuda:0"
)

# 키포인트 시퀀스 분류
result = classifier.classify_video_sequence(
    keypoints=keypoints_array,    # Shape: (T, 17, 2)
    scores=scores_array,          # Shape: (T, 17)
    window_size=30,
    stride=15,
    img_shape=(480, 640)
)
```

**윈도우 기반 분류:**
- 긴 비디오를 겹치는 윈도우로 분할하여 처리
- 각 윈도우별 예측 결과를 신뢰도 가중 투표로 통합
- 최종 예측의 신뢰도와 일관성 분석 제공

### 5. MetricsCalculator 클래스

종합적인 성능 평가 메트릭을 계산합니다.

#### 사용법

```python
from metrics_calculator import MetricsCalculator

calculator = MetricsCalculator()

# 종합 메트릭 계산
results = calculator.calculate_comprehensive_metrics(
    predictions=[1, 0, 1, 0, 1],
    ground_truths=[1, 0, 0, 0, 1],
    confidences=[0.9, 0.8, 0.7, 0.85, 0.95],
    video_names=['video1.mp4', 'video2.mp4', ...]
)
```

**제공 메트릭:**

1. **혼동 행렬 (Confusion Matrix)**
   - TP (True Positive): Fight를 Fight로 정확히 분류
   - TN (True Negative): NonFight를 NonFight로 정확히 분류
   - FP (False Positive): NonFight를 Fight로 잘못 분류
   - FN (False Negative): Fight를 NonFight로 잘못 분류

2. **분류 성능 지표**
   - **정확도 (Accuracy)**: (TP + TN) / (TP + TN + FP + FN)
   - **정밀도 (Precision)**: TP / (TP + FP)
   - **재현율 (Recall)**: TP / (TP + FN)
   - **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - **특이도 (Specificity)**: TN / (TN + FP)

3. **신뢰도 분석**
   - 평균 신뢰도, 표준편차, 최소/최대값
   - 올바른 예측 vs 틀린 예측의 신뢰도 비교

### 6. VideoOverlayGenerator 클래스

시각화 및 오버레이 비디오 생성을 담당합니다.

#### 사용법

```python
from video_overlay import VideoOverlayGenerator

overlay_gen = VideoOverlayGenerator(
    joint_color=(0, 255, 0),      # 관절 색상 (BGR)
    skeleton_color=(255, 0, 0),   # 스켈레톤 색상 (BGR)
    text_color=(255, 255, 255),   # 텍스트 색상 (BGR)
)

# 오버레이 비디오 생성
success = overlay_gen.create_overlay_video(
    video_path="/path/to/input.mp4",
    pose_results=pose_results,
    prediction_result=classification_result,
    output_path="/path/to/output_overlay.mp4"
)
```

**시각화 요소:**
- **관절 키포인트**: COCO 17-point 키포인트 원형 표시
- **스켈레톤 연결**: 키포인트 간 연결선
- **예측 결과**: Fight/NonFight 라벨과 신뢰도
- **프레임 정보**: 현재/전체 프레임 번호
- **정답 비교**: 실제 라벨과 예측 결과 동시 표시

---

## 사용법

### 1. 환경 설정

```bash
# 파이프라인 설정
cd /path/to/inference_pipeline
python setup_pipeline.py

# 빠른 테스트
python quick_test.py
```

### 2. 기본 실행

#### 단일 비디오 처리

```bash
python run_inference.py \
    --mode single \
    --input /path/to/video.mp4 \
    --annotations annotations.txt \
    --label-map label_map.txt \
    --output ./results \
    --generate-overlay
```

#### 배치 처리

```bash
python run_inference.py \
    --mode batch \
    --input /path/to/videos/ \
    --annotations annotations.txt \
    --label-map label_map.txt \
    --output ./results \
    --batch-size 8
```

#### 벤치마크 평가

```bash
python run_inference.py \
    --mode benchmark \
    --input /path/to/test_videos/ \
    --annotations annotations.txt \
    --label-map label_map.txt \
    --output ./results \
    --generate-overlay
```

### 3. Python API 직접 사용

```python
from main_pipeline import EndToEndPipeline

# 파이프라인 초기화
pipeline = EndToEndPipeline(
    pose_config="configs/rtmo_config.py",
    pose_checkpoint="checkpoints/rtmo_checkpoint.pth",
    gcn_config="configs/stgcn_config.py",
    gcn_checkpoint="checkpoints/stgcn_checkpoint.pth"
)

# 단일 비디오 처리
result = pipeline.process_single_video(
    video_path="test_video.mp4",
    ground_truth_label=1,
    generate_overlay=True
)

print(f"예측: {result['classification']['prediction_label']}")
print(f"신뢰도: {result['classification']['confidence']:.3f}")

# 리소스 정리
pipeline.cleanup()
```

---

## 설정 가이드

### 1. 기본 설정 (config.py)

#### 하드웨어 설정
```python
INFERENCE_CONFIG = {
    'device': 'cuda:0',        # 추론 디바이스
    'max_workers': 4,          # 병렬 작업 수
    'batch_size': 8,           # 배치 크기
}
```

#### 모델 설정
```python
INFERENCE_CONFIG = {
    'sequence_length': 30,          # STGCN++ 입력 시퀀스 길이
    'pose_score_threshold': 0.3,    # 포즈 키포인트 신뢰도 임계값
    'confidence_threshold': 0.5,    # 분류 신뢰도 임계값
}
```

#### Fight-우선 트래킹 설정
```python
INFERENCE_CONFIG = {
    # 5영역 가중치
    'region_weights': {
        'center': 1.0,         # 중앙 영역 (가장 중요)
        'top_left': 0.7,       # 좌상단
        'top_right': 0.7,      # 우상단
        'bottom_left': 0.6,    # 좌하단
        'bottom_right': 0.6    # 우하단
    },
    
    # 복합 점수 가중치
    'composite_weights': {
        'position': 0.3,       # 위치 점수 (30%)
        'movement': 0.25,      # 움직임 점수 (25%)
        'interaction': 0.25,   # 상호작용 점수 (25%)
        'detection': 0.1,      # 검출 신뢰도 (10%)
        'consistency': 0.1     # 시간적 일관성 (10%)
    }
}
```

### 2. 입력 파일 형식

#### annotations.txt
```
video1.mp4,1
video2.mp4,0
video3.mp4,1
fight_scene_01.mp4,1
normal_scene_01.mp4,0
```

#### label_map.txt
```
Fight: 1
NonFight: 0
```

### 3. 출력 구조

```
results/
├── batch_results.json           # 전체 배치 결과
├── detailed_metrics.json        # 상세 성능 메트릭
├── performance_report.md        # 성능 보고서
├── individual_results/          # 개별 비디오 결과
│   ├── video1_result.json
│   └── video2_result.json
└── overlays/                    # 오버레이 비디오
    ├── video1_overlay.mp4
    └── video2_overlay.mp4
```

---

## 성능 메트릭

### 1. 분류 성능 지표

#### 혼동 행렬
|          | 예측 NonFight | 예측 Fight |
|----------|---------------|------------|
| 실제 NonFight | TN           | FP         |
| 실제 Fight | FN           | TP         |

#### 핵심 메트릭
- **정확도**: 전체 예측 중 올바른 예측의 비율
- **정밀도**: Fight 예측 중 실제 Fight인 비율 (False Alarm 최소화)
- **재현율**: 실제 Fight 중 올바르게 예측한 비율 (누락 최소화)
- **F1-Score**: 정밀도와 재현율의 조화평균

### 2. 신뢰도 분석

#### 예측 신뢰도 분포
- 올바른 예측의 평균 신뢰도
- 틀린 예측의 평균 신뢰도
- 신뢰도 표준편차 (예측 안정성)

#### 예측 일관성
- 윈도우 기반 예측 간 일관성
- 시간적 안정성 측정

### 3. 클래스별 성능

#### Fight 클래스
- 실제 Fight 비디오 수
- Fight로 예측된 비디오 수
- 올바르게 분류된 Fight 비디오 수
- Fight 클래스 정확도

#### NonFight 클래스
- 실제 NonFight 비디오 수
- NonFight로 예측된 비디오 수
- 올바르게 분류된 NonFight 비디오 수
- NonFight 클래스 정확도

---

## 문제 해결

### 1. 일반적인 오류

#### GPU 메모리 부족
```bash
# CPU 모드로 실행
python run_inference.py --device cpu --batch-size 2

# 배치 크기 줄이기
python run_inference.py --batch-size 4
```

#### 모델 파일 누락
```bash
# 설정 검증
python setup_pipeline.py
python quick_test.py

# 모델 경로 확인
ls -la /path/to/checkpoints/
```

#### 의존성 문제
```bash
# 필수 패키지 설치
pip install torch torchvision
pip install mmpose mmaction2 mmengine mmcv
pip install opencv-python numpy
```

### 2. 성능 최적화

#### 처리 속도 향상
- GPU 메모리 사용량 모니터링
- 배치 크기 조정
- 병렬 처리 활용

#### 정확도 향상
- 포즈 신뢰도 임계값 조정
- Fight-우선 트래킹 가중치 튜닝
- 윈도우 크기 및 간격 최적화

### 3. 로그 및 디버깅

#### 상세 로그 활성화
```bash
python run_inference.py --verbose
```

#### 로그 파일 확인
```bash
tail -f inference.log
```

#### 설정 검증
```bash
python run_inference.py --dry-run
```

---

## 고급 사용법

### 1. 커스텀 설정

#### Fight-우선 트래킹 파라미터 조정
```python
# 더 강한 중앙 집중
region_weights = {
    'center': 1.5,
    'top_left': 0.5,
    'top_right': 0.5,
    'bottom_left': 0.4,
    'bottom_right': 0.4
}

# 움직임 중심 분석
composite_weights = {
    'position': 0.2,
    'movement': 0.4,
    'interaction': 0.2,
    'detection': 0.1,
    'consistency': 0.1
}
```

#### 윈도우 기반 분류 파라미터
```python
result = classifier.classify_video_sequence(
    keypoints=keypoints,
    scores=scores,
    window_size=45,      # 더 긴 시퀀스
    stride=10,           # 더 촘촘한 윈도우
    img_shape=(480, 640)
)
```

### 2. 배치 처리 최적화

#### 대용량 데이터 처리
```python
# 스트리밍 방식 처리
for batch in video_batches:
    batch_result = pipeline.process_batch_videos(
        batch, 
        save_individual_results=False,  # 메모리 절약
        generate_overlay=False          # 처리 속도 향상
    )
    # 중간 결과 저장
    save_intermediate_results(batch_result)
```

### 3. 실시간 처리

#### 웹캠 입력 처리
```python
import cv2

cap = cv2.VideoCapture(0)
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_buffer.append(frame)
    
    if len(frame_buffer) >= 30:  # 30프레임마다 처리
        # 포즈 추정
        pose_results = estimator.estimate_poses_batch(frame_buffer)
        
        # Fight-우선 트래킹
        selected_kpts, selected_scores = tracker.process_video_sequence(pose_results)
        
        # 분류
        result = classifier.classify_sequence(selected_kpts, selected_scores)
        
        print(f"예측: {result['prediction_label']} (신뢰도: {result['confidence']:.3f})")
        
        frame_buffer = []  # 버퍼 초기화

cap.release()
```

---

이 API 가이드는 STGCN++ Violence Detection 시스템의 모든 구성 요소와 사용법을 다룹니다. 추가적인 질문이나 특정 사용 사례에 대한 문의는 개발팀에 연락주시기 바랍니다.