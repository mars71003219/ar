# 모듈화된 폭력 탐지 시스템

4단계 파이프라인을 통한 효율적이고 확장 가능한 폭력 탐지 시스템입니다.

## 🚀 빠른 시작

```bash
# 기본 추론
python recognizer/main.py --mode inference --input video.mp4

# PKL 생성 + 시각화 (성능 평가 포함)
python recognizer/main.py --mode inference --input video.mp4 --enable_evaluation --enable_visualization

# 프리셋 사용
python recognizer/main.py --config configs/presets/inference_with_evaluation.yaml --input video.mp4
```

**자세한 사용법은 [USAGE.md](USAGE.md)를 참고하세요.**

## 🏗️ 시스템 아키텍처

### 4단계 파이프라인
1. **포즈 추정 (Pose Estimation)** - RTMO 모델을 통한 실시간 포즈 검출
2. **객체 추적 (Object Tracking)** - ByteTracker를 통한 다중 객체 추적
3. **복합 점수 계산 (Composite Scoring)** - 5영역 기반 복합점수 계산
4. **행동 분류 (Action Classification)** - ST-GCN++를 통한 폭력/비폭력 분류

### 주요 특징
- ✅ **모듈화된 설계**: 각 단계별 독립적인 모듈로 구성
- ✅ **팩토리 패턴**: 쉬운 모델 교체 및 확장
- ✅ **표준화된 API**: 일관된 인터페이스 제공
- ✅ **실시간 처리**: RTSP 스트림 및 웹캠 지원
- ✅ **배치 처리**: 대용량 비디오 처리 최적화
- ✅ **시각화 도구**: 결과 분석 및 검증 도구 제공

## 📦 설치 및 환경 설정

### 필수 요구사항
- Python 3.8+
- PyTorch 1.11+
- CUDA 11.0+ (GPU 사용 시)
- MMPose, MMAction2

### 의존성 설치
```bash
# 기본 의존성
pip install torch torchvision opencv-python numpy matplotlib seaborn pyyaml

# MMPose 설치 (포즈 추정)
cd mmpose
pip install -e .

# MMAction2 설치 (행동 분류)
cd mmaction2  
pip install -e .
```

## 🚀 빠른 시작

### 1. 기본 사용법

```python
from recognizer import factory
from recognizer.utils.data_structure import *
from recognizer.pipelines.unified_pipeline import *

# 설정 생성
pose_config = PoseEstimationConfig(
    model_name='rtmo',
    config_file='path/to/rtmo_config.py',
    model_path='path/to/rtmo_model.pth'
)

# 파이프라인 설정
pipeline_config = PipelineConfig(
    pose_config=pose_config,
    tracking_config=TrackingConfig(tracker_name='bytetrack'),
    scoring_config=ScoringConfig(scorer_name='region_based'),
    classification_config=ActionClassificationConfig(model_name='stgcn')
)

# 통합 파이프라인 실행
with UnifiedPipeline(pipeline_config) as pipeline:
    result = pipeline.process_video('input_video.mp4')
    print(f"처리 완료: {result.avg_fps:.1f} FPS")
```

### 2. 설정 파일 사용

```python
import yaml
from recognizer.examples.config_usage import *

# YAML 설정 로드
config = yaml.safe_load(open('configs/default_config.yaml'))
pose_config, tracking_config, scoring_config, classification_config = create_configs_from_yaml(config)

# 파이프라인 실행
pipeline_config = PipelineConfig(pose_config, tracking_config, scoring_config, classification_config)
```

### 3. 실시간 처리

```python
from recognizer.pipelines.inference_pipeline import *

# 실시간 설정
realtime_config = RealtimeConfig(
    pose_config=pose_config,
    # ... 기타 설정
    target_fps=30.0,
    alert_threshold=0.7
)

# 실시간 파이프라인
with InferencePipeline(realtime_config) as pipeline:
    # 알림 콜백 등록
    pipeline.add_alert_callback(lambda alert: print(f"[알림] {alert.alert_type}"))
    
    # 웹캠에서 실시간 처리
    pipeline.start_realtime_processing(source=0)
```

## 📁 프로젝트 구조

```
recognizer/
├── __init__.py                    # 메인 모듈 및 팩토리 초기화
├── utils/
│   ├── data_structure.py         # 표준 데이터 구조
│   └── factory.py                # 모듈 팩토리 패턴
├── pose_estimation/              # 포즈 추정 모듈
│   ├── base.py                   # 기본 추상 클래스
│   └── rtmo/                     # RTMO 구현
├── tracking/                     # 객체 추적 모듈  
│   ├── base.py                   # 기본 추상 클래스
│   └── bytetrack/                # ByteTracker 구현
├── scoring/                      # 복합점수 계산 모듈
│   ├── base.py                   # 기본 추상 클래스
│   └── region_based/             # 영역 기반 점수 계산
├── action_classification/        # 행동 분류 모듈
│   ├── base.py                   # 기본 추상 클래스
│   └── stgcn/                    # ST-GCN++ 구현
├── pipelines/                    # 통합 파이프라인
│   ├── unified_pipeline.py       # 전체 4단계 파이프라인
│   ├── annotation_pipeline.py    # 어노테이션 구축용
│   └── inference_pipeline.py     # 실시간 추론용
├── visualization/                # 시각화 도구
│   ├── pose_visualizer.py        # 포즈 시각화
│   ├── result_visualizer.py      # 결과 분석 시각화
│   └── annotation_visualizer.py  # 어노테이션 도구
├── examples/                     # 사용 예제
│   ├── basic_usage.py            # 기본 사용법
│   └── config_usage.py           # 설정 파일 사용법
└── configs/                      # 설정 파일
    └── default_config.yaml       # 기본 설정
```

## 📊 모듈 상세

### 포즈 추정 (Pose Estimation)
- **RTMO**: 실시간 다중 객체 포즈 검출
- **지원 형식**: 이미지, 비디오, 실시간 스트림
- **출력**: 17개 키포인트 + 바운딩 박스

### 객체 추적 (Tracking)  
- **ByteTracker**: 고성능 다중 객체 추적
- **특징**: Kalman 필터 기반, ID 유지
- **출력**: 트랙 ID가 할당된 포즈 시퀀스

### 복합점수 계산 (Scoring)
- **5영역 분할**: 화면을 5개 영역으로 나누어 분석
- **5가지 점수**: 움직임, 위치, 상호작용, 시간일관성, 지속성
- **가중 합산**: 설정 가능한 가중치로 최종 점수 계산

### 행동 분류 (Classification)
- **ST-GCN++**: 스켈레톤 기반 행동 인식
- **슬라이딩 윈도우**: 100 프레임 단위 분석
- **클래스**: Fight/NonFight (확장 가능)

## 🔧 확장 방법

### 새로운 포즈 추정 모델 추가

```python
from recognizer.pose_estimation.base import BasePoseEstimator

class NewPoseEstimator(BasePoseEstimator):
    def initialize_model(self):
        # 모델 초기화 로직
        pass
    
    def extract_poses(self, image, frame_idx):
        # 포즈 추정 로직
        pass

# 팩토리에 등록
factory.register_pose_estimator('new_model', NewPoseEstimator)
```

### 새로운 트래커 추가

```python
# MMTracking 기반 트래커 사용 (권장)
from recognizer.tracking.mmtracking_adapter import MMTrackingAdapter

# 기본 제공 트래커들: 'bytetrack', 'deepsort', 'sort'
tracker_config = TrackingConfig(
    tracker_name='bytetrack',
    device='cuda:0'
)

tracker = MMTrackingAdapter(tracker_config)
```

## 📈 성능 최적화

### GPU 가속
```python
# GPU 설정
config.device = 'cuda:0'
config.enable_gpu = True
config.batch_size = 4  # GPU 메모리에 따라 조정
```

### 실시간 최적화
```python
# 실시간 성능 향상
realtime_config.skip_frames = 2  # 프레임 건너뛰기
realtime_config.resize_input = (480, 480)  # 입력 크기 축소
realtime_config.inference_stride = 50  # 추론 간격 증가
```

### 메모리 최적화
```python
# 메모리 사용량 감소
config.save_intermediate_results = False
config.max_queue_size = 50
```

## 📊 결과 분석

### 시각화 도구
```python
from recognizer.visualization import *

# 포즈 시각화
pose_viz = PoseVisualizer()
pose_viz.visualize_video_poses('input.mp4', poses, 'output_poses.mp4')

# 결과 분석
result_viz = ResultVisualizer()
result_viz.visualize_classification_results(results, 'analysis.png')
result_viz.create_timeline_visualization(results, 'timeline.png')
```

### 성능 메트릭
- **처리 속도**: FPS, 단계별 처리 시간
- **정확도**: 분류 정확도, 신뢰도 분포
- **자원 사용량**: GPU/CPU 사용률, 메모리 사용량

## 🐛 문제 해결

### 일반적인 문제
1. **CUDA 메모리 부족**: 배치 크기 감소, 입력 크기 축소
2. **모델 로드 실패**: 경로 확인, 의존성 설치 확인
3. **낮은 FPS**: GPU 사용, 프레임 건너뛰기 활성화

### 디버그 모드
```python
config.debug.verbose = True
config.debug.save_intermediate = True  # 중간 결과 저장
config.debug.profile_performance = True  # 성능 프로파일링
```

## 📝 라이선스

이 프로젝트는 기존 rtmo_gcn_pipeline의 코드를 새로운 모듈화 구조에 맞게 재구성한 것입니다.

## 🤝 기여 방법

1. 새로운 모델 구현 시 해당 모듈의 base 클래스 상속
2. 표준 데이터 구조 사용
3. 팩토리 패턴을 통한 등록
4. 단위 테스트 작성

## 📞 지원

문제가 있거나 기능 요청이 있으시면 이슈를 생성해주세요.