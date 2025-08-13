# RTMO Pose Track - 구조화된 폭력 감지 파이프라인

## 개요

이 프로젝트는 RTMO (Real-Time Multi-Object) 포즈 추출과 GCN (Graph Convolutional Network) 기반 행동 분류를 결합한 폭력 감지 시스템입니다. 코드를 기능별로 구조화하여 가독성과 유지보수성을 향상시켰습니다.

## 새로운 프로젝트 구조

```
rtmo_pose_track/
├── core/                          # 핵심 기능 모듈
│   ├── __init__.py
│   ├── model_manager.py          # 모델 초기화/관리
│   └── tracker.py                # ByteTracker 구현
├── processing/                    # 처리 파이프라인
│   ├── __init__.py
│   └── base_processor.py         # 기본 처리기 클래스
├── utils/                         # 유틸리티
│   ├── __init__.py
│   ├── file_utils.py             # 파일 처리 유틸
│   ├── video_utils.py            # 비디오 관련 유틸
│   ├── data_utils.py             # 데이터 변환 유틸
│   └── annotation_utils.py       # 어노테이션 처리
├── visualization/                 # 시각화
│   ├── __init__.py
│   └── visualizer.py             # 메인 시각화 클래스
├── logging/                       # 로깅 시스템
│   ├── __init__.py
│   └── error_logger.py           # 에러 로깅
├── scripts/                       # ⭐ 실행 스크립트 (메인 사용)
│   ├── __init__.py
│   ├── separated_pose_pipeline.py  # 🎯 분리 처리 파이프라인
│   ├── inference_pipeline.py       # 🎯 추론 파이프라인
│   ├── run_pose_extraction.py      # 통합 실행
│   └── run_visualization.py        # 시각화 실행
├── configs/                       # 설정 파일 (기존 유지)
├── output/                        # 출력 폴더 (기존 유지)
├── test_data/                     # 테스트 데이터
└── _legacy_backup/               # 🗂️ 원본 파일 백업
    ├── enhanced_rtmo_bytetrack_pose_extraction.py
    ├── unified_pose_processor.py
    └── [기타 원본 파일들]
```

## 주요 개선사항

### 1. 코드 분할 및 모듈화
- **core/** : 포즈 추출, 트래킹, 윈도우 처리 등 핵심 기능
- **processing/** : 다양한 처리 파이프라인 구현
- **utils/** : 공통 유틸리티 함수들
- **visualization/** : 시각화 관련 기능
- **logging/** : 에러 로깅 시스템

### 2. 중복 코드 제거
- 모델 초기화 로직 통합
- 비디오 파일 처리 로직 통합
- ByteTracker 설정 로직 통합
- 공통 유틸리티 함수 분리

### 3. 명확한 의존성 관리
- 각 모듈별 `__init__.py` 파일로 public API 정의
- 상대 임포트 사용으로 모듈간 의존성 명확화

## 사용법

### 1. 분리된 포즈 추정 파이프라인 (주요 사용)
```bash
# 전체 파이프라인 실행 (3단계)
python scripts/separated_pose_pipeline.py

# 특정 단계만 실행
python scripts/separated_pose_pipeline.py --stage 1  # 포즈 추정만
python scripts/separated_pose_pipeline.py --stage 2  # 트래킹만
python scripts/separated_pose_pipeline.py --stage 3  # 통합만

# 사용자 정의 설정 파일 사용
python scripts/separated_pose_pipeline.py --config configs/custom_config.py

# Resume 기능 (이미 처리된 파일은 건너뜀)
python scripts/separated_pose_pipeline.py --resume
```

### 2. 추론 파이프라인 (주요 사용)
```bash
# 기본 추론 실행 (resume 모드)
python scripts/inference_pipeline.py --config configs/inference_config.py

# 모든 비디오 강제 재처리
python scripts/inference_pipeline.py --config configs/inference_config.py --force

# Config 오버라이드와 함께 사용
python scripts/inference_pipeline.py --config configs/inference_config.py gpu=1 debug_mode=True
```

### 3. 통합 처리 파이프라인 (선택적 사용)
```bash
# 기본 설정으로 실행
python scripts/run_pose_extraction.py

# 사용자 정의 설정 파일 사용
python scripts/run_pose_extraction.py configs/custom_config.py

# 설정 오버라이드
python scripts/run_pose_extraction.py mode=full gpu=0 max_workers=1
```

### 2. 시각화 실행
```bash
# 기본 설정으로 시각화
python scripts/run_visualization.py

# 사용자 정의 경로 지정
python scripts/run_visualization.py /path/to/input /path/to/output
```

### 3. 모듈별 사용 예제

#### 핵심 모듈 사용
```python
from core import ModelManager, PoseExtractor, ByteTracker
from processing import UnifiedProcessor
from utils import collect_video_files, get_video_info

# 모델 관리자 초기화
model_manager = ModelManager(device='cuda:0')
pose_model = model_manager.initialize_pose_model(config_file, checkpoint_file)

# 통합 처리기 사용
processor = UnifiedProcessor(
    detector_config=config_file,
    detector_checkpoint=checkpoint_file,
    device='cuda:0'
)
```

#### 유틸리티 사용
```python
from utils import collect_video_files, get_video_info, save_pkl_data

# 비디오 파일 수집
video_files = collect_video_files('/path/to/videos')

# 비디오 정보 추출
info = get_video_info('/path/to/video.mp4')

# 데이터 저장
save_pkl_data(data, '/path/to/output.pkl')
```

## 마이그레이션 가이드

### 기존 코드에서 새 구조로 전환

**이전:**
```python
from enhanced_rtmo_bytetrack_pose_extraction import ByteTracker
from unified_pose_processor import UnifiedPoseProcessor
from error_logger import ProcessingErrorLogger
```

**이후:**
```python
from core import ByteTracker
from processing import UnifiedProcessor  
from logging import ProcessingErrorLogger
```

### 설정 파일 업데이트

기존 설정 파일들은 `configs/` 폴더에 그대로 유지되며, 새로운 구조에서도 동일하게 사용 가능합니다.

## 개발 가이드

### 새로운 기능 추가
1. 해당 기능에 맞는 모듈 폴더에 파일 생성
2. `__init__.py` 파일에 public API 추가
3. 적절한 테스트 코드 작성

### 의존성 관리
- 모듈간 순환 의존성 방지
- 상대 임포트 사용 권장
- 외부 라이브러리 의존성 최소화

## 테스트

새로운 구조에서 기존 기능들이 정상 동작하는지 확인:

```bash
# 통합 처리 테스트
python scripts/run_unified.py configs/test_config.py

# 시각화 테스트  
python scripts/run_visualization.py test_data output
```

## 기여 가이드

1. 새로운 기능은 적절한 모듈에 배치
2. 공통 기능은 `utils/`에 추가
3. 각 모듈의 `__init__.py` 업데이트
4. 문서화 및 테스트 코드 작성

## 호환성

- 기존 설정 파일과 100% 호환
- 기존 출력 형식과 동일
- 기존 CLI 인터페이스 유지