# Recognizer 설정 파일 구조

## 개요

설정 파일들은 계층 구조로 구성되어 중복을 최소화하고 유지보수성을 향상시켰습니다.

## 파일 구조

```
configs/
├── base_config.yaml           # 베이스 설정 (모든 파이프라인 공통)
├── main_config.yaml          # 메인 실행기 설정 
├── inference_config.yaml     # 실시간 추론 파이프라인 설정
├── separated_pipeline_config.yaml  # 분리형 파이프라인 설정
├── unified_pipeline_config.yaml    # 통합 파이프라인 설정
└── README.md                 # 이 파일
```

## 설정 파일 상속 구조

```
base_config.yaml (공통 설정)
    ├── main_config.yaml (메인 실행기)
    ├── inference_config.yaml (실시간 추론)
    ├── separated_pipeline_config.yaml (분리형)
    └── unified_pipeline_config.yaml (통합)
```

## 각 설정 파일의 역할

### 1. base_config.yaml
- **역할**: 모든 파이프라인에서 공통으로 사용되는 설정
- **포함 내용**:
  - 모델 설정 (RTMO, ByteTrack, STGCN)
  - 기본 성능 설정 (디바이스, 배치 크기, 윈도우 설정)
  - 로깅 및 출력 설정
  - 메모리 관리 및 오류 처리

### 2. main_config.yaml
- **역할**: main.py 실행기의 기본 설정
- **특징**: base_config.yaml 상속하여 필요한 부분만 오버라이드
- **포함 내용**:
  - 실행 모드 설정
  - 기능 활성화 설정 (평가, 시각화)
  - 모드별 특화 설정

### 3. inference_config.yaml
- **역할**: 실시간 추론 파이프라인 특화 설정
- **특징**: 실시간 처리에 최적화된 설정
- **포함 내용**:
  - 큐 설정 및 실시간 처리 옵션
  - 알림 및 콜백 설정
  - 실시간 로깅 설정

### 4. separated_pipeline_config.yaml
- **역할**: 분리형 파이프라인 특화 설정
- **특징**: 4단계 분리 처리에 최적화
- **포함 내용**:
  - 스테이지별 출력 디렉토리
  - Resume 기능 설정
  - 멀티프로세스 설정

### 5. unified_pipeline_config.yaml
- **역할**: 통합 파이프라인 특화 설정
- **특징**: 단일 비디오 및 배치 처리에 최적화
- **포함 내용**:
  - 중간 결과 저장 설정
  - 진행률 콜백 설정
  - 배치 처리 최적화

## 사용법

### 1. 기본 사용 (main.py)
```bash
# 기본 설정으로 실행
python main.py --mode inference --input video.mp4

# 특정 설정 파일 사용
python main.py --config configs/inference_config.yaml --mode inference --input video.mp4
```

### 2. 파이프라인별 직접 사용
```python
# 실시간 추론
from pipelines import InferencePipeline
from pipelines.inference.config import RealtimeConfig

# 분리형 파이프라인  
from pipelines import SeparatedPipeline
from pipelines.separated.config import SeparatedPipelineConfig

# 통합 파이프라인
from pipelines import UnifiedPipeline
from pipelines.unified.config import PipelineConfig
```

## 설정 우선순위

1. **명령행 인자** (최고 우선순위)
2. **특화 설정 파일** (inference_config.yaml 등)
3. **베이스 설정 파일** (base_config.yaml)
4. **코드 내 기본값** (최저 우선순위)

## 설정 확장

새로운 파이프라인이나 기능을 추가할 때:

1. **공통 설정**은 `base_config.yaml`에 추가
2. **특화 설정**은 새로운 설정 파일을 생성하여 base_config 상속
3. **임시 설정**은 명령행 인자로 오버라이드

## 주요 개선사항

- **중복 제거**: 공통 설정을 base_config.yaml로 통합
- **모듈화**: 파이프라인별 특화 설정 분리
- **유지보수성**: 상속 구조로 변경 영향 최소화
- **확장성**: 새로운 파이프라인 설정 쉽게 추가 가능