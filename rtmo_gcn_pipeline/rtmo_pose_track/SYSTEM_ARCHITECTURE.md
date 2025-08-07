# Inference Pipeline 시스템 구성도

이 문서는 `inference_pipeline.py` 코드의 로직을 분석하여 시스템의 전체 구성과 데이터 흐름을 시각적으로 표현하고, 각 구성 요소의 역할을 상세히 설명합니다.

## 시스템 구성도 (Mermaid)

```mermaid
graph TD
    subgraph "입력 및 설정"
        A[Video Files]
        B[/inference_config.py/]
        C[CLI Arguments]
    end

    subgraph "메인 컨트롤러"
        D{파이프라인 시작}
        E{처리 모드 결정}
        F[처리할 비디오 목록 생성]
        G{GPU 처리 방식 결정}
    end

    subgraph "단일 비디오 처리 루프 (병렬 처리 단위)"
        H[/Single Video/] --> I[1. 포즈 추정] --> J[/프레임별 포즈 데이터/]
        J --> K[2. 객체 추적 및 랭킹] --> L[/추적된 인물별 시계열 포즈/]
        L --> M[3. 슬라이딩 윈도우 생성] --> N[/윈도우별 포즈 데이터/]
        N --> O[4. 행동 인식] --> P[윈도우 예측 결과]
    end

    subgraph "결과 종합 및 평가"
        Q[5. 비디오별 최종 판정] --> R[모든 비디오 결과 종합]
        R --> S[6. 성능 평가]
        R --> T[/결과 파일/]
        S --> U[콘솔 출력]
    end

    %% 데이터 흐름 연결
    C & B --> D --> E --> F --> G

    %% GPU 처리 분기
    G -- Single-GPU --> H
    G -- Multi-GPU --> G_multi(분산 처리)

    subgraph "Multi-GPU 처리"
        direction LR
        G_multi --> P1[GPU 1: 처리 루프 실행] --> P1_out(Result 1)
        G_multi --> P2[GPU 2: 처리 루프 실행] --> P2_out(Result 2)
        G_multi --> Pn[GPU N: 처리 루프 실행] --> Pn_out(Result N)
    end
  
    %% 결과 취합
    P --> Q  %% Single-GPU 경로
    P1_out & P2_out & Pn_out --> Q %% Multi-GPU 경로
  
    Q --> R
```

---

## 구성도 설명

이 구성도는 `inference_pipeline.py`의 전체 워크플로우를 나타냅니다.

### 1. 입력 및 설정 (Input & Configuration)

- **Video Files**: 분석할 원본 비디오 파일들입니다.
- **inference_config.py**: 파이프라인의 모든 동작을 제어하는 핵심 설정 파일입니다. 모델 경로, 임계값, GPU 설정 등이 정의되어 있습니다.
- **CLI Arguments**: 사용자가 파이프라인 실행 시 전달하는 인자들입니다. `--force` (강제 재처리)나 `--resume` (이어하기) 같은 실행 모드를 제어합니다.

### 2. 메인 컨트롤러 (Main Controller)

- `FightInferenceProcessor` 클래스가 이 역할을 수행합니다.
- 사용자 입력과 설정 파일을 바탕으로 **처리 모드**를 결정하고, 입력 디렉토리에서 비디오를 읽어와 **처리할 비디오 목록**을 만듭니다.
- 설정된 GPU 수에 따라 **단일/다중 GPU 처리 방식**을 결정합니다. 다중 GPU일 경우, 비디오 목록을 각 GPU에 분배하여 병렬 처리를 준비합니다.

### 3. 단일 비디오 처리 루프 (Single Video Processing Loop)

- 이 블록은 각 비디오 파일에 대해 수행되는 핵심 분석 과정이며, 다중 GPU 사용 시 여러 프로세스에서 병렬로 실행됩니다.
- **(1) 포즈 추정 (Pose Estimation)**: `EnhancedRTMOPoseExtractor`가 RTMO 모델을 사용하여 비디오의 모든 프레임에서 인물들의 2D 관절 좌표를 추출합니다.
- **(2) 객체 추적 및 랭킹 (Tracking & Ranking)**: ByteTrack 알고리즘으로 프레임 간 동일 인물을 추적하여 고유 ID를 부여합니다. 이후 움직임, 위치, 상호작용 등을 기반으로 계산된 **복합 점수(Composite Score)**를 통해 각 인물의 중요도(Rank)를 매깁니다.
- **(3) 슬라이딩 윈도우 생성 (Sliding Window Generation)**: 추적된 시계열 포즈 데이터를 `clip_len`과 `inference_stride` 설정에 따라 일정한 길이의 클립(윈도우)으로 분할합니다.
- **(4) 행동 인식 (Action Recognition)**: 분할된 각 윈도우 데이터를 ST-GCN++ 모델에 입력하여 해당 윈도우가 'Fight'일 확률(Score)을 예측합니다.

### 4. 결과 종합 및 평가 (Aggregation & Evaluation)

- **(5) 비디오별 최종 판정 (Final Video Prediction)**: 한 비디오의 모든 윈도우 예측 결과를 종합합니다. `consecutive_event_threshold` 설정에 따라 연속적인 'Fight' 윈도우가 임계값 이상일 경우, 해당 비디오를 최종 'Fight'로 판정합니다.
- 모든 비디오에 대한 처리가 끝나면, 개별 비디오의 판정 결과를 모두 모아 **종합**합니다.
- **(6) 성능 평가 (Performance Evaluation)**: 종합된 결과와 실제 레이블을 비교하여 정확도, 정밀도, 재현율, F1-Score 등 최종 성능 지표를 계산합니다.
- 모든 중간/최종 결과는 **JSON**과 **PKL** 파일로 저장되며, 최종 성능 보고서는 **콘솔에 출력**됩니다.
