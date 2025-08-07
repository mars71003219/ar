# RTMO-GCN 파이프라인 문서

이 문서는 `inference_pipeline.py`와 `inference_config.py`를 기반으로 한 폭력 탐지 추론 파이프라인의 작동 방식, 설정, 사용법을 설명합니다.

## 1. 개요

본 파이프라인은 비디오 파일로부터 폭력 행위를 탐지하기 위한 엔드투엔드(End-to-End) 추론 시스템입니다. 설정 파일을 통해 파라미터를 유연하게 제어할 수 있으며, 다음과 같은 단계로 구성됩니다.

1.  **포즈 추정 (Pose Estimation)**: `MMPose`의 RTMO 모델을 사용하여 비디오의 각 프레임에서 인물들의 2D 포즈(신체 키포인트)를 탐지합니다.
2.  **객체 추적 (Tracking)**: ByteTrack 알고리즘을 기반으로 탐지된 인물들에게 고유 ID를 부여하고 프레임 간 추적을 수행합니다.
3.  **액션 인식 (Action Recognition)**: `MMAction2`의 ST-GCN++ 모델을 사용하여 추적된 인물들의 시계열 포즈 데이터를 분석하고, 특정 시간 윈도우 내의 행동을 'Fight' 또는 'NonFight'로 분류합니다.
4.  **연속 이벤트 처리 (Consecutive Event Processing)**: 개별 윈도우의 예측 결과를 종합하여, 설정된 임계값 이상의 연속적인 'Fight' 이벤트가 발생했는지 판단하여 비디오 전체의 최종 결과를 결정합니다.
5.  **성능 평가 (Performance Evaluation)**: 모든 비디오 처리가 완료된 후, 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score 등의 성능 지표를 계산하고 결과를 저장합니다.

## 2. 주요 기능

-   **설정 기반 실행**: 모든 주요 파라미터를 `inference_config.py` 파일에서 관리하여 코드 수정 없이 실험 환경을 변경할 수 있습니다.
-   **이어하기 (Resume Mode)**: 이미 처리된 비디오를 건너뛰고 나머지 비디오부터 파이프라인을 재개하는 기능을 지원하여 시간과 자원을 절약합니다.
-   **강제 재처리 (Force Mode)**: 기존 처리 결과와 상관없이 모든 비디오를 강제로 다시 처리합니다.
-   **다중 GPU 지원**: 여러 개의 GPU를 병렬로 사용하여 대량의 비디오를 빠르게 처리할 수 있습니다.
-   **상세 결과 저장**: 윈도우별, 비디오별 예측 결과와 최종 성능 지표를 JSON 형식으로 저장하여 분석이 용이합니다.
-   **오류 로깅**: 처리 과정에서 발생하는 오류를 기록하여 디버깅을 돕습니다.

## 3. 실행 방법

터미널에서 `rtmo_gcn_pipeline/rtmo_pose_track/_legacy_backup` 디렉토리로 이동한 후 다음 명령어를 사용하여 파이프라인을 실행합니다.

### 기본 실행 (이어하기 모드)

```bash
python inference_pipeline.py --config configs/inference_config.py
```

### 모든 비디오 강제 재처리

```bash
python inference_pipeline.py --config configs/inference_config.py --force
```

### 이어하기 비활성화 (모든 비디오 처리)

```bash
python inference_pipeline.py --config configs/inference_config.py --no-resume
```

### 설정 오버라이드와 함께 실행

커맨드 라인에서 특정 설정 값을 즉시 변경하여 실행할 수 있습니다.

```bash
# GPU 1번 사용 및 디버그 모드 활성화
python inference_pipeline.py --config configs/inference_config.py gpu=1 debug_mode=True

# GPU 0, 1번 사용 및 분류 임계값 변경
python inference_pipeline.py --force gpu=0,1 classification_threshold=0.7
```

## 4. 상세 워크플로우

1.  **초기화**:
    -   `FightInferenceProcessor` 클래스가 `inference_config.py` 파일의 설정을 로드합니다.
    -   설정된 GPU 정보에 따라 PyTorch 디바이스를 설정합니다.
    -   `EnhancedRTMOPoseExtractor` (포즈 추정 및 추적기)와 `MMAction2 Recognizer` (행동 분류기) 모델을 초기화하고 메모리에 로드합니다.

2.  **비디오 파일 수집**:
    -   `input_dir`에 지정된 경로에서 `.mp4`, `.avi` 등의 비디오 파일을 재귀적으로 탐색하여 처리할 목록을 생성합니다.

3.  **처리 모드 결정**:
    -   `--force` 옵션이 있으면 모든 비디오를 처리 대상으로 설정합니다.
    -   `--resume` 모드(기본값)에서는 `output_dir/windows` 디렉토리에 저장된 중간 결과(.pkl)를 확인하여 이미 처리된 비디오를 제외한 나머지 비디오만 처리 대상으로 설정합니다.

4.  **추론 실행**:
    -   **단일 GPU**: 설정된 GPU 하나를 사용하여 비디오 목록을 순차적으로 처리합니다.
    -   **다중 GPU**: `ProcessPoolExecutor`를 사용하여 각 GPU에 비디오 목록을 분배하고 병렬로 처리합니다. 각 프로세스는 독립적으로 모델을 로드하고 할당된 비디오를 처리합니다.
    -   **비디오 처리 단위 (`_process_video`)**:
        1.  `extract_poses_only`: 비디오 전체에서 프레임별 포즈 데이터를 추출합니다.
        2.  `_augment_frames`: 포즈 데이터의 전체 길이가 `clip_len`보다 짧을 경우, 데이터를 복제하여 길이를 맞춥니다.
        3.  **슬라이딩 윈도우**: `clip_len`과 `inference_stride` 설정에 따라 시계열 포즈 데이터를 여러 개의 윈도우로 분할합니다.
        4.  **윈도우 처리 단위**:
            -   `apply_tracking_to_poses`: 윈도우 내의 포즈 데이터에 추적 알고리즘을 적용하여 각 인물에게 ID를 할당하고, 랭킹(중요도)을 매깁니다.
            -   `_predict_window_static`: 추적된 포즈 데이터를 ST-GCN++ 모델의 입력 형식에 맞게 가공하여 `inference_recognizer`로 전달하고, 'Fight'일 확률(점수)을 예측받습니다.
            -   예측 점수가 `classification_threshold`보다 높으면 'Fight'(1), 아니면 'NonFight'(0)로 레이블을 지정합니다.
        5.  **중간 결과 저장**: 비디오별 윈도우 결과(`window_results`)를 `output_dir/windows/{label}/{video_name}_windows.pkl` 파일로 저장합니다.
        6.  **비디오 최종 예측**: `_apply_consecutive_event_rule_static` 함수를 호출하여 모든 윈도우 결과를 바탕으로 비디오의 최종 예측 레이블을 결정합니다.

5.  **결과 종합 및 저장**:
    -   모든 비디오 처리가 완료되면, 수집된 `window_results`와 `video_summary`를 각각 `results/window_results.json`과 `results/video_results.json` 파일로 저장합니다.

6.  **성능 평가 및 출력**:
    -   `_calculate_performance_metrics`: `video_results.json`의 실제 레이블과 예측 레이블을 비교하여 혼동 행렬(Confusion Matrix)과 성능 지표(Accuracy, Precision, Recall, F1-Score)를 계산합니다.
    -   계산된 성능을 `results/performance_metrics.json` 파일로 저장하고, 콘솔에 최종 성능 보고서를 출력합니다.

## 5. 설정 파일 (`inference_config.py`)

추론 파이프라인의 모든 동작은 이 설정 파일을 통해 제어됩니다.

| 파라미터 | 설명 |
| --- | --- |
| `mode` | 현재 설정의 모드 ('inference'). |
| `input_dir` | 처리할 비디오가 있는 입력 디렉토리 경로. |
| `output_dir` | 모든 결과물이 저장될 출력 디렉토리 경로. |
| `detector_config` | MMPose RTMO 모델의 설정 파일 경로. |
| `detector_checkpoint` | MMPose RTMO 모델의 사전 학습된 가중치(체크포인트) 파일 경로. |
| `action_config` | MMAction2 ST-GCN++ 모델의 설정 파일 경로. |
| `action_checkpoint` | MMAction2 ST-GCN++ 모델의 학습된 가중치(체크포인트) 파일 경로. |
| `gpu` | 사용할 GPU ID. 쉼표로 구분하여 여러 개 지정 가능 (예: '0,1'). 'cpu'로 설정 시 CPU 사용. |
| `score_thr` | 포즈 추정 시 인물로 판단할 최소 신뢰도 점수. |
| `nms_thr` | 비최대 억제(NMS) 임계값. 겹치는 바운딩 박스를 제거할 때 사용. |
| `track_high_thresh` | ByteTrack에서 추적을 시작하기 위한 높은 신뢰도 임계값. |
| `track_low_thresh` | ByteTrack에서 기존 트랙과 연결하기 위한 낮은 신뢰도 임계값. |
| `track_max_disappeared` | 추적 객체가 몇 프레임 동안 보이지 않으면 추적을 종료할지 결정. |
| `track_min_hits` | 추적을 시작하기 위해 필요한 최소 프레임 수. |
| `quality_threshold` | 추적 대상의 품질을 판단하는 점수 임계값. |
| `min_track_length` | 유효한 트랙으로 간주하기 위한 최소 길이. |
| `movement_weight`, `position_weight`, ... | 추적 대상의 랭킹을 매길 때 사용되는 복합 점수의 가중치. |
| `clip_len` | 행동 인식을 위한 슬라이딩 윈도우의 길이 (프레임 수). |
| `inference_stride` | 추론 시 슬라이딩 윈도우를 이동시킬 간격 (프레임 수). |
| `consecutive_event_threshold` | 비디오를 최종 'Fight'로 판정하기 위해 필요한 연속적인 'Fight' 윈도우의 최소 개수. |
| `focus_person` | 시각화 시 상위 몇 명의 인물에 초점을 맞출지 결정. |
| `classification_threshold` | 윈도우 예측 시 'Fight'로 분류할 확률 임계값. |
| `debug_mode` | `True`로 설정 시, 첫 비디오의 첫 윈도우 처리 시 상세한 디버그 로그를 출력. |
| `debug_single_video` | `True`로 설정 시, 테스트를 위해 단 하나의 비디오만 처리. |

## 6. 출력 파일 구조

`output_dir`에 지정된 경로에 다음과 같은 구조로 결과가 저장됩니다.

```
{output_dir}/
├── windows/
│   ├── Fight/
│   │   └── {video_name}_windows.pkl  # Fight 비디오의 윈도우별 결과 (pickle)
│   └── NonFight/
│       └── {video_name}_windows.pkl  # NonFight 비디오의 윈도우별 결과 (pickle)
├── results/
│   ├── window_results.json         # 모든 비디오의 모든 윈도우 결과 종합
│   ├── video_results.json          # 모든 비디오의 최종 예측 결과 및 요약 정보
│   └── performance_metrics.json    # 최종 성능 평가 지표
└── visualizations/ (시각화 기능 사용 시)
    ├── Fight/
    └── NonFight/
```
