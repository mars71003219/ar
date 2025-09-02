# 성능 검증 모드 사용 가이드

테스트 데이터셋을 통한 모델 성능 평가를 위한 evaluation 모드 가이드입니다.

## 개요

evaluation 모드는 기존 inference.analysis 코드를 재활용하여 다음 기능을 제공합니다:

1. **비디오별 추론**: 테스트 데이터셋의 모든 비디오에 대해 윈도우 단위 추론 수행
2. **상세 결과 CSV**: 윈도우별 상세 추론 결과를 CSV로 저장
3. **통합 결과 CSV**: consecutive 설정을 적용한 비디오별 최종 결과를 CSV로 저장
4. **성능 지표 계산**: 혼동행렬, precision, recall, F1-score, mAP 계산
5. **결과 시각화**: 혼동행렬 시각화 및 저장

## 데이터셋 구조

테스트 데이터셋은 다음과 같은 폴더 구조를 가져야 합니다:

```
test_dataset/
├── Fight/           # Fight 클래스 비디오들
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── NonFight/        # NonFight 클래스 비디오들
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

- **폴더명이 클래스 라벨로 사용됩니다**
- Fight 또는 violence가 포함된 폴더명은 클래스 라벨 1 (Fight)로 분류
- 나머지는 클래스 라벨 0 (NonFight)로 분류
- 지원하는 비디오 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv

## 설정 파일

evaluation 모드를 위한 설정 파일 예시 (`configs/evaluation_config.yaml`):

```yaml
# 실행 모드
mode: evaluation

# 평가 설정
evaluation:
  # 테스트 데이터셋 경로
  test_dataset_path: "/path/to/test/dataset"
  
  # 결과 출력 디렉토리
  output_dir: "output/evaluation"
  
  # 연속 프레임 설정 (이 값 이상의 연속 Fight 예측이 있어야 최종 Fight로 판정)
  consecutive_frames: 3

# 모델 설정 (기존 inference 설정과 동일)
models:
  pose_estimation:
    inference_mode: onnx  # pth, onnx, tensorrt 중 선택
    onnx:
      model_name: rtmo_onnx
      model_path: "mmaction2/checkpoints/rtmo-l_16xb16-600e_body7-640x640.onnx"
      device: "cuda:0"
      score_threshold: 0.3
      input_size: [640, 640]
  
  tracking:
    tracker_name: bytetrack
    track_thresh: 0.4
    track_buffer: 50
    match_thresh: 0.4
  
  scoring:
    scorer_name: region_based
    quality_threshold: 0.3
    min_track_length: 10
  
  action_classification:
    model_name: stgcn
    config_file: "mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py"
    checkpoint_path: "mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.pth"
    device: "cuda:0"
    window_size: 100
    confidence_threshold: 0.4
    class_names: ["NonFight", "Fight"]

performance:
  window_size: 100
  window_stride: 50
```

## 실행 방법

### 1. Docker 컨테이너에서 실행

```bash
# Docker 컨테이너에서 실행 (권장)
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --config configs/evaluation_config.yaml --mode evaluation --log-level INFO"
```

### 2. 설정 파일 모드 직접 지정

설정 파일에서 `mode: evaluation`을 설정한 경우:

```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --config configs/evaluation_config.yaml --log-level INFO"
```

### 3. 사용 가능한 모드 확인

```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --list-modes"
```

## 출력 결과

evaluation 모드는 다음 파일들을 생성합니다:

### 1. detailed_results.csv
윈도우별 상세 추론 결과:

| 필드 | 설명 |
|------|------|
| video_clip | 비디오 파일명 |
| video_path | 비디오 파일 경로 |
| window_idx | 윈도우 인덱스 |
| start_frame | 윈도우 시작 프레임 번호 |
| window_size | 윈도우 크기 (100) |
| fight_score | Fight 신뢰도 점수 |
| predict | 예측 결과 (0: NonFight, 1: Fight) |
| predicted_class | 예측 클래스명 |
| label_class | 실제 라벨 클래스명 |
| true_label | 실제 라벨 (0: NonFight, 1: Fight) |

### 2. summary_results.csv
비디오별 통합 결과 (consecutive 설정 적용):

| 필드 | 설명 |
|------|------|
| video_clip | 비디오 파일명 |
| video_path | 비디오 파일 경로 |
| final_predict | 최종 예측 결과 |
| predicted_class | 최종 예측 클래스명 |
| label_class | 실제 라벨 클래스명 |
| true_label | 실제 라벨 |
| performance_type | 성능 분류 (TP/FP/FN/TN) |
| total_windows | 전체 윈도우 개수 |
| fight_windows | Fight로 예측된 윈도우 개수 |
| avg_confidence | 평균 신뢰도 |

### 3. confusion_matrix.png
혼동행렬 시각화 이미지

### 4. 성능 지표 로그
실행 로그에서 다음 성능 지표를 확인할 수 있습니다:

- **Accuracy**: 정확도
- **Precision**: 정밀도
- **Recall**: 재현율 (민감도)
- **F1-Score**: F1 점수
- **mAP**: 평균 정밀도
- **Specificity**: 특이도
- **Confusion Matrix**: 혼동행렬 (TP, FP, FN, TN)

## Consecutive 설정 이해

`consecutive_frames` 설정은 최종 예측 결정에 중요한 역할을 합니다:

- **기본값**: 3
- **의미**: 연속으로 3개 이상의 윈도우에서 Fight로 예측되어야 해당 비디오를 Fight로 최종 판정
- **목적**: False Positive 감소, 더 안정적인 예측

예시:
- 윈도우 예측 결과: [0, 1, 1, 1, 0, 1, 0] (최대 연속 3개의 1이 있음)
- consecutive_frames=3인 경우 → 최종 예측: Fight (1) (3개 이상 조건 만족)
- consecutive_frames=4인 경우 → 최종 예측: Fight (1) (3개 이상 조건 만족)

추가 예시:
- 윈도우 예측 결과: [0, 1, 1, 0, 1, 0] (최대 연속 2개의 1이 있음)
- consecutive_frames=3인 경우 → 최종 예측: NonFight (0) (3개 이상 조건 불만족)
- consecutive_frames=2인 경우 → 최종 예측: Fight (1) (2개 이상 조건 만족)

## 실행 예시

실제 실행 예시와 출력 결과:

```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --config configs/evaluation_config.yaml --mode evaluation --log-level INFO"

# 출력 예시:
# INFO - Starting evaluation mode
# INFO - Test dataset: /path/to/test/dataset
# INFO - Found 100 video files
# INFO - Processing video 1/100: video001.mp4
# ...
# INFO - === Performance Metrics ===
# INFO - Accuracy: 0.9200
# INFO - Precision: 0.8900
# INFO - Recall: 0.9100
# INFO - F1-Score: 0.9000
# INFO - mAP: 0.9350
# INFO - Specificity: 0.9300
# INFO - Confusion Matrix:
# INFO -   TP: 45, FP: 3
# INFO -   FN: 4, TN: 48
# INFO - === Output Files ===
# INFO - Detailed results: output/evaluation/detailed_results.csv
# INFO - Summary results: output/evaluation/summary_results.csv
```

## 주의사항

1. **Docker 환경 필수**: 모든 실행은 Docker 컨테이너 내에서 수행해야 합니다
2. **GPU 메모리**: 대용량 데이터셋 처리시 GPU 메모리 사용량을 모니터링하세요
3. **경로 설정**: 설정 파일의 모든 경로는 Docker 컨테이너 내 경로를 기준으로 설정
4. **의존성**: matplotlib/seaborn이 설치되어 있어야 혼동행렬 시각화가 가능합니다

## 문제 해결

### 일반적인 오류와 해결책

1. **"Dataset path does not exist"**
   - 테스트 데이터셋 경로를 확인하세요
   - Docker 컨테이너 내 경로로 올바르게 마운트되었는지 확인

2. **"No video files found"**
   - 데이터셋 폴더 구조를 확인하세요
   - 지원하는 비디오 형식인지 확인

3. **GPU 메모리 부족**
   - 설정에서 batch_size를 줄이거나
   - 더 작은 input_size 사용

4. **의존성 오류**
   - Docker 컨테이너가 올바르게 설정되었는지 확인
   - 필요한 모델 파일들이 존재하는지 확인

evaluation 모드를 통해 모델의 정량적 성능을 정확하게 평가하고, CSV 결과 파일을 통해 상세한 분석을 수행할 수 있습니다.