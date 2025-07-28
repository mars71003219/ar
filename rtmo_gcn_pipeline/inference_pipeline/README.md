# End-to-End STGCN++ Violence Detection Inference Pipeline

이 파이프라인은 학습된 RTMO-m 포즈 추정 모델과 STGCN++ 분류 모델을 사용하여 비디오에서 폭력 행동을 검출하는 완전한 추론 시스템입니다.

## 주요 기능

1. **비디오/폴더 입력 처리**: 단일 비디오 클립 또는 복수 비디오가 있는 폴더 처리
2. **RTMO-m 포즈 추정**: 학습된 MMPose RTMO-m 모델로 포즈 추정
3. **Fight-우선 트래킹**: 움직임, 영역, 상호작용 등을 고려한 정렬 데이터 생성
4. **STGCN++ 예측**: 학습된 모델 가중치로 Fight/NonFight 분류
5. **성능 메트릭 계산**: TP/TN/FP/FN 기반 정확도, 정밀도, 재현율, F1-score 산출
6. **오버레이 비디오 생성**: 관절과 추론 결과가 표시된 비디오 파일 생성

## 디렉토리 구조

```
inference_pipeline/
├── README.md                    # 이 파일
├── config.py                    # 파이프라인 설정
├── pose_estimator.py           # RTMO-m 포즈 추정 모듈
├── fight_tracker.py            # Fight-우선 트래킹 시스템
├── action_classifier.py        # STGCN++ 분류 모듈
├── metrics_calculator.py       # 성능 메트릭 계산
├── video_overlay.py            # 오버레이 비디오 생성
├── main_pipeline.py            # 메인 파이프라인 통합
└── run_inference.py            # 실행 스크립트
```

## 사용법

### 1. 단일 비디오 처리
```bash
python run_inference.py \
    --input /path/to/video.mp4 \
    --annotations /path/to/annotations.txt \
    --label-map /path/to/label_map.txt \
    --output /path/to/output/
```

### 2. 폴더 처리
```bash
python run_inference.py \
    --input /path/to/videos/ \
    --annotations /path/to/annotations.txt \
    --label-map /path/to/label_map.txt \
    --output /path/to/output/ \
    --batch-size 4
```

### 3. 벤치마크 모드
```bash
python run_inference.py \
    --input /path/to/test_videos/ \
    --annotations /path/to/annotations.txt \
    --label-map /path/to/label_map.txt \
    --output /path/to/output/ \
    --mode benchmark \
    --generate-overlay
```

## 입력 파일 형식

### annotations.txt
```
video1.mp4,1
video2.mp4,0
video3.mp4,1
```

### label_map.txt
```
Fight: 1
NonFight: 0
```

## 출력 결과

- `results.json`: 전체 결과 요약
- `metrics.json`: 성능 메트릭 상세
- `predictions/`: 개별 비디오 예측 결과
- `overlays/`: 오버레이 비디오 파일 (옵션)
- `analysis_report.md`: 상세 분석 보고서