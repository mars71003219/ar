# Recognizer 통합 사용 가이드

하나의 메인 파일(`main.py`)로 모든 기능을 실행할 수 있습니다. 더 이상 examples 폴더의 여러 파일을 사용할 필요가 없습니다.

## 🚀 빠른 시작

### 1. 기본 추론 (PKL 파일 생성 없음)
```bash
python recognizer/main.py --mode inference --input video.mp4
```

### 2. 성능 평가 포함 추론 (PKL + 시각화)
```bash
python recognizer/main.py --mode inference --input video.mp4 --enable_evaluation --enable_visualization
```

### 3. 분리 파이프라인 실행
```bash
python recognizer/main.py --mode separated --input data/videos --output output/separated
```

### 4. PKL 파일 시각화
```bash
python recognizer/main.py --mode annotation --pkl_file stage2_result.pkl --video_file original.mp4
```

## 📋 프리셋 설정 사용

미리 준비된 설정 파일을 사용하여 간편하게 실행할 수 있습니다:

### 기본 추론
```bash
python recognizer/main.py --config configs/presets/inference_basic.yaml --input video.mp4
```

### 성능 평가 포함 추론
```bash
python recognizer/main.py --config configs/presets/inference_with_evaluation.yaml --input video.mp4
```

### 분리 파이프라인
```bash
python recognizer/main.py --config configs/presets/separated_pipeline.yaml --input data/videos
```

### 어노테이션 시각화
```bash
python recognizer/main.py --config configs/presets/annotation_visualization.yaml --pkl_file stage2.pkl --video_file video.mp4
```

## 🎛️ 고급 옵션

### 멀티GPU 사용
```bash
# 명령행 플래그로 활성화
python recognizer/main.py --mode inference --input video.mp4 --multi_gpu --gpus 0,1,2,3 --enable_evaluation

# 또는 프리셋 사용
python recognizer/main.py --config configs/presets/multi_gpu_inference.yaml --input video.mp4
```

### 멀티프로세스 사용
```bash
# 명령행 플래그로 활성화
python recognizer/main.py --mode separated --input data/videos --multiprocess --workers 8

# 또는 프리셋 사용  
python recognizer/main.py --config configs/presets/multiprocess_separated.yaml --input data/videos
```

### RTSP 스트림 처리
```bash
python recognizer/main.py --mode inference --input rtsp://192.168.1.100/stream --duration 60 --enable_evaluation
```

### 웹캠 사용
```bash
python recognizer/main.py --mode inference --input 0 --duration 30 --enable_visualization
```

## 📁 출력 구조

실행 후 다음과 같은 구조로 결과가 생성됩니다:

```
output/
├── evaluation/                    # 성능 평가 결과 (--enable_evaluation 시)
│   ├── performance_metrics/
│   │   ├── performance_metrics.json
│   │   └── performance_metrics.pkl
│   ├── results/
│   │   ├── detailed_results.json
│   │   └── detailed_results.csv
│   └── overlay_data/              # 시각화용 PKL 파일들
│       ├── video_window_0_overlay.pkl
│       └── video_window_1_overlay.pkl
├── visualizations/                # 시각화 파일들 (--enable_visualization 시)
│   ├── classification_results.png
│   ├── timeline_visualization.png
│   ├── confusion_matrix.png
│   └── overlay_videos/
│       └── visualization.mp4
└── logs/
    └── recognizer.log
```

## ⚙️ 커스텀 설정

### 1. 설정 파일 생성
`configs/my_config.yaml` 파일을 생성하고 `configs/main_config.yaml`을 참고하여 설정을 조정합니다.

### 2. 설정 파일 사용
```bash
python recognizer/main.py --config configs/my_config.yaml
```

### 3. 명령행에서 설정 오버라이드
```bash
python recognizer/main.py --config configs/my_config.yaml --input new_video.mp4 --device cuda:1
```

## 🔧 주요 설정 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 (inference, separated, annotation, unified) | inference |
| `--input` | 입력 소스 (비디오 파일, RTSP, 웹캠 인덱스) | - |
| `--output_dir` | 출력 디렉토리 | output |
| `--enable_evaluation` | 성능 평가 활성화 | False |
| `--enable_visualization` | 시각화 생성 활성화 | False |
| `--multi_gpu` | 멀티GPU 사용 | False |
| `--multiprocess` | 멀티프로세스 사용 | False |
| `--device` | GPU 디바이스 | cuda:0 |
| `--window_size` | 분류 윈도우 크기 | 100 |
| `--duration` | 처리 시간 제한 (초) | None |

## 🎯 사용 사례별 가이드

### 사례 1: 비디오 파일에서 폭력 탐지 + 성능 분석
```bash
# 실행
python recognizer/main.py --mode inference --input fight_video.mp4 --enable_evaluation --enable_visualization

# 결과 확인
ls output/evaluation/          # 성능 지표 확인
ls output/visualizations/      # 차트 확인
```

### 사례 2: 대량 비디오 배치 처리
```bash
# 분리 파이프라인으로 배치 처리
python recognizer/main.py --mode separated --input data/videos/ --multiprocess --workers 8 --enable_evaluation

# 결과 확인
ls output/separated/stage3_unification/  # 최종 결과 확인
```

### 사례 3: 실시간 CCTV 모니터링
```bash
# RTSP 스트림 실시간 처리
python recognizer/main.py --mode inference --input rtsp://admin:pass@192.168.1.100/stream --enable_evaluation --duration 3600
```

### 사례 4: 기존 결과 시각화
```bash
# Stage2 PKL 파일 시각화
python recognizer/main.py --mode annotation --pkl_file output/separated/stage2_tracking/video_windows.pkl --video_file data/original_video.mp4 --output_video output/annotated_video.mp4
```

## 🚫 더 이상 사용하지 않는 방식

examples 폴더의 개별 스크립트들은 더 이상 사용할 필요가 없습니다:

~~❌ `python examples/inference_with_evaluation.py`~~  
✅ `python main.py --mode inference --enable_evaluation`

~~❌ `python examples/separated_pipeline_usage.py`~~  
✅ `python main.py --mode separated`

~~❌ `python examples/stage2_visualization_example.py`~~  
✅ `python main.py --mode annotation`

## 🆘 문제 해결

### 1. 디버그 모드
```bash
python recognizer/main.py --mode inference --input video.mp4 --debug
```

### 2. 조용한 모드 (에러만 출력)
```bash
python recognizer/main.py --mode inference --input video.mp4 --quiet
```

### 3. 도움말 보기
```bash
python recognizer/main.py --help
```

### 4. 지원되는 모드 확인
```bash
python recognizer/main.py --mode invalid_mode  # 지원되는 모드 목록이 출력됨
```

이제 하나의 통합된 인터페이스로 모든 기능을 사용할 수 있습니다!