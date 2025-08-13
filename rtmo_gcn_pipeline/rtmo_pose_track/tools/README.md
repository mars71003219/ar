# Tools Directory

RTMO 포즈 추정 및 트래킹 파이프라인을 위한 핵심 도구들입니다.

## 📋 핵심 실행 스크립트

### 🎯 메인 파이프라인
- **`inference_pipeline.py`** - 병렬 추론 파이프라인 (GPU/멀티스레딩 지원)
- **`separated_pose_pipeline.py`** - 분리된 3단계 포즈 처리 파이프라인

### 🔧 개별 실행 도구
- **`run_pose_extraction.py`** - 포즈 추정 전용 실행
- **`run_visualization.py`** - 시각화 결과 생성
- **`run_realtime_detection.py`** - 실시간 감지 실행
- **`run_pose_analysis.py`** - 포즈 분석 도구

## 🛠️ 유틸리티
- **`extract_video_results.py`** - 특정 비디오 결과 추출
- **`pkl_to_json_converter.py`** - PKL → JSON 변환 도구

## 사용법

```bash
# 메인 파이프라인 실행
python inference_pipeline.py --config configs/inference_config.py

# 분리된 처리
python separated_pose_pipeline.py --stage 1

# 개별 포즈 추정
python run_pose_extraction.py --input video.mp4

# 결과 변환
python pkl_to_json_converter.py --input results.pkl --output results.json
```

모든 스크립트는 `--help` 옵션으로 상세 사용법을 확인할 수 있습니다.