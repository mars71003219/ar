# Recognizer - 모듈화된 비디오 분석 시스템

**완전 모듈화된 8-모드 통합 실행 시스템**

RTMO 포즈 추정, ByteTrack 추적, STGCN 행동 분류를 통합한 실시간 및 배치 비디오 분석 프레임워크

## 🎯 주요 특징

- **8개 독립 모드**: 추론 3개 + 어노테이션 5개 모드
- **완전 모듈화**: 각 모드별 독립적 구현
- **설정 파일 중심**: argparse 최소화 (3개 인자만)
- **20초 제한 해결**: 실시간/분석 로직 완전 분리
- **확장 가능**: 새로운 모드 추가 용이

## 🚀 빠른 시작

### 설치
```bash
cd /workspace/recognizer
pip install -e .
```

### 기본 사용법
```bash
# 모드 목록 확인
python main.py --list-modes

# 기본 실행 (분석 모드)
python main.py

# 특정 모드 실행
python main.py --mode inference.analysis
python main.py --mode annotation.stage1
```

## 📁 시스템 구조

```
recognizer/
├── main.py                     # 통합 실행기 (3개 인자만)
├── config.yaml                 # 통합 설정 파일
├── core/                       # 모드 관리 엔진
│   ├── mode_manager.py         # 통합 모드 매니저
│   ├── inference_modes.py      # 추론 모드들
│   └── annotation_modes.py     # 어노테이션 모드들
├── pipelines/                  # 처리 파이프라인
├── models/                     # AI 모델들 
├── utils/                      # 공통 유틸리티
├── visualization/              # 시각화 모듈
└── tools/                      # 보조 도구들
```

## 🎮 8개 실행 모드

### 추론 모드 (Inference)

#### 1. `inference.analysis` - 분석 모드
**목적**: 비디오 → JSON/PKL 파일 생성 (시각화 없음)
```bash
python main.py --mode inference.analysis
```
- 전체 비디오 완전 분석
- JSON 결과 + PKL 데이터 저장
- 20초 제한 문제 완전 해결

#### 2. `inference.realtime` - 실시간 모드  
**목적**: 실시간 디스플레이 + 선택적 저장
```bash
python main.py --mode inference.realtime
```
- 실시간 비디오 스트림 처리
- 라이브 오버레이 표시
- 선택적 결과 비디오 저장

#### 3. `inference.visualize` - 시각화 모드
**목적**: PKL 파일 + 원본 비디오 → 오버레이 비디오
```bash
python main.py --mode inference.visualize
```
- 기존 분석 결과 시각화
- 고품질 오버레이 생성
- 배치 처리 지원

### 어노테이션 모드 (Annotation)

#### 4. `annotation.stage1` - 포즈 추정
**목적**: 비디오 → 포즈 추정 PKL 파일
```bash
python main.py --mode annotation.stage1
```
- RTMO 포즈 추정만 수행
- 키포인트 데이터 저장
- 다음 단계 준비

#### 5. `annotation.stage2` - 트래킹 및 정렬
**목적**: 포즈 PKL → 트래킹/정렬 PKL 파일
```bash
python main.py --mode annotation.stage2
```
- ByteTrack 객체 추적
- 복합 점수 기반 정렬
- 고품질 추적 데이터

#### 6. `annotation.stage3` - 데이터셋 통합
**목적**: 비디오별 PKL → train/val/test 통합 PKL
```bash
python main.py --mode annotation.stage3
```
- 데이터셋 분할 (7:1.5:1.5)
- 모델 학습용 형식 변환
- 메타데이터 생성

#### 7. `annotation.visualize` - 어노테이션 시각화
**목적**: 각 stage별 결과 시각화
```bash
python main.py --mode annotation.visualize
```
- stage1: 포즈 키포인트 표시
- stage2: 추적 ID + 정렬 순위
- stage3: 데이터셋 통계

## ⚙️ 설정 관리

### 통합 설정 파일 (`config.yaml`)

```yaml
# 기본 실행 모드
mode: "inference.analysis"

# 추론 모드 설정
inference:
  analysis:
    input: "video.mp4"
    output_dir: "output/analysis"
  
  realtime:
    input: "video.mp4"
    save_output: false
    display_width: 1280
    display_height: 720
  
  visualize:
    results_dir: "output/analysis"
    video_file: "video.mp4"
    save_mode: false

# 어노테이션 모드 설정
annotation:
  stage1:
    input_dir: "/workspace/videos"
    output_dir: "output/stage1"
  
  stage2:
    poses_dir: "output/stage1"
    output_dir: "output/stage2"
  
  stage3:
    tracking_dir: "output/stage2"
    output_dir: "output/stage3"
    split_ratios: {train: 0.7, val: 0.15, test: 0.15}
  
  visualize:
    stage: "stage2"
    results_dir: "output/stage2"
    video_dir: "/workspace/videos"

# 모델 설정 (모든 모드 공통)
models:
  pose_estimation: {...}
  tracking: {...}
  action_classification: {...}
```

### 실행 인자 (최소화)

```bash
python main.py [OPTIONS]

OPTIONS:
  --config FILE        설정 파일 경로 (기본: config.yaml)
  --mode MODE         실행 모드 오버라이드
  --log-level LEVEL   로그 레벨 (DEBUG/INFO/WARNING/ERROR)
  --list-modes        사용 가능한 모드 목록
```

## 📋 워크플로우 예시

### 완전한 분석 워크플로우
```bash
# 1단계: 분석 수행
python main.py --mode inference.analysis
# → output/analysis/json/, pkl/ 생성

# 2단계: 결과 시각화
python main.py --mode inference.visualize
# → 실시간 오버레이 또는 비디오 저장
```

### 어노테이션 파이프라인
```bash
# 1단계: 포즈 추정
python main.py --mode annotation.stage1
# → output/stage1/*.pkl

# 2단계: 트래킹 및 정렬
python main.py --mode annotation.stage2
# → output/stage2/*.pkl

# 3단계: 데이터셋 통합
python main.py --mode annotation.stage3
# → output/stage3/train.pkl, val.pkl, test.pkl

# 4단계: 결과 확인
python main.py --mode annotation.visualize
# → stage별 시각화
```

## 🔧 고급 사용법

### 사용자 정의 설정
```bash
# 커스텀 설정 파일
python main.py --config my_config.yaml --mode inference.analysis

# 특정 로그 레벨
python main.py --mode annotation.stage1 --log-level DEBUG
```

### 배치 처리
```yaml
# config.yaml에서 폴더 처리 설정
inference:
  analysis:
    input_dir: "/workspace/videos"  # 폴더 처리
    output_dir: "output/batch"
```

### 성능 최적화
```yaml
# 성능 설정
performance:
  device: "cuda:0"
  window_size: 100
  window_stride: 50
  batch_size: 8
```

## 🏗️ AI 모델 아키텍처

### 포즈 추정 (RTMO)
- **입력**: 비디오 프레임
- **출력**: 17개 키포인트 좌표
- **특징**: 실시간 다중 객체 지원

### 객체 추적 (ByteTrack)
- **입력**: 프레임별 포즈 박스
- **출력**: 추적 ID + 궤적
- **특징**: 하이브리드 매칭 (IoU + 키포인트)

### 행동 분류 (STGCN)
- **입력**: 100프레임 키포인트 시퀀스
- **출력**: Fight/NonFight 확률
- **특징**: 시공간 그래프 컨볼루션

## 📊 출력 형식

### JSON 결과 (`results.json`)
```json
{
  "input_video": "video.mp4",
  "total_frames": 3000,
  "total_windows": 59,
  "classification_results": [
    {
      "window_id": 0,
      "window_start": 0,
      "window_end": 100,
      "predicted_class": "NonFight",
      "confidence": 0.823,
      "probabilities": [0.823, 0.177]
    }
  ]
}
```

### PKL 데이터 구조
- **프레임 포즈**: `{frame_id: [(x,y,score), ...], ...}`
- **윈도우 어노테이션**: `{keypoints, scores, tracking_ids, ...}`
- **분류 결과**: `{window_data, predictions, metadata}`

## 🛠️ 문제 해결

### 자주 발생하는 오류

1. **"Failed to create pose_estimator module"**
   ```bash
   # 모델 경로 확인
   ls /workspace/mmpose/checkpoints/
   ```

2. **"Input directory does not exist"**
   ```bash
   # 입력 경로 확인
   python main.py --mode inference.analysis --log-level DEBUG
   ```

3. **"CUDA out of memory"**
   ```yaml
   # config.yaml에서 배치 크기 조정
   performance:
     batch_size: 4  # 기본값 8에서 감소
   ```

### 디버깅 모드
```bash
# 상세 로그로 문제 진단
python main.py --mode [MODE] --log-level DEBUG
```

## 🔗 관련 프로젝트

- **MMPose**: 포즈 추정 프레임워크
- **MMAction2**: 행동 인식 프레임워크  
- **ByteTrack**: 다중 객체 추적

## 📜 라이선스

OpenMMLab 라이선스 정책을 따릅니다.

---

**완전 모듈화된 8-모드 통합 시스템으로 간편하고 강력한 비디오 분석을 경험하세요!** 🚀