# STGCN++ Violence Detection - 빠른 시작 가이드

## 📋 목차

1. [사전 준비](#사전-준비)
2. [설치 및 설정](#설치-및-설정)
3. [5분 빠른 시작](#5분-빠른-시작)
4. [기본 사용법](#기본-사용법)
5. [결과 해석](#결과-해석)
6. [문제 해결](#문제-해결)

---

## 🔧 사전 준비

### 필수 요구사항

- **Python**: 3.8 이상
- **CUDA**: 11.0 이상 (GPU 사용 시)
- **RAM**: 8GB 이상 권장
- **Storage**: 10GB 이상 여유 공간

### 필수 패키지

```bash
# PyTorch 설치
pip install torch torchvision torchaudio

# OpenMMLab 패키지
pip install mmpose mmaction2 mmengine mmcv

# 기타 의존성
pip install opencv-python numpy matplotlib
```

---

## ⚙️ 설치 및 설정

### 1. 파이프라인 설정

```bash
cd /home/gaonpf/hsnam/mmlabs/rtmo_gcn_pipeline/inference_pipeline

# 자동 설정 실행
python setup_pipeline.py
```

### 2. 모델 파일 확인

```bash
# 빠른 검증
python quick_test.py
```

**결과 예시:**
```
=== 파이프라인 빠른 테스트 ===
1. 설정 검증...
✓ 설정 검증 통과
2. GPU 사용 가능성 확인...
✓ GPU 사용 가능
3. 핵심 모듈 import 테스트...
✓ 모든 모듈 import 성공
=== 빠른 테스트 완료 ===
```

---

## ⚡ 5분 빠른 시작

### 단계 1: 샘플 데이터 준비

```bash
# 샘플 어노테이션 파일 생성 (자동)
cat sample_annotations.txt
```
```
Fight_1.mp4,1
Fight_2.mp4,1
NonFight_1.mp4,0
NonFight_2.mp4,0
```

### 단계 2: 첫 번째 추론 실행

```bash
# 단일 비디오 처리 (가장 간단한 방법)
python run_inference.py \
    --mode single \
    --input /path/to/your/video.mp4 \
    --output ./my_first_results
```

### 단계 3: 결과 확인

```bash
# 결과 디렉토리 확인
ls -la ./my_first_results/

# 결과 파일 보기
cat ./my_first_results/video_result.json
```

**결과 예시:**
```json
{
  "video_name": "test_video.mp4",
  "classification": {
    "prediction": 1,
    "prediction_label": "Fight",
    "confidence": 0.847
  },
  "processing_time": 12.3,
  "status": "success"
}
```

---

## 📖 기본 사용법

### 모드별 실행 방법

#### 1. 단일 비디오 모드

```bash
python run_inference.py \
    --mode single \
    --input video.mp4 \
    --annotations annotations.txt \
    --generate-overlay
```

**특징:**
- 한 개의 비디오만 처리
- 빠른 테스트에 적합
- 오버레이 비디오 생성 가능

#### 2. 배치 처리 모드

```bash
python run_inference.py \
    --mode batch \
    --input /path/to/videos/ \
    --annotations annotations.txt \
    --batch-size 4
```

**특징:**
- 폴더 내 모든 비디오 처리
- 병렬 처리로 효율적
- 개별 결과 자동 저장

#### 3. 벤치마크 모드

```bash
python run_inference.py \
    --mode benchmark \
    --input /path/to/test_videos/ \
    --annotations annotations.txt \
    --generate-overlay
```

**특징:**
- 성능 평가 전용
- 상세한 메트릭 제공
- 혼동 행렬 및 분석 보고서 생성

### Python API 직접 사용

```python
from main_pipeline import EndToEndPipeline

# 초기화
pipeline = EndToEndPipeline(
    pose_config="configs/rtmo_config.py",
    pose_checkpoint="checkpoints/rtmo.pth",
    gcn_config="configs/stgcn_config.py", 
    gcn_checkpoint="checkpoints/stgcn.pth"
)

# 단일 비디오 처리
result = pipeline.process_single_video("test.mp4")

print(f"예측 결과: {result['classification']['prediction_label']}")
print(f"신뢰도: {result['classification']['confidence']:.3f}")
```

---

## 📊 결과 해석

### 1. 분류 결과

```json
{
  "classification": {
    "prediction": 1,                    // 0: NonFight, 1: Fight
    "prediction_label": "Fight",        // 사람이 읽기 쉬운 라벨
    "confidence": 0.847,               // 예측 신뢰도 (0.0~1.0)
    "window_predictions": [1,1,0,1,1], // 윈도우별 예측
    "window_confidences": [0.9,0.8,0.6,0.85,0.9] // 윈도우별 신뢰도
  }
}
```

**해석 가이드:**
- **confidence > 0.8**: 매우 확실한 예측
- **confidence 0.6-0.8**: 보통 확실한 예측  
- **confidence < 0.6**: 불확실한 예측 (추가 검토 필요)

### 2. 성능 메트릭 (벤치마크 모드)

```json
{
  "metrics": {
    "accuracy": 0.85,        // 전체 정확도
    "precision": 0.82,       // 정밀도 (False Alarm 방지)
    "recall": 0.88,          // 재현율 (누락 방지)
    "f1_score": 0.85         // F1 점수 (균형 지표)
  }
}
```

**메트릭 해석:**
- **Precision 중시**: False Alarm 최소화가 중요한 경우
- **Recall 중시**: 폭력 상황 누락 방지가 중요한 경우
- **F1-Score**: 전반적인 균형 성능

### 3. Fight-우선 트래킹 분석

```json
{
  "tracking": {
    "sequence_length": 30,
    "selected_keypoints_shape": [30, 17, 2],  // 30프레임, 17키포인트, XY좌표
    "selected_scores_shape": [30, 17]         // 키포인트별 신뢰도
  }
}
```

**트래킹 품질 지표:**
- **중앙 영역 집중도**: 싸움이 화면 중앙에서 발생하는지
- **움직임 강도**: 격렬한 동작이 감지되는지
- **인물간 상호작용**: 복수 인물이 가까이 있는지

---

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. GPU 메모리 부족

**증상:**
```
CUDA out of memory. Tried to allocate X GB
```

**해결 방법:**
```bash
# CPU 모드로 실행
python run_inference.py --device cpu

# 배치 크기 줄이기
python run_inference.py --batch-size 2
```

#### 2. 모델 파일 없음

**증상:**
```
FileNotFoundError: No such file or directory: '/path/to/checkpoint.pth'
```

**해결 방법:**
```bash
# 모델 경로 확인
python quick_test.py

# 설정 파일 수정
nano config.py
```

#### 3. 의존성 버전 충돌

**증상:**
```
ImportError: cannot import name 'xxx' from 'mmpose'
```

**해결 방법:**
```bash
# 패키지 재설치
pip uninstall mmpose mmaction2
pip install mmpose mmaction2

# 버전 확인
python -c "import mmpose; print(mmpose.__version__)"
```

### 성능 최적화 팁

#### 1. 처리 속도 향상

```bash
# 오버레이 생성 비활성화 (속도 우선)
python run_inference.py --mode batch

# GPU 메모리 최적화
export CUDA_VISIBLE_DEVICES=0
```

#### 2. 정확도 향상

```python
# config.py에서 파라미터 조정
INFERENCE_CONFIG = {
    'pose_score_threshold': 0.5,    # 더 엄격한 포즈 필터링
    'confidence_threshold': 0.7,    # 더 확실한 예측만 채택
}
```

### 로그 및 디버깅

#### 상세 로그 확인

```bash
# 실행 중 상세 로그
python run_inference.py --verbose

# 로그 파일 실시간 확인
tail -f inference.log
```

#### 설정 검증

```bash
# 실제 실행 없이 설정만 확인
python run_inference.py --dry-run
```

---

## 🎯 다음 단계

### 고급 사용법 학습

1. **API_GUIDE.md**: 상세한 API 문서
2. **run_example.sh**: 다양한 실행 예제
3. **config.py**: 세부 설정 조정

### 커스터마이징

1. **Fight-우선 트래킹 파라미터** 조정
2. **윈도우 기반 분류** 최적화  
3. **성능 메트릭** 추가 정의

### 통합 및 배포

1. **웹 API** 개발
2. **실시간 스트리밍** 적용
3. **대용량 데이터** 배치 처리

---

## 📞 지원 및 문의

- **문제 신고**: GitHub Issues
- **기술 문의**: 개발팀 이메일
- **사용법 질문**: API 가이드 참조

**성공적인 폭력 검출 시스템 구축을 위해 이 가이드를 활용해보세요!** 🚀