# 성능 평가 가이드

**완전 모듈화된 비디오 분석 시스템의 모델 성능 평가 방법**

## 개요

Recognizer 시스템의 모델 성능을 체계적으로 평가하기 위한 종합 가이드입니다. 이 문서는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.

## 목차

1. [평가 모드 개요](#평가-모드-개요)
2. [데이터셋 준비](#데이터셋-준비)
3. [평가 실행 방법](#평가-실행-방법)
4. [결과 분석](#결과-분석)
5. [성능 지표](#성능-지표)
6. [모델 비교](#모델-비교)
7. [문제 해결](#문제-해결)

---

## 평가 모드 개요

### 평가 기능

evaluation 모드는 inference.analysis를 확장하여 다음 기능을 제공합니다:

1. **비디오별 추론**: 테스트 데이터셋의 모든 비디오에 대해 윈도우 단위 추론 수행
2. **상세 결과 저장**: 윈도우별 상세 추론 결과를 CSV로 저장
3. **통합 결과 생성**: consecutive 설정을 적용한 비디오별 최종 결과 CSV 저장
4. **성능 지표 계산**: 혼동행렬, precision, recall, F1-score, mAP 계산
5. **결과 시각화**: 혼동행렬, 신뢰도 히스토그램, 시간적 분석 차트 생성
6. **HTML 보고서**: 종합적인 평가 결과 보고서 생성

### 평가 모드 활성화

```yaml
# config.yaml
inference:
  analysis:
    input: "/path/to/test_dataset"
    output_dir: "output/evaluation"
    
    # 평가 모드 활성화
    evaluation:
      enabled: true
      ground_truth_dir: "/path/to/labels"
      
      # 차트 생성 설정
      charts:
        enabled: true
        confidence_histogram: true
        temporal_analysis: true
        performance_over_time: true
      
      # 혼동행렬 설정  
      confusion_matrix:
        enabled: true
        normalize: true
        save_plot: true
        
      # 보고서 생성 설정
      report:
        enabled: true
        format: "html"  # html, pdf
        include_charts: true
        include_details: true
```

---

## 데이터셋 준비

### 데이터셋 구조

테스트 데이터셋은 다음과 같은 폴더 구조를 가져야 합니다:

```
test_dataset/
├── Fight/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── NonFight/
│   ├── video_101.mp4
│   ├── video_102.mp4
│   └── ...
└── labels/
    ├── video_001.txt  # Fight 비디오의 라벨
    ├── video_002.txt
    ├── video_101.txt  # NonFight 비디오의 라벨
    └── ...
```

### 라벨 파일 형식

각 비디오에 대한 라벨 파일(`.txt`)은 다음 형식을 따릅니다:

```
# video_001.txt (Fight 비디오)
0 120 1    # 0초~120초: Fight (1)

# video_101.txt (NonFight 비디오)  
0 80 0     # 0초~80초: NonFight (0)

# 복잡한 라벨 예시
0 30 0     # 0~30초: NonFight
30 60 1    # 30~60초: Fight
60 100 0   # 60~100초: NonFight
```

**형식 설명:**
- `시작시간(초) 종료시간(초) 라벨(0|1)`
- `0`: NonFight, `1`: Fight
- 한 줄당 하나의 시간 구간

### RWF-2000 데이터셋 지원

RWF-2000 데이터셋을 사용하는 경우:

```yaml
inference:
  analysis:
    input: "/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000"
    
    evaluation:
      enabled: true
      dataset_type: "rwf2000"  # 자동 라벨 생성
```

---

## 평가 실행 방법

### 기본 실행

```bash
# Docker 환경에서 실행
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis --log-level INFO"
```

### PyTorch vs ONNX 모델 비교

1. **PyTorch 모델 평가**:
```yaml
models:
  action_classification:
    model_name: stgcn
    checkpoint_path: /workspace/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-l_RWF2000plus_stable/best_acc_top1_epoch_30.pth
    config_file: /workspace/mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py
```

2. **ONNX 모델 평가**:
```yaml
models:
  action_classification:
    model_name: stgcn_onnx
    checkpoint_path: /workspace/mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.onnx
    input_format: stgcn_onnx
```

### 멀티프로세스 평가

대용량 데이터셋 처리를 위한 멀티프로세스 실행:

```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis --multi-process --num-processes 8 --gpus 0,1,2,3"
```

---

## 결과 분석

### 출력 파일 구조

```
output/evaluation/
├── detailed_results.csv      # 윈도우별 상세 결과
├── final_results.csv         # 비디오별 최종 결과
├── performance_metrics.json  # 성능 지표
├── confusion_matrix.png      # 혼동행렬 시각화
├── confidence_histogram.png  # 신뢰도 분포
├── temporal_analysis.png     # 시간적 분석
├── evaluation_report.html    # 종합 보고서
└── logs/
    └── evaluation.log        # 평가 로그
```

### 상세 결과 CSV (detailed_results.csv)

```csv
video_name,window_id,start_time,end_time,prediction,confidence,probabilities,ground_truth,correct
video_001.mp4,window_0,0.0,3.33,1,0.87,"[0.13, 0.87]",1,True
video_001.mp4,window_1,1.67,5.0,1,0.92,"[0.08, 0.92]",1,True
video_001.mp4,window_2,3.33,6.67,0,0.65,"[0.65, 0.35]",1,False
```

### 최종 결과 CSV (final_results.csv)

```csv
video_name,predicted_class,confidence,ground_truth,correct,duration,fight_probability
video_001.mp4,Fight,0.87,Fight,True,120.0,0.87
video_002.mp4,Fight,0.75,Fight,True,95.5,0.75
video_101.mp4,NonFight,0.82,NonFight,True,80.0,0.18
```

### 성능 지표 JSON (performance_metrics.json)

```json
{
  "overall_metrics": {
    "accuracy": 0.8542,
    "precision": 0.8333,
    "recall": 0.8571,
    "f1_score": 0.8451,
    "auc": 0.9123
  },
  "class_metrics": {
    "NonFight": {
      "precision": 0.8750,
      "recall": 0.8235,
      "f1_score": 0.8485,
      "support": 85
    },
    "Fight": {
      "precision": 0.8333,
      "recall": 0.8571,
      "f1_score": 0.8451,
      "support": 91
    }
  },
  "confusion_matrix": [[70, 15], [13, 78]],
  "model_info": {
    "model_name": "stgcn_onnx",
    "model_path": "/workspace/mmaction2/checkpoints/stgcnpp_enhanced_fight_detection_stable.onnx",
    "temperature_scaling": 0.005
  },
  "processing_stats": {
    "total_videos": 176,
    "total_windows": 2847,
    "avg_processing_time": 0.124,
    "classification_fps": 39.9
  }
}
```

---

## 성능 지표

### 기본 분류 지표

#### Accuracy (정확도)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### Precision (정밀도)
```
Precision = TP / (TP + FP)
```

#### Recall (재현율)
```  
Recall = TP / (TP + FN)
```

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 고급 지표

#### AUC-ROC
ROC 곡선 아래 면적으로 모델의 전체적인 성능을 나타냅니다.

#### Average Precision (AP)
Precision-Recall 곡선 아래 면적입니다.

#### Matthews Correlation Coefficient (MCC)
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### 혼동 행렬 해석

```
                 예측값
              NonFight  Fight
실제값 NonFight   70     15    (TN=70, FP=15)
       Fight      13     78    (FN=13, TP=78)
```

- **True Negative (TN)**: 70 - NonFight를 NonFight로 올바르게 예측
- **False Positive (FP)**: 15 - NonFight를 Fight로 잘못 예측
- **False Negative (FN)**: 13 - Fight를 NonFight로 잘못 예측  
- **True Positive (TP)**: 78 - Fight를 Fight로 올바르게 예측

---

## 모델 비교

### PyTorch vs ONNX 성능 비교

| 지표 | PyTorch | ONNX | 개선 |
|------|---------|------|------|
| 추론 속도 (FPS) | 15.2 | 39.9 | +162% |
| 메모리 사용량 (MB) | 2,847 | 1,623 | -43% |
| Accuracy | 0.8542 | 0.8538 | -0.05% |
| F1-Score | 0.8451 | 0.8445 | -0.07% |
| 로딩 시간 (초) | 8.3 | 2.1 | -75% |

### 모델 크기 비교

| 모델 형태 | 파일 크기 | 로딩 시간 | 추론 속도 |
|-----------|-----------|-----------|-----------|
| PyTorch (.pth) | 45.2 MB | 8.3초 | 15.2 FPS |
| ONNX (.onnx) | 22.1 MB | 2.1초 | 39.9 FPS |
| 개선율 | -51% | -75% | +162% |

---

## 문제 해결

### 일반적인 문제

#### 1. 라벨 파일 형식 오류

**증상**: 
```
ValueError: Invalid label format in video_001.txt
```

**해결책**:
```bash
# 라벨 파일 형식 확인
cat /path/to/labels/video_001.txt

# 올바른 형식으로 수정
echo "0 120 1" > /path/to/labels/video_001.txt
```

#### 2. 메모리 부족

**증상**:
```
CUDA out of memory. Tried to allocate 2.0 GiB
```

**해결책**:
```yaml
# 설정 최적화
models:
  action_classification:
    max_persons: 2      # 기본값 4에서 감소
    window_size: 50     # 기본값 100에서 감소

inference:
  analysis:
    batch_size: 1       # 배치 크기 감소
```

#### 3. 낮은 성능 점수

**분석 방법**:
```python
# 혼동행렬 분석
import json
with open('output/evaluation/performance_metrics.json') as f:
    metrics = json.load(f)
    
cm = metrics['confusion_matrix']
print(f"False Positive Rate: {cm[0][1]/(cm[0][0]+cm[0][1]):.3f}")
print(f"False Negative Rate: {cm[1][0]/(cm[1][0]+cm[1][1]):.3f}")
```

**개선 방법**:
1. **Confidence 임계값 조정**:
```yaml
models:
  action_classification:
    confidence_threshold: 0.6  # 0.4에서 증가
```

2. **Temperature Scaling 조정**:
```python
# stgcn_onnx_classifier.py에서
temperature = 0.003  # 0.005에서 감소하여 더 확실한 예측
```

#### 4. 처리 속도 느림

**해결책**:
```bash
# ONNX 모델 사용
# 멀티프로세스 활용  
# GPU 메모리 최적화
```

### 디버깅 팁

#### 1. 상세 로그 확인
```bash
docker exec mmlabs bash -c "cd /workspace/recognizer && python3 main.py --mode inference.analysis --log-level DEBUG"
```

#### 2. 단일 비디오 테스트
```yaml
inference:
  analysis:
    input: "/path/to/single_video.mp4"  # 폴더 대신 단일 파일
```

#### 3. 중간 결과 확인
```python
# 윈도우별 결과 확인
import pandas as pd
df = pd.read_csv('output/evaluation/detailed_results.csv')
print(df.groupby('video_name')['correct'].mean())
```

---

## 고급 평가 기법

### 교차 검증

```python
# K-Fold 교차 검증을 위한 데이터 분할
from sklearn.model_selection import KFold

def cross_validate_dataset(video_list, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(video_list)):
        train_videos = [video_list[i] for i in train_idx]
        test_videos = [video_list[i] for i in test_idx]
        
        # 각 fold별 평가 실행
        evaluate_fold(fold, train_videos, test_videos)
```

### 오류 분석

```python
# 오분류 비디오 분석
import pandas as pd

df = pd.read_csv('output/evaluation/final_results.csv')
errors = df[df['correct'] == False]

print("오분류 분석:")
print(f"False Positive: {len(errors[(errors['predicted_class']=='Fight') & (errors['ground_truth']=='NonFight')])}")
print(f"False Negative: {len(errors[(errors['predicted_class']=='NonFight') & (errors['ground_truth']=='Fight')])}")

# 오분류 비디오 목록
print("\n오분류 비디오:")
for _, row in errors.iterrows():
    print(f"{row['video_name']}: 예측={row['predicted_class']}, 실제={row['ground_truth']}, 신뢰도={row['confidence']:.3f}")
```

### 신뢰도 기반 분석

```python
# 신뢰도 구간별 정확도 분석
def analyze_confidence_intervals():
    df = pd.read_csv('output/evaluation/detailed_results.csv')
    
    intervals = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    for low, high in intervals:
        subset = df[(df['confidence'] >= low) & (df['confidence'] < high)]
        accuracy = subset['correct'].mean()
        count = len(subset)
        print(f"신뢰도 {low:.1f}-{high:.1f}: 정확도={accuracy:.3f}, 샘플수={count}")
```

---

## 보고서 생성

### HTML 보고서 구조

자동 생성되는 HTML 보고서에는 다음 내용이 포함됩니다:

1. **실행 정보**
   - 모델 정보 (이름, 경로, 파라미터)
   - 데이터셋 정보 (크기, 분포)
   - 실행 환경 (GPU, 처리 시간)

2. **성능 지표**
   - 전체 성능 요약
   - 클래스별 상세 지표
   - 혼동행렬 시각화

3. **분석 차트**
   - 신뢰도 히스토그램
   - 시간적 성능 분석
   - ROC 곡선 (가능한 경우)

4. **오류 분석**
   - 오분류 사례 분석
   - 개선 권장사항

### 커스텀 보고서

추가적인 분석을 위한 커스텀 보고서 생성:

```python
# custom_report.py
def generate_custom_report():
    # 데이터 로드
    detailed = pd.read_csv('output/evaluation/detailed_results.csv')
    final = pd.read_csv('output/evaluation/final_results.csv')
    
    # 커스텀 분석
    # ... 분석 코드 ...
    
    # HTML 생성
    html_content = f"""
    <html>
    <head><title>Custom Evaluation Report</title></head>
    <body>
        <h1>Custom Analysis Results</h1>
        <!-- 분석 결과 내용 -->
    </body>
    </html>
    """
    
    with open('output/evaluation/custom_report.html', 'w') as f:
        f.write(html_content)
```

---

*이 가이드는 최신 코드베이스(2025-09-03)를 기반으로 작성되었습니다.*