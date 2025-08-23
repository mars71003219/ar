# PyTorch to ONNX Converter for STGCN Fight Detection

STGCN Fight Detection 모델을 PyTorch에서 ONNX 형식으로 변환하는 도구입니다.

## 📋 기능

- ✅ **동적/정적 입력 크기 지원**: 실시간 추론용 동적 크기, 배치 처리용 정적 크기
- ✅ **다양한 최적화 옵션**: 상수 폴딩, opset 버전 선택
- ✅ **자동 검증**: PyTorch vs ONNX 출력 결과 비교  
- ✅ **성능 벤치마크**: 변환 전후 성능 측정
- ✅ **사전 정의된 프리셋**: 용도별 최적화된 설정
- ✅ **상세한 로깅**: 변환 과정 및 결과 상세 출력

## 🚀 빠른 시작

### 1. 환경 설정 확인

```bash
# 필요한 패키지 설치 확인
pip install torch onnx onnxruntime mmaction2
```

### 2. 간편 스크립트 사용

```bash
# 실시간 추론용 모델 변환 (권장)
./tools/convert_stgcn_to_onnx.sh realtime

# 배치 처리용 모델 변환
./tools/convert_stgcn_to_onnx.sh batch

# 개발/디버깅용 모델 변환
./tools/convert_stgcn_to_onnx.sh development
```

### 3. Python 스크립트 직접 사용

```bash
# 기본 변환 (동적 크기)
python3 tools/pytorch_to_onnx_converter.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --output /path/to/output.onnx \
    --dynamic \
    --verify

# 정적 크기 변환 (성능 최적화)  
python3 tools/pytorch_to_onnx_converter.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --output /path/to/output.onnx \
    --batch-size 8 \
    --benchmark
```

## 📖 사용법

### 스크립트 옵션

```bash
./tools/convert_stgcn_to_onnx.sh [프리셋] [옵션]

# 프리셋:
#   realtime     - 실시간 추론용 (동적 프레임, 배치=1)
#   batch        - 배치 처리용 (정적 크기, 배치=8)
#   development  - 개발용 (최적화 비활성화)
#   custom       - 사용자 정의 설정

# 옵션:
#   -h, --help         도움말 출력
#   -d, --device       디바이스 설정 (cuda:0, cpu)
#   -o, --output       출력 파일 경로
#   -v, --verbose      상세 출력
#   --no-verify        검증 생략
#   --benchmark        성능 벤치마크 실행
```

### Python 스크립트 옵션

```bash
python3 tools/pytorch_to_onnx_converter.py --help

# 주요 옵션:
#   --config              MMAction2 설정 파일
#   --checkpoint          PyTorch 체크포인트 파일
#   --output              출력 ONNX 파일 경로
#   --dynamic             동적 입력 크기 활성화
#   --dynamic-frames      프레임 수 동적 설정
#   --batch-size          배치 크기 (기본: 1)
#   --num-frames          프레임 수 (기본: 100)
#   --opset-version       ONNX opset 버전 (기본: 11)
#   --verify              출력 검증 (기본: True)
#   --benchmark           성능 벤치마크
```

## 📊 사용 예시

### 1. 실시간 추론용 변환

```bash
# 동적 프레임 크기, 고정 배치 크기
./tools/convert_stgcn_to_onnx.sh realtime -d cuda:0 -v

# 결과: 윈도우 크기 가변 지원 (50~200 프레임)
# 출력: checkpoints/stgcn_fight_realtime.onnx
```

### 2. 배치 처리용 변환

```bash  
# 정적 크기, 높은 처리량
./tools/convert_stgcn_to_onnx.sh batch --benchmark

# 결과: 고정 크기로 최적화된 성능
# 출력: checkpoints/stgcn_fight_batch.onnx
```

### 3. 사용자 정의 변환

```bash
python3 tools/pytorch_to_onnx_converter.py \
    --config mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py \
    --checkpoint mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable/best_acc_top1_epoch_14.pth \
    --output checkpoints/my_custom_model.onnx \
    --batch-size 4 \
    --num-frames 150 \
    --dynamic \
    --dynamic-frames \
    --opset-version 12 \
    --verify \
    --benchmark \
    --verbose
```

## 🔧 입력 형태

STGCN 모델의 입력 형태는 다음과 같습니다:

```
[N, M, T, V, C]
- N: Batch Size (배치 크기)
- M: Max Persons (최대 인원 수) = 4
- T: Time Frames (시간 프레임) = 100 (가변 가능)
- V: Keypoints (키포인트 수) = 17 (COCO)
- C: Coordinates (좌표 차원) = 2 (x, y)
```

### 동적 축 설정

```yaml
# 실시간 추론용 (프레임 수 가변)
dynamic_axes:
  input: {0: 'batch_size', 2: 'num_frames'}
  output: {0: 'batch_size'}

# 배치 처리용 (모든 축 고정)  
dynamic_axes: null
```

## 📈 성능 비교

| 모델 타입 | 배치 크기 | 프레임 수 | 추론 시간 | 메모리 사용량 | 권장 용도 |
|----------|----------|----------|----------|------------|-----------|
| PyTorch  | 1        | 100      | ~15ms    | ~800MB     | 개발/디버깅 |
| ONNX (정적) | 1     | 100      | ~8ms     | ~400MB     | 실시간 추론 |
| ONNX (동적) | 1     | 50-200   | ~8-20ms  | ~400-600MB | 유연한 추론 |
| ONNX (배치) | 8     | 100      | ~45ms    | ~1.2GB     | 배치 처리 |

## 🔍 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   --batch-size 1
   
   # CPU 사용
   --device cpu
   ```

2. **동적 축 오류**
   ```bash
   # 동적 축 비활성화
   # --dynamic 옵션 제거
   ```

3. **검증 실패**
   ```bash
   # 허용 오차 조정
   # 코드에서 rtol 값 증가 (1e-3 → 1e-2)
   ```

### 로그 확인

```bash
# 상세 로그 출력
./tools/convert_stgcn_to_onnx.sh realtime --verbose

# 특정 단계에서 멈춤
python3 -c "import torch; print(torch.__version__)"
python3 -c "import onnx; print(onnx.__version__)"
```

## 📁 출력 파일

변환된 ONNX 모델은 다음 위치에 저장됩니다:

```
checkpoints/
├── stgcn_fight_realtime.onnx    # 실시간 추론용
├── stgcn_fight_batch.onnx       # 배치 처리용  
├── stgcn_fight_dev.onnx         # 개발용
└── my_custom_model.onnx         # 사용자 정의
```

## 🔗 추가 정보

- [ONNX 공식 문서](https://onnx.ai/onnx/)
- [ONNX Runtime 문서](https://onnxruntime.ai/)
- [MMAction2 공식 문서](https://mmaction2.readthedocs.io/)

---

## 💡 팁

1. **실시간 추론**: `realtime` 프리셋 사용 (동적 프레임)
2. **배치 처리**: `batch` 프리셋 사용 (정적 크기, 높은 처리량)
3. **개발/디버깅**: `development` 프리셋 사용 (최적화 비활성화)
4. **성능 측정**: `--benchmark` 옵션으로 변환 전후 성능 비교
5. **검증**: `--verify` 옵션으로 정확성 확인 (기본 활성화)