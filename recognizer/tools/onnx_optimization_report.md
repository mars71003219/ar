# ONNX GPU 최적화 벤치마크 보고서

## 시스템 정보

- **GPU**: NVIDIA RTX A5000
- **GPU 메모리**: 23.7 GB
- **CUDA**: 12.1
- **cuDNN**: 8902
- **ONNXRuntime**: 1.18.0

## 🏆 최적 설정

**성능**: 13.12ms (76.2 FPS)

```yaml
# config.yaml에 적용할 설정
models:
  pose_estimation:
    onnx:
      cudnn_conv_algo_search: "HEURISTIC"
      do_copy_in_default_stream: False
      cudnn_conv_use_max_workspace: True
      tunable_op_enable: True
      tunable_op_tuning_enable: True
      gpu_mem_limit_gb: 8
      arena_extend_strategy: "kNextPowerOfTwo"
      execution_mode: "ORT_SEQUENTIAL"
      graph_optimization_level: "ORT_ENABLE_ALL"
      enable_cpu_mem_arena: True
      enable_mem_pattern: True
```

## 📊 성능 비교

| 순위 | 알고리즘 | FPS | 시간(ms) | GPU메모리(GB) | 스트림 | 작업공간 |
|------|----------|-----|----------|---------------|--------|----------|
| 1 | HEURISTIC | 75.6 | 13.22 | 4 | False | True |
| 2 | HEURISTIC | 76.2 | 13.12 | 8 | False | True |
| 3 | HEURISTIC | 75.8 | 13.19 | 16 | False | True |
| 4 | HEURISTIC | 76.0 | 13.16 | 21 | False | True |
| 5 | EXHAUSTIVE | 41.3 | 24.21 | 4 | False | True |
| 6 | EXHAUSTIVE | 45.2 | 22.13 | 8 | False | True |
| 7 | EXHAUSTIVE | 44.5 | 22.48 | 16 | False | True |
| 8 | EXHAUSTIVE | 40.6 | 24.66 | 21 | False | True |
| 9 | DEFAULT | 66.3 | 15.09 | 4 | True | False |
| 10 | DEFAULT | 66.1 | 15.12 | 8 | True | False |

## 🔍 설정별 분석

- **HEURISTIC**: 평균 73.6 FPS, 최고 76.2 FPS
- **EXHAUSTIVE**: 평균 42.9 FPS, 최고 45.2 FPS
- **DEFAULT**: 평균 66.1 FPS, 최고 66.3 FPS

## 💡 권장사항

✅ **HEURISTIC 알고리즘 권장**
- 빠른 초기화와 우수한 성능의 균형
- 실시간 추론에 최적화
- 프로덕션 환경에 권장

**메모리 설정**: 8GB 할당
**스트림 모드**: 별도 스트림 사용