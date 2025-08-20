#!/bin/bash
# ONNX GPU 빠른 최적화 스크립트

echo "🚀 ONNX GPU 자동 최적화 도구"
echo "=============================="

# CUDA 환경 설정
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# 현재 GPU 정보 출력
echo "현재 GPU 환경:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "ONNX 최적화 실행 중..."

# 기본 경로 설정
MODEL_PATH="/workspace/mmpose/checkpoints/end2end.onnx"
CONFIG_PATH="/workspace/recognizer/configs/config.yaml"
REPORT_PATH="/workspace/recognizer/gpu_optimization_$(date +%Y%m%d_%H%M%S).md"

# 최적화 실행
python /workspace/recognizer/tools/onnx_optimizer.py \
    --model "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --output "$REPORT_PATH" \
    --warmup 15 \
    --runs 30

echo
echo "✅ 최적화 완료!"
echo "보고서: $REPORT_PATH"
echo "설정 파일이 자동으로 업데이트되었습니다."

echo
echo "🔧 적용된 최적화 확인:"
grep -A 10 "cudnn_conv_algo_search" "$CONFIG_PATH"

echo
echo "🚀 이제 최적화된 ONNX 추론을 사용할 수 있습니다!"