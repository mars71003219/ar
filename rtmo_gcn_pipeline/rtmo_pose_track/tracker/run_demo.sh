#!/bin/bash

# Enhanced ByteTracker with RTMO Demo Run Script

# 프로젝트 루트 디렉토리로 이동
cd /workspace/rtmo_gcn_pipeline/rtmo_pose_track/tracker

# 출력 디렉토리 생성
mkdir -p ./output

echo "======================================="
echo "Enhanced ByteTracker with RTMO Demo"
echo "======================================="

# Python 경로 설정
export PYTHONPATH="/workspace:$PYTHONPATH"

# CUDA 설정 확인
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

# 메모리 정리
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

echo ""
echo "Starting demo with test videos..."
echo "Output directory: ./output"
echo ""

# 데모 실행
python3 demo_main.py \
    --output-dir ./output \
    --config-mode balanced \
    --device cuda:0 \
    --max-frames 300 \
    --save \
    2>&1 | tee ./output/demo_log.txt

echo ""
echo "Demo completed! Check ./output directory for results."
echo "Log saved to: ./output/demo_log.txt"