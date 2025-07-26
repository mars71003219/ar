#!/bin/bash

# Optimized STGCN++ Violence Detection Pipeline - Execution Script
# 최적화된 싸움 분류 파이프라인 실행 스크립트

set -e  # 오류 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로고 출력
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Optimized STGCN++ Violence Detection Pipeline      ║"
echo "║                최적화된 싸움 분류 파이프라인                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 기본 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 함수 정의
show_help() {
    echo -e "${YELLOW}사용법:${NC}"
    echo "  $0 [MODE] [OPTIONS]"
    echo ""
    echo -e "${YELLOW}모드:${NC}"
    echo "  single     - 단일 비디오 테스트"
    echo "  batch      - 배치 비디오 처리"
    echo "  benchmark  - 벤치마크 테스트 (정확도 평가)"
    echo "  config     - 설정 파일 검증"
    echo "  compare    - 성능 비교 분석"
    echo ""
    echo -e "${YELLOW}옵션:${NC}"
    echo "  --video PATH        - 단일 비디오 파일 경로"
    echo "  --input PATH        - 입력 디렉토리 경로"
    echo "  --output PATH       - 출력 디렉토리 경로"
    echo "  --max-videos N      - 최대 처리 비디오 수 (기본: 10)"
    echo "  --device DEVICE     - 추론 디바이스 (기본: cuda:0)"
    echo "  --workers N         - 병렬 워커 수 (기본: 4)"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo "  $0 single --video /path/to/video.mp4"
    echo "  $0 batch --input /path/to/videos/ --max-videos 20"
    echo "  $0 benchmark --input /data/test_videos/"
    echo "  $0 config"
    echo "  $0 compare"
}

check_dependencies() {
    echo -e "${BLUE}🔧 의존성 확인 중...${NC}"
    
    # Python 확인
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Error: Python이 설치되지 않았습니다.${NC}"
        exit 1
    fi
    
    # 필요한 Python 패키지 확인
    python -c "
import sys
required_packages = ['torch', 'mmpose', 'mmaction', 'cv2', 'numpy']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'❌ 누락된 패키지: {missing}')
    sys.exit(1)
else:
    print('✅ 모든 필요한 패키지가 설치되어 있습니다.')
"
    
    # GPU 확인
    if python -c "import torch; print('✅ CUDA 사용 가능:' if torch.cuda.is_available() else '⚠️ CUDA 사용 불가능 (CPU 모드)')" 2>/dev/null; then
        :
    else
        echo -e "${YELLOW}⚠️ GPU 상태를 확인할 수 없습니다.${NC}"
    fi
}

validate_config() {
    echo -e "${BLUE}🔧 설정 파일 검증 중...${NC}"
    if python pipeline_config.py; then
        echo -e "${GREEN}✅ 설정 검증 완료${NC}"
        return 0
    else
        echo -e "${RED}❌ 설정 검증 실패${NC}"
        return 1
    fi
}

run_single_test() {
    local video_path="$1"
    local output_dir="${2:-./results}"
    
    if [[ -z "$video_path" ]]; then
        echo -e "${RED}❌ Error: 비디오 파일 경로가 필요합니다.${NC}"
        echo "사용법: $0 single --video /path/to/video.mp4"
        exit 1
    fi
    
    if [[ ! -f "$video_path" ]]; then
        echo -e "${RED}❌ Error: 비디오 파일을 찾을 수 없습니다: $video_path${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎬 단일 비디오 테스트 시작: $(basename "$video_path")${NC}"
    python test_optimized_pipeline.py --mode single --video "$video_path" --output "$output_dir"
}

run_batch_test() {
    local input_dir="$1"
    local output_dir="${2:-./results}"
    local max_videos="${3:-10}"
    
    if [[ -z "$input_dir" ]]; then
        echo -e "${RED}❌ Error: 입력 디렉토리 경로가 필요합니다.${NC}"
        echo "사용법: $0 batch --input /path/to/videos/"
        exit 1
    fi
    
    if [[ ! -d "$input_dir" ]]; then
        echo -e "${RED}❌ Error: 입력 디렉토리를 찾을 수 없습니다: $input_dir${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}📁 배치 비디오 테스트 시작: $input_dir (최대 $max_videos개)${NC}"
    python test_optimized_pipeline.py --mode batch --input "$input_dir" --output "$output_dir" --max-videos "$max_videos"
}

run_benchmark() {
    local input_dir="$1"
    local output_dir="${2:-./results}"
    
    if [[ -z "$input_dir" ]]; then
        echo -e "${RED}❌ Error: 테스트 데이터 디렉토리 경로가 필요합니다.${NC}"
        echo "사용법: $0 benchmark --input /path/to/test_data/"
        exit 1
    fi
    
    if [[ ! -d "$input_dir" ]]; then
        echo -e "${RED}❌ Error: 테스트 디렉토리를 찾을 수 없습니다: $input_dir${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🏁 벤치마크 테스트 시작: $input_dir${NC}"
    python test_optimized_pipeline.py --mode benchmark --input "$input_dir" --output "$output_dir"
}

run_comparison() {
    echo -e "${GREEN}📊 성능 비교 분석 시작${NC}"
    python performance_comparison.py
}

# 기본값 설정
MODE=""
VIDEO_PATH=""
INPUT_DIR=""
OUTPUT_DIR="./results"
MAX_VIDEOS="10"
DEVICE="cuda:0"
WORKERS="4"

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        single|batch|benchmark|config|compare)
            MODE="$1"
            shift
            ;;
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-videos)
            MAX_VIDEOS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 알 수 없는 옵션: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 모드가 지정되지 않은 경우
if [[ -z "$MODE" ]]; then
    echo -e "${RED}❌ Error: 실행 모드를 지정해주세요.${NC}"
    show_help
    exit 1
fi

# 의존성 확인
check_dependencies

# 모드별 실행
case "$MODE" in
    config)
        validate_config
        ;;
    single)
        if validate_config; then
            run_single_test "$VIDEO_PATH" "$OUTPUT_DIR"
        else
            exit 1
        fi
        ;;
    batch)
        if validate_config; then
            run_batch_test "$INPUT_DIR" "$OUTPUT_DIR" "$MAX_VIDEOS"
        else
            exit 1
        fi
        ;;
    benchmark)
        if validate_config; then
            run_benchmark "$INPUT_DIR" "$OUTPUT_DIR"
        else
            exit 1
        fi
        ;;
    compare)
        run_comparison
        ;;
    *)
        echo -e "${RED}❌ 지원하지 않는 모드: $MODE${NC}"
        show_help
        exit 1
        ;;
esac

echo -e "${GREEN}✅ 실행 완료!${NC}"