#!/bin/bash
# STGCN Fight Detection Model ONNX Conversion Script
# 다양한 변환 옵션을 제공하는 스크립트

set -e  # 오류 발생 시 스크립트 종료

# 색상 출력 함수
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$BASE_DIR/mmaction2/configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection_stable.py"
CHECKPOINT_PATH="$BASE_DIR/mmaction2/work_dirs/stgcnpp-bone-ntu60_rtmo-m_RWF2000plus_stable/best_acc_top1_epoch_14.pth"
OUTPUT_DIR="$BASE_DIR/checkpoints"
CONVERTER_SCRIPT="$SCRIPT_DIR/pytorch_to_onnx_converter.py"

# 도움말 출력
show_help() {
    cat << EOF
STGCN Fight Detection Model ONNX Conversion Script

사용법:
    $0 [옵션] [프리셋]

프리셋:
    realtime        - 실시간 추론용 (동적 프레임, 배치=1)
    batch           - 배치 처리용 (정적 크기, 배치=8)  
    development     - 개발용 (정적 크기, 최적화 비활성화)
    custom          - 사용자 정의 설정

옵션:
    -h, --help      - 도움말 출력
    -c, --config    - 설정 파일 경로 (기본: auto-detect)
    -k, --checkpoint - 체크포인트 파일 경로 (기본: auto-detect)
    -o, --output    - 출력 파일 경로 (기본: ./checkpoints/)
    -d, --device    - 디바이스 (기본: cuda:0)
    -v, --verbose   - 상세 출력
    --no-verify     - 검증 생략
    --no-benchmark  - 벤치마크 생략

예시:
    $0 realtime                     # 실시간 추론용 변환
    $0 batch -o my_model.onnx       # 배치 처리용 변환
    $0 custom --dynamic --verify    # 사용자 정의 동적 변환

EOF
}

# 파일 존재 확인
check_file_exists() {
    if [[ ! -f "$1" ]]; then
        print_error "파일을 찾을 수 없습니다: $1"
        exit 1
    fi
}

# Python 환경 확인
check_python_env() {
    print_info "Python 환경 확인 중..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3가 설치되어 있지 않습니다"
        exit 1
    fi
    
    # 필요한 패키지 확인
    required_packages=("torch" "onnx" "onnxruntime" "mmaction2")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            print_error "필요한 패키지가 설치되어 있지 않습니다: $package"
            exit 1
        fi
    done
    
    print_success "Python 환경 확인 완료"
}

# 실시간 추론용 변환
convert_realtime() {
    print_info "실시간 추론용 모델 변환 시작..."
    
    local output_file="${OUTPUT_DIR}/stgcn_fight_realtime.onnx"
    mkdir -p "$OUTPUT_DIR"
    
    python3 "$CONVERTER_SCRIPT" \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$output_file" \
        --batch-size 1 \
        --num-frames 100 \
        --num-persons 4 \
        --dynamic \
        --dynamic-frames \
        --verify \
        --benchmark \
        --device "${DEVICE:-cuda:0}" \
        ${VERBOSE:+--verbose}
    
    print_success "실시간 추론용 모델 변환 완료: $output_file"
}

# 배치 처리용 변환
convert_batch() {
    print_info "배치 처리용 모델 변환 시작..."
    
    local output_file="${OUTPUT_DIR}/stgcn_fight_batch.onnx"
    mkdir -p "$OUTPUT_DIR"
    
    python3 "$CONVERTER_SCRIPT" \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$output_file" \
        --batch-size 8 \
        --num-frames 100 \
        --num-persons 4 \
        --verify \
        --benchmark \
        --device "${DEVICE:-cuda:0}" \
        ${VERBOSE:+--verbose}
    
    print_success "배치 처리용 모델 변환 완료: $output_file"
}

# 개발용 변환
convert_development() {
    print_info "개발용 모델 변환 시작..."
    
    local output_file="${OUTPUT_DIR}/stgcn_fight_dev.onnx"
    mkdir -p "$OUTPUT_DIR"
    
    python3 "$CONVERTER_SCRIPT" \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$output_file" \
        --batch-size 1 \
        --num-frames 100 \
        --num-persons 4 \
        --no-optimize \
        --verify \
        --device "${DEVICE:-cuda:0}" \
        --verbose
    
    print_success "개발용 모델 변환 완료: $output_file"
}

# 사용자 정의 변환
convert_custom() {
    print_info "사용자 정의 모델 변환 시작..."
    
    local output_file="${OUTPUT_FILE:-${OUTPUT_DIR}/stgcn_fight_custom.onnx}"
    mkdir -p "$(dirname "$output_file")"
    
    local cmd_args=(
        --config "$CONFIG_PATH"
        --checkpoint "$CHECKPOINT_PATH"
        --output "$output_file"
        --device "${DEVICE:-cuda:0}"
    )
    
    # 추가 인자 처리
    [[ $DYNAMIC == "true" ]] && cmd_args+=(--dynamic)
    [[ $VERIFY != "false" ]] && cmd_args+=(--verify)
    [[ $BENCHMARK == "true" ]] && cmd_args+=(--benchmark)
    [[ $VERBOSE == "true" ]] && cmd_args+=(--verbose)
    
    python3 "$CONVERTER_SCRIPT" "${cmd_args[@]}"
    
    print_success "사용자 정의 모델 변환 완료: $output_file"
}

# 메인 실행 부분
main() {
    # 기본값 설정
    PRESET=""
    DEVICE="cuda:0"
    VERIFY="true"
    BENCHMARK="false"
    DYNAMIC="false"
    VERBOSE="false"
    
    # 인자 파싱
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            -k|--checkpoint)
                CHECKPOINT_PATH="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            --no-verify)
                VERIFY="false"
                shift
                ;;
            --no-benchmark)
                BENCHMARK="false"
                shift
                ;;
            --benchmark)
                BENCHMARK="true"
                shift
                ;;
            --dynamic)
                DYNAMIC="true"
                shift
                ;;
            realtime|batch|development|custom)
                PRESET="$1"
                shift
                ;;
            *)
                print_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 프리셋이 지정되지 않은 경우 기본값 설정
    if [[ -z "$PRESET" ]]; then
        PRESET="realtime"
        print_warning "프리셋이 지정되지 않아 'realtime'을 사용합니다"
    fi
    
    print_info "=== STGCN Fight Detection Model ONNX Conversion ==="
    print_info "프리셋: $PRESET"
    print_info "디바이스: $DEVICE"
    print_info "설정 파일: $CONFIG_PATH"
    print_info "체크포인트: $CHECKPOINT_PATH"
    
    # 파일 존재 확인
    check_file_exists "$CONFIG_PATH"
    check_file_exists "$CHECKPOINT_PATH" 
    check_file_exists "$CONVERTER_SCRIPT"
    
    # Python 환경 확인
    check_python_env
    
    # 프리셋에 따른 변환 실행
    case $PRESET in
        realtime)
            convert_realtime
            ;;
        batch)
            convert_batch
            ;;
        development)
            convert_development
            ;;
        custom)
            convert_custom
            ;;
        *)
            print_error "지원하지 않는 프리셋: $PRESET"
            exit 1
            ;;
    esac
    
    print_success "=== 변환 작업 완료 ==="
}

# 스크립트 실행
main "$@"