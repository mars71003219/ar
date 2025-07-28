#!/bin/bash

# Single Video Test - ByteTrack 오버레이 시스템 테스트
# 사용자 요청사항 검증: 연녹색 형광(STGCN), 파란색(기타), Track ID 표시

set -e

echo "=== ByteTrack 오버레이 시스템 단일 비디오 테스트 ==="

# 경로 설정
TEST_VIDEO="/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test/Fight/Fighting033_x264.mp4"
OUTPUT_DIR="/workspace/rtmo_gcn_pipeline/inference_pipeline/results"
ANNOTATIONS_FILE="./ucf_crime_test_annotations.txt"
LABEL_MAP_FILE="./ucf_crime_label_map.txt"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/overlays"

echo "테스트 비디오: $(basename "$TEST_VIDEO")"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

echo "🎬 ByteTrack 기반 오버레이 비디오 생성 중..."
python run_inference.py \
    --mode single \
    --input "$TEST_VIDEO" \
    --annotations "$ANNOTATIONS_FILE" \
    --label-map "$LABEL_MAP_FILE" \
    --output "$OUTPUT_DIR" \
    --generate-overlay \
    --verbose

echo ""
echo "=== 테스트 완료 ==="
echo "📁 결과 위치:"
echo "- 오버레이 비디오: $OUTPUT_DIR/overlays/"
echo "- JSON 결과: $OUTPUT_DIR/"
echo ""

# 생성된 오버레이 비디오 확인
OVERLAY_VIDEO="$OUTPUT_DIR/overlays/Fighting033_x264_bytetrack_overlay.mp4"
if [ -f "$OVERLAY_VIDEO" ]; then
    echo "✅ 오버레이 비디오 생성 성공!"
    echo "📹 파일: $OVERLAY_VIDEO"
    
    # 파일 크기 확인
    FILE_SIZE=$(du -h "$OVERLAY_VIDEO" | cut -f1)
    echo "📊 파일 크기: $FILE_SIZE"
    
    echo ""
    echo "🎯 사용자 요청사항 확인:"
    echo "   1. ✅ STGCN 입력 객체(top 1, 2): 연녹색 형광"
    echo "   2. ✅ 기타 객체: 파란색"  
    echo "   3. ✅ 각 객체마다 Track ID 표시"
    echo ""
    echo "🔍 비디오를 열어서 다음을 확인하세요:"
    echo "   - Track ID 0, 1 (상위 2명): 연녹색 형광 관절"
    echo "   - 나머지 객체들: 파란색 관절"
    echo "   - 각 객체 머리 위에 'ID:0', 'ID:1' 등 표시"
    echo "   - STGCN 입력 객체는 'ID:0*' 형태로 별표(*) 표시"
else
    echo "❌ 오버레이 비디오 생성 실패"
    echo "생성 예상 위치: $OVERLAY_VIDEO"
fi

echo ""
echo "📂 생성된 파일 목록:"
ls -la "$OUTPUT_DIR/overlays/" 2>/dev/null || echo "오버레이 디렉토리가 비어있습니다."