#!/bin/bash

# UCF Crime Simple Test - Direct source processing
# 기존 방식: results/overlays/ 에 영상, results/ 에 json 파일

set -e

echo "=== UCF Crime 단순 테스트 ==="

# 경로 설정
FIGHT_DIR="/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test/Fight"
NORMAL_DIR="/aivanas/raw/surveillance/action/violence/action_recognition/data/UCF_Crime_test/Normal"
OUTPUT_DIR="/workspace/rtmo_gcn_pipeline/inference_pipeline/results"
ANNOTATIONS_FILE="./ucf_crime_test_annotations.txt"
LABEL_MAP_FILE="./ucf_crime_label_map.txt"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/overlays"

echo "데이터셋 정보:"
echo "- Fight 비디오: $(ls "$FIGHT_DIR"/*.mp4 | wc -l)개"
echo "- Normal 비디오: $(ls "$NORMAL_DIR"/*.mp4 | wc -l)개"
echo ""

echo "Fight 비디오 처리 중..."
python run_inference.py \
    --mode batch \
    --input "$FIGHT_DIR" \
    --annotations "$ANNOTATIONS_FILE" \
    --label-map "$LABEL_MAP_FILE" \
    --output "$OUTPUT_DIR" \
    --batch-size 6 \
    --generate-overlay \
    --verbose

echo ""
echo "Normal 비디오 처리 중..."
python run_inference.py \
    --mode batch \
    --input "$NORMAL_DIR" \
    --annotations "$ANNOTATIONS_FILE" \
    --label-map "$LABEL_MAP_FILE" \
    --output "$OUTPUT_DIR" \
    --batch-size 6 \
    --generate-overlay \
    --verbose

echo ""
echo "=== UCF Crime 테스트 완료 ==="
echo "결과 위치:"
echo "- 오버레이 비디오: $OUTPUT_DIR/overlays/"
echo "- JSON 결과: $OUTPUT_DIR/batch_results.json"
echo ""

# 결과 확인
if [ -f "$OUTPUT_DIR/batch_results.json" ]; then
    echo "배치 결과 파일 확인됨!"
    echo ""
    python -c "
import json
with open('$OUTPUT_DIR/batch_results.json', 'r') as f:
    data = json.load(f)
    if 'performance_metrics' in data and data['performance_metrics']:
        metrics = data['performance_metrics']['metrics']
        print(f'정확도: {metrics[\"accuracy\"]:.3f}')
        print(f'정밀도: {metrics[\"precision\"]:.3f}')
        print(f'재현율: {metrics[\"recall\"]:.3f}')
        print(f'F1-점수: {metrics[\"f1_score\"]:.3f}')
        print(f'처리된 비디오 수: {len(data.get(\"results\", []))}개')
    else:
        print('성능 메트릭을 찾을 수 없습니다.')
    "
else
    echo "배치 결과 파일이 생성되지 않았습니다."
fi