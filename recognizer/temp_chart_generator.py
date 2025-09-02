import json
import sys
import os
sys.path.append('/workspace/recognizer')
from utils.evaluation_visualizer import EvaluationVisualizer

# 데이터 로드
with open('output/UBI_demo/evaluation/summary_results.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# CSV 데이터 파싱
summary_data = []
for line in lines[1:]:  # 헤더 제외
    parts = line.strip().split(',')
    if len(parts) >= 6:
        video_name = parts[0].strip('"')
        class_label = parts[4].strip('"')
        prediction = parts[5].strip('"')
        confusion_type = parts[6].strip('"')
        
        summary_data.append({
            'video_filename': video_name,
            'class_label': class_label,
            'prediction': prediction,
            'confusion_matrix_type': confusion_type,
            'avg_confidence': 0.7 if prediction == 'Fight' else 0.3
        })

# detailed 데이터 로드
detailed_data = []
with open('output/UBI_demo/evaluation/detailed_results.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines[1:]:  # 헤더 제외
    parts = line.strip().split(',')
    if len(parts) >= 6:
        video_name = parts[1].strip('"')
        fight_score = float(parts[5])
        
        detailed_data.append({
            'video_filename': video_name,
            'fight_score': fight_score
        })

# metrics 로드
with open('output/UBI_demo/evaluation/performance_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f'Summary data: {len(summary_data)} videos')
print(f'Detailed data: {len(detailed_data)} windows')
print(f'Metrics: {list(metrics.keys())}')

# 시각화 도구 초기화 및 모든 차트 생성
visualizer = EvaluationVisualizer('output/evaluation')
success = visualizer.generate_all_charts(summary_data, detailed_data, metrics)
print(f'Charts generation success: {success}')
