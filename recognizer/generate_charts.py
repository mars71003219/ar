#!/usr/bin/env python3
"""
UBI_demo 성능평가 차트 생성 스크립트
"""

import sys
import os
import json
import yaml
from pathlib import Path

# 경로 설정
sys.path.append('/workspace/recognizer')
os.chdir('/workspace/recognizer')

from core.inference_modes import AnalysisMode

def main():
    print("=== UBI_demo 차트 및 보고서 생성 ===")
    
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # AnalysisMode 인스턴스 생성
    mode = AnalysisMode(config)
    
    # evaluation 결과 로드
    output_dir = Path('output/UBI_demo')
    eval_dir = output_dir / 'evaluation'
    
    if not eval_dir.exists():
        print("evaluation 디렉토리가 없습니다.")
        return
    
    # CSV 파일 확인
    summary_file = eval_dir / 'summary_results.csv'
    detailed_file = eval_dir / 'detailed_results.csv'
    
    if not summary_file.exists() or not detailed_file.exists():
        print("CSV 파일이 없습니다.")
        return
    
    print(f"Summary file: {summary_file}")
    print(f"Detailed file: {detailed_file}")
    
    # CSV 파일을 JSON 형태로 읽기
    import pandas as pd
    
    # Summary 결과 로드
    summary_df = pd.read_csv(summary_file)
    summary_results = summary_df.to_dict('records')
    
    # Detailed 결과 로드  
    detailed_df = pd.read_csv(detailed_file)
    detailed_results = detailed_df.to_dict('records')
    
    print(f"Summary results: {len(summary_results)} videos")
    print(f"Detailed results: {len(detailed_results)} windows")
    
    # 성능 지표 계산 및 차트 생성
    consecutive_frames = config.get('events', {}).get('min_consecutive_detections', 3)
    success = mode._calculate_and_visualize_metrics(summary_results, detailed_results, eval_dir, consecutive_frames)
    
    if success:
        print("차트 및 보고서 생성 완료!")
        
        # 생성된 파일들 확인
        charts_dir = eval_dir / 'charts'
        if charts_dir.exists():
            print(f"\n생성된 차트 파일들:")
            for chart_file in charts_dir.glob("*.png"):
                print(f"- {chart_file.name}")
    else:
        print("차트 생성 실패")

if __name__ == "__main__":
    main()