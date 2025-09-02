#!/usr/bin/env python3
"""
성능평가 및 차트 생성 스크립트
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
    print("=== 성능평가 및 차트 생성 시작 ===")
    
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # AnalysisMode 인스턴스 생성
    mode = AnalysisMode(config)
    
    # 성능평가 실행
    input_path = '/aivanas/raw/surveillance/action/violence/action_recognition/data/UBI_demo'
    output_dir = 'output'
    
    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    
    success = mode._run_performance_evaluation(input_path, output_dir)
    print(f'Performance evaluation success: {success}')
    
    # 결과 확인
    eval_dir = Path(output_dir) / 'UBI_demo' / 'evaluation'
    print(f"\n=== 평가 결과 파일들 ===")
    if eval_dir.exists():
        for file in eval_dir.iterdir():
            print(f"- {file.name} ({file.stat().st_size} bytes)")
    else:
        print("평가 디렉토리가 존재하지 않습니다.")

if __name__ == "__main__":
    main()