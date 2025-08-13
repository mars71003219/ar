#!/usr/bin/env python3
"""
Video Results Extractor
window_results.json에서 특정 비디오의 데이터만 추출

사용법:
  python extract_video_results.py --input /path/to/window_results.json --video-name cam04_04 --output /path/to/cam04_04_analysis.json
  python extract_video_results.py --input /path/to/window_results.json --video-name cam04_04 --output /path/to/cam04_04_analysis.json --pretty
  python extract_video_results.py --input /path/to/window_results.json --video-name cam04_04 --summary-only
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

def convert_numpy_to_serializable(obj: Any) -> Any:
    """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        return [convert_numpy_to_serializable(i) for i in obj]
    return obj

def extract_video_data(data: List[Dict], video_name: str) -> List[Dict]:
    """특정 비디오 이름의 데이터만 추출"""
    return [item for item in data if item.get('video_name') == video_name]

def analyze_video_data(video_data: List[Dict]) -> Dict[str, Any]:
    """추출된 비디오 데이터 분석"""
    if not video_data:
        return {'message': 'No data found for the specified video'}
    
    analysis = {
        'video_name': video_data[0].get('video_name', 'Unknown'),
        'total_windows': len(video_data),
        'true_label': video_data[0].get('true_label', 'Unknown'),
        'label_folder': video_data[0].get('label_folder', 'Unknown'),
        'windows_range': {
            'start_frame': min(item.get('start_frame', 0) for item in video_data),
            'end_frame': max(item.get('end_frame', 0) for item in video_data),
            'window_indices': [item.get('window_idx', 0) for item in video_data]
        },
        'predictions': {
            'scores': [item.get('prediction', 0) for item in video_data],
            'labels': [item.get('predicted_label', 0) for item in video_data],
            'fight_windows': sum(1 for item in video_data if item.get('predicted_label') == 1),
            'normal_windows': sum(1 for item in video_data if item.get('predicted_label') == 0),
            'avg_score': np.mean([item.get('prediction', 0) for item in video_data]),
            'max_score': np.max([item.get('prediction', 0) for item in video_data]),
            'min_score': np.min([item.get('prediction', 0) for item in video_data])
        },
        'persons_count': [item.get('persons_count', 0) for item in video_data]
    }
    
    return analysis

def extract_summary_data(video_data: List[Dict]) -> List[Dict]:
    """pose_data 제외한 요약 데이터 추출"""
    summary_data = []
    for item in video_data:
        summary_item = {k: v for k, v in item.items() if k != 'pose_data'}
        summary_data.append(summary_item)
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="Extract specific video data from window_results.json")
    parser.add_argument('--input', type=str, required=True,
                       help='Input window_results.json file path')
    parser.add_argument('--video-name', type=str, required=True,
                       help='Video name to extract (e.g., cam04_04)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: auto-generated)')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print JSON with indentation')
    parser.add_argument('--summary-only', action='store_true',
                       help='Extract summary only (exclude pose_data)')
    parser.add_argument('--analyze', action='store_true',
                       help='Include analysis and statistics')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    # 출력 파일명 자동 생성
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        suffix = "_summary" if args.summary_only else ""
        args.output = os.path.join(input_dir, f"{args.video_name}_windows{suffix}.json")
    
    print(f"Loading JSON file: {args.input}")
    
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        print(f"전체 윈도우 수: {len(all_data)}")
        
        # 특정 비디오 데이터 추출
        video_data = extract_video_data(all_data, args.video_name)
        print(f"{args.video_name} 윈도우 수: {len(video_data)}")
        
        if not video_data:
            print(f"비디오 '{args.video_name}'에 대한 데이터를 찾을 수 없습니다.")
            
            # 사용 가능한 비디오 이름들 표시
            available_videos = list(set(item.get('video_name', 'Unknown') for item in all_data))
            print(f"사용 가능한 비디오 이름들: {available_videos[:10]}...")
            return
        
        # 데이터 처리
        if args.summary_only:
            extracted_data = extract_summary_data(video_data)
        else:
            extracted_data = convert_numpy_to_serializable(video_data)
        
        # 분석 정보 추가
        output_data = {
            'metadata': {
                'source_file': args.input,
                'video_name': args.video_name,
                'extraction_time': str(np.datetime64('now')),
                'total_windows': len(video_data),
                'summary_only': args.summary_only
            },
            'data': extracted_data
        }
        
        # 분석 정보 추가
        if args.analyze:
            output_data['analysis'] = analyze_video_data(video_data)
            print("분석 정보 추가됨")
        
        # JSON 저장
        print(f"결과 저장 중: {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        
        print(f"추출 완료: {args.output}")
        print(f"파일 크기: {os.path.getsize(args.output) / 1024:.2f} KB")
        
        # 간단한 요약 정보 출력
        if video_data:
            sample = video_data[0]
            print(f"\n추출된 데이터 요약:")
            print(f"  비디오: {sample.get('video_name', 'Unknown')}")
            print(f"  레이블: {'Fight' if sample.get('true_label') == 1 else 'Normal'}")
            print(f"  윈도우 수: {len(video_data)}")
            if args.analyze and 'analysis' in output_data:
                pred_info = output_data['analysis']['predictions']
                print(f"  Fight 윈도우: {pred_info['fight_windows']}")
                print(f"  평균 점수: {pred_info['avg_score']:.4f}")
                print(f"  최고 점수: {pred_info['max_score']:.4f}")
        
    except Exception as e:
        print(f"추출 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()