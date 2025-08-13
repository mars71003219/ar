#!/usr/bin/env python3
"""
PKL to JSON Converter
윈도우 결과 pkl 파일을 분석 가능한 JSON 형태로 변환

사용법:
  python pkl_to_json_converter.py --input /path/to/file.pkl --output /path/to/output.json
  python pkl_to_json_converter.py --input /path/to/file.pkl --output /path/to/output.json --pretty
  python pkl_to_json_converter.py --input /path/to/file.pkl --output /path/to/output.json --summary-only
"""

import pickle
import json
import numpy as np
import argparse
import os
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

def analyze_pkl_structure(data: Any) -> Dict[str, Any]:
    """PKL 파일 구조 분석"""
    analysis = {
        'data_type': str(type(data).__name__),
        'total_items': 0,
        'structure_info': {}
    }
    
    if isinstance(data, list):
        analysis['total_items'] = len(data)
        if len(data) > 0:
            first_item = data[0]
            analysis['structure_info'] = {
                'item_type': str(type(first_item).__name__),
                'sample_keys': list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dictionary'
            }
            
            # 각 키의 데이터 타입 분석
            if isinstance(first_item, dict):
                key_types = {}
                for key, value in first_item.items():
                    key_types[key] = str(type(value).__name__)
                    if isinstance(value, (list, np.ndarray)):
                        if hasattr(value, 'shape'):
                            key_types[key] += f" shape: {value.shape}"
                        elif hasattr(value, '__len__'):
                            key_types[key] += f" length: {len(value)}"
                analysis['structure_info']['key_types'] = key_types
    
    elif isinstance(data, dict):
        analysis['structure_info'] = {
            'keys': list(data.keys()),
            'key_types': {k: str(type(v).__name__) for k, v in data.items()}
        }
    
    return analysis

def extract_summary_only(data: Any) -> Dict[str, Any]:
    """요약 정보만 추출 (pose_data 제외)"""
    if isinstance(data, list):
        summary_data = []
        for item in data:
            if isinstance(item, dict):
                summary_item = {k: v for k, v in item.items() if k != 'pose_data'}
                summary_data.append(summary_item)
            else:
                summary_data.append(item)
        return summary_data
    return data

def extract_pose_data_summary(pose_data: Any) -> Dict[str, Any]:
    """pose_data의 요약 정보 추출"""
    if not isinstance(pose_data, dict):
        return {'type': str(type(pose_data).__name__), 'content': str(pose_data)[:200]}
    
    summary = {}
    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                'type': 'ndarray',
                'shape': value.shape,
                'dtype': str(value.dtype),
                'sample': value.flatten()[:10].tolist() if value.size > 0 else []
            }
        elif isinstance(value, dict):
            summary[key] = {
                'type': 'dict',
                'keys': list(value.keys()),
                'summary': {k: str(type(v).__name__) for k, v in value.items()}
            }
        elif isinstance(value, list):
            summary[key] = {
                'type': 'list',
                'length': len(value),
                'item_types': [str(type(item).__name__) for item in value[:5]]
            }
        else:
            summary[key] = {
                'type': str(type(value).__name__),
                'value': value if len(str(value)) < 100 else str(value)[:100] + '...'
            }
    
    return summary

def convert_pkl_to_json(input_file: str, output_file: str, pretty: bool = False, summary_only: bool = False):
    """PKL 파일을 JSON으로 변환"""
    print(f"Loading PKL file: {input_file}")
    
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"PKL 파일 로드 완료. 데이터 타입: {type(data)}")
        
        # 구조 분석
        structure_analysis = analyze_pkl_structure(data)
        print(f"구조 분석 완료:")
        print(f"  - 데이터 타입: {structure_analysis['data_type']}")
        print(f"  - 총 항목 수: {structure_analysis['total_items']}")
        
        # 데이터 변환
        if summary_only:
            print("요약 모드: pose_data 제외하고 변환")
            converted_data = extract_summary_only(data)
        else:
            print("전체 데이터 변환 중...")
            converted_data = convert_numpy_to_serializable(data)
            
            # pose_data가 있는 경우 요약도 추가
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'pose_data' in data[0]:
                print("pose_data 요약 정보 추가")
                for i, item in enumerate(converted_data):
                    if 'pose_data' in item:
                        item['pose_data_summary'] = extract_pose_data_summary(data[i]['pose_data'])
        
        # 메타데이터 추가
        output_data = {
            'metadata': {
                'source_file': input_file,
                'conversion_time': str(np.datetime64('now')),
                'data_structure': structure_analysis,
                'summary_only': summary_only
            },
            'data': converted_data
        }
        
        # JSON 저장
        print(f"JSON 파일 저장 중: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        
        print(f"변환 완료: {output_file}")
        print(f"파일 크기: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PKL files to JSON format")
    parser.add_argument('--input', type=str, required=True,
                       help='Input PKL file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty print JSON with indentation')
    parser.add_argument('--summary-only', action='store_true',
                       help='Extract summary only (exclude pose_data)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze structure without conversion')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    if args.analyze_only:
        print("구조 분석 모드")
        try:
            with open(args.input, 'rb') as f:
                data = pickle.load(f)
            
            analysis = analyze_pkl_structure(data)
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"분석 실패: {e}")
        return
    
    success = convert_pkl_to_json(args.input, args.output, args.pretty, args.summary_only)
    if success:
        print("변환이 성공적으로 완료되었습니다.")
    else:
        print("변환에 실패했습니다.")

if __name__ == "__main__":
    main()