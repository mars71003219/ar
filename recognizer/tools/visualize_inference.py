#!/usr/bin/env python3
"""
Inference 결과 시각화 도구

실시간 추론 결과를 원본 비디오에 오버레이하여 시각화합니다.

사용법:
    python tools/visualize_inference.py --input video.mp4 --results results.json --output visualization.mp4
    
    또는 main.py 결과 디렉토리에서:
    python tools/visualize_inference.py --results-dir output/
"""

import argparse
import json
import pickle
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent
sys.path.insert(0, str(recognizer_root))

from visualization import create_inference_visualization


def load_classification_results(results_path: str) -> List[Dict[str, Any]]:
    """분류 결과 파일 로드"""
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # results.json 형태인 경우
        if 'classification_results' in data:
            return data['classification_results']
        
        # 직접 분류 결과 리스트인 경우
        if isinstance(data, list):
            return data
        
        logging.error(f"Unknown results format in {results_path}")
        return []
        
    except Exception as e:
        logging.error(f"Failed to load results from {results_path}: {str(e)}")
        return []


def load_frame_poses_results(poses_path: str) -> Optional[List]:
    """포즈 결과 파일 로드"""
    try:
        if not Path(poses_path).exists():
            return None
            
        with open(poses_path, 'rb') as f:
            frame_poses_results = pickle.load(f)
        
        logging.info(f"Loaded {len(frame_poses_results)} frame poses from {poses_path}")
        return frame_poses_results
        
    except Exception as e:
        logging.warning(f"Failed to load pose data from {poses_path}: {str(e)}")
        return None


def find_video_from_results(results_data: Dict[str, Any]) -> str:
    """결과 파일에서 원본 비디오 경로 추출"""
    if 'input_video' in results_data:
        return results_data['input_video']
    return ""


def main():
    parser = argparse.ArgumentParser(description="Inference 결과 시각화 도구")
    
    # 입력 옵션
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='입력 비디오 파일')
    group.add_argument('--results-dir', type=str, help='main.py 결과 디렉토리')
    
    parser.add_argument('--results', type=str, help='분류 결과 JSON 파일 (results-dir 사용시 자동 감지)')
    parser.add_argument('--output', type=str, help='출력 비디오 파일 (기본: input_visualization.mp4)')
    
    # 시각화 옵션
    parser.add_argument('--window-size', type=int, default=100, help='윈도우 크기 (기본: 100)')
    parser.add_argument('--stride', type=int, default=50, help='스트라이드 간격 (기본: 50)')
    
    # 로깅 옵션
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨 (기본: INFO)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 입력 파일들 결정
        if args.results_dir:
            # 결과 디렉토리에서 자동 탐지
            results_dir = Path(args.results_dir)
            if not results_dir.exists():
                logging.error(f"Results directory not found: {results_dir}")
                return False
            
            # results.json 파일 찾기
            results_file = results_dir / "results.json"
            if not results_file.exists():
                logging.error(f"Results file not found: {results_file}")
                return False
            
            # 결과 로드
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            classification_results = results_data.get('classification_results', [])
            input_video = find_video_from_results(results_data)
            
            if not input_video:
                logging.error("Input video path not found in results file")
                return False
            
            # 포즈 데이터 로드 시도
            poses_file = results_dir / "frame_poses.pkl"
            frame_poses_results = load_frame_poses_results(str(poses_file))
            
            # 출력 파일명 결정
            if not args.output:
                video_name = Path(input_video).stem
                output_video = results_dir / f"{video_name}_visualization.mp4"
            else:
                output_video = args.output
                
        else:
            # 직접 지정
            input_video = args.input
            
            if not args.results:
                logging.error("--results must be specified when using --input")
                return False
            
            classification_results = load_classification_results(args.results)
            
            # 포즈 데이터는 직접 지정 시 로드하지 않음 (결과 디렉토리만 지원)
            frame_poses_results = None
            
            # 출력 파일명 결정
            if not args.output:
                video_path = Path(input_video)
                output_video = video_path.parent / f"{video_path.stem}_visualization.mp4"
            else:
                output_video = args.output
        
        # 입력 검증
        if not Path(input_video).exists():
            logging.error(f"Input video not found: {input_video}")
            return False
        
        if not classification_results:
            logging.error("No classification results found")
            return False
        
        logging.info(f"Input video: {input_video}")
        logging.info(f"Classification results: {len(classification_results)} windows")
        if frame_poses_results:
            logging.info(f"Frame poses: {len(frame_poses_results)} frames")
        else:
            logging.info("No pose data available (only classification overlay)")
        logging.info(f"Output video: {output_video}")
        
        # 시각화 실행
        success = create_inference_visualization(
            input_video=input_video,
            classification_results=classification_results,
            output_path=str(output_video),
            frame_poses_results=frame_poses_results,
            window_size=args.window_size,
            stride=args.stride
        )
        
        if success:
            logging.info("Visualization completed successfully!")
            logging.info(f"Output saved to: {output_video}")
            return True
        else:
            logging.error("Visualization failed")
            return False
            
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)