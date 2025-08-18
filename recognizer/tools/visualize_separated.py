#!/usr/bin/env python3
"""
Separated 파이프라인 결과 시각화 도구

separated 파이프라인 결과를 윈도우별 분할 영상으로 시각화합니다.

사용법:
    python tools/visualize_separated.py --input video.mp4 --annotations annotations.pkl --output-dir window_videos/
    
    또는 separated 파이프라인 결과에서:
    python tools/visualize_separated.py --results-dir separated_output/
"""

import argparse
import pickle
import logging
import sys
from pathlib import Path
from typing import List

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent
sys.path.insert(0, str(recognizer_root))

from visualization import create_separated_visualization
from utils.data_structure import WindowAnnotation


def load_window_annotations(annotations_path: str) -> List[WindowAnnotation]:
    """윈도우 어노테이션 파일 로드"""
    try:
        with open(annotations_path, 'rb') as f:
            annotations = pickle.load(f)
        
        # WindowAnnotation 객체들의 리스트인지 확인
        if isinstance(annotations, list) and all(isinstance(a, WindowAnnotation) for a in annotations):
            return annotations
        
        logging.error(f"Invalid annotation format in {annotations_path}")
        return []
        
    except Exception as e:
        logging.error(f"Failed to load annotations from {annotations_path}: {str(e)}")
        return []


def find_annotations_and_video_from_results(results_dir: Path) -> tuple:
    """결과 디렉토리에서 어노테이션과 비디오 파일 찾기"""
    # 일반적인 패턴들 시도
    annotation_patterns = [
        "window_annotations.pkl",
        "annotations.pkl", 
        "*_annotations.pkl",
        "*_windows.pkl"
    ]
    
    video_patterns = [
        "*.mp4",
        "*.avi", 
        "*.mov",
        "*.mkv"
    ]
    
    # 어노테이션 파일 찾기
    annotations_file = None
    for pattern in annotation_patterns:
        files = list(results_dir.glob(pattern))
        if files:
            annotations_file = files[0]
            break
    
    # 비디오 파일 찾기 (또는 경로 정보 파일에서)
    video_file = None
    
    # 먼저 info.json이나 config.json에서 원본 비디오 경로 찾기
    info_files = ["info.json", "config.json", "results.json"]
    for info_file in info_files:
        info_path = results_dir / info_file
        if info_path.exists():
            try:
                import json
                with open(info_path, 'r') as f:
                    data = json.load(f)
                
                # 가능한 키들
                video_keys = ["input_video", "video_path", "source", "input"]
                for key in video_keys:
                    if key in data and data[key]:
                        video_file = data[key]
                        break
                
                if video_file:
                    break
                    
            except Exception:
                continue
    
    # 정보 파일에서 찾지 못했으면 디렉토리에서 직접 찾기
    if not video_file:
        for pattern in video_patterns:
            files = list(results_dir.glob(pattern))
            if files:
                video_file = str(files[0])
                break
    
    return annotations_file, video_file


def main():
    parser = argparse.ArgumentParser(description="Separated 파이프라인 결과 시각화 도구")
    
    # 입력 옵션
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='입력 비디오 파일')
    group.add_argument('--results-dir', type=str, help='separated 파이프라인 결과 디렉토리')
    
    parser.add_argument('--annotations', type=str, help='윈도우 어노테이션 파일 (results-dir 사용시 자동 감지)')
    parser.add_argument('--output-dir', type=str, help='출력 디렉토리 (기본: window_videos/)')
    
    # 시각화 옵션
    parser.add_argument('--num-persons', type=int, default=2, help='상위 정렬 person 수 (기본: 2)')
    parser.add_argument('--window-size', type=int, default=100, help='윈도우 크기 (기본: 100)')
    
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
            
            annotations_file, input_video = find_annotations_and_video_from_results(results_dir)
            
            if not annotations_file:
                logging.error(f"Window annotations file not found in {results_dir}")
                logging.info("Expected files: window_annotations.pkl, annotations.pkl")
                return False
            
            if not input_video:
                logging.error(f"Input video not found in {results_dir}")
                return False
            
            # 출력 디렉토리 결정
            if not args.output_dir:
                output_dir = results_dir / "window_videos"
            else:
                output_dir = args.output_dir
                
        else:
            # 직접 지정
            input_video = args.input
            
            if not args.annotations:
                logging.error("--annotations must be specified when using --input")
                return False
            
            annotations_file = args.annotations
            
            # 출력 디렉토리 결정
            if not args.output_dir:
                output_dir = Path("window_videos")
            else:
                output_dir = Path(args.output_dir)
        
        # 입력 검증
        if not Path(input_video).exists():
            logging.error(f"Input video not found: {input_video}")
            return False
        
        if not Path(annotations_file).exists():
            logging.error(f"Annotations file not found: {annotations_file}")
            return False
        
        # 어노테이션 로드
        window_annotations = load_window_annotations(str(annotations_file))
        if not window_annotations:
            logging.error("No window annotations found")
            return False
        
        logging.info(f"Input video: {input_video}")
        logging.info(f"Annotations file: {annotations_file}")
        logging.info(f"Window annotations: {len(window_annotations)} windows")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Top-ranked persons: {args.num_persons}")
        
        # 시각화 실행
        success = create_separated_visualization(
            input_video=input_video,
            window_annotations=window_annotations,
            output_dir=str(output_dir),
            num_persons=args.num_persons,
            window_size=args.window_size
        )
        
        if success:
            logging.info("Visualization completed successfully!")
            logging.info(f"Window videos saved to: {output_dir}")
            
            # 생성된 파일 목록 표시
            if output_dir.exists():
                video_files = list(output_dir.glob("*.mp4"))
                logging.info(f"Generated {len(video_files)} window videos:")
                for video_file in sorted(video_files):
                    logging.info(f"  - {video_file.name}")
            
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