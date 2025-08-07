#!/usr/bin/env python3
"""
Enhanced Visualization Script - 개선된 시각화 실행 스크립트
"""

import os
import sys
import argparse

# 상위 디렉토리를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from visualization.modes.inference_mode import InferenceVisualizerMode
from visualization.modes.separated_mode import SeparatedVisualizerMode

# 설정 파일 로드
try:
    from configs.visualizer_config import config as global_config
except ImportError:
    print("Warning: Could not import visualizer config. Using default settings.")
    global_config = None


def create_main_parser():
    """메인 명령행 파서 생성"""
    parser = argparse.ArgumentParser(
        description='Enhanced RTMO Pose Tracking Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 추론 모드로 시각화
  python run_visualization.py inference --input-dir /path/to/data --output-dir /path/to/output
  
  # 분리된 파이프라인 모드로 시각화
  python run_visualization.py separated --input-dir /path/to/data --output-dir /path/to/output
  
  # 스켈레톤 없이 시각화
  python run_visualization.py inference --input-dir /path/to/data --no-skeleton
        """)
    
    # 공통 인수
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory containing videos and PKL files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualization results')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--version', action='version', version='Enhanced Visualization Tool v1.0')
    
    # 서브 명령
    subparsers = parser.add_subparsers(dest='mode', help='Visualization modes')
    
    # 추론 모드 서브파서
    inference_mode = InferenceVisualizerMode()
    inference_parser = subparsers.add_parser('inference', 
                                           parents=[inference_mode.create_parser()],
                                           add_help=False,
                                           help='Inference mode visualization')
    
    # 분리된 모드 서브파서
    separated_mode = SeparatedVisualizerMode()
    separated_parser = subparsers.add_parser('separated',
                                           parents=[separated_mode.create_parser()],
                                           add_help=False, 
                                           help='Separated pipeline mode visualization')
    
    return parser


def main():
    """메인 실행 함수"""
    parser = create_main_parser()
    
    # 인수가 없는 경우 도움말 표시
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        args = parser.parse_args()
        
        # 모드가 지정되지 않은 경우
        if not args.mode:
            print("Error: Please specify a visualization mode (inference or separated)")
            parser.print_help()
            return
        
        print(f"Starting {args.mode} visualization mode...")
        
        # 사용자 지정 설정 파일 로드 (옵션)
        config_to_use = global_config
        if args.config_file:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("custom_config", args.config_file)
                custom_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_config)
                if hasattr(custom_config, 'config'):
                    config_to_use = custom_config.config
                    print(f"Loaded custom configuration from: {args.config_file}")
                else:
                    print(f"Warning: No 'config' object found in {args.config_file}")
            except Exception as e:
                print(f"Warning: Failed to load custom config file: {e}")
                print("Using default configuration.")
        
        # 해당 모드의 시각화 도구 생성 (설정 파일 전달)
        if args.mode == 'inference':
            visualizer = InferenceVisualizerMode(args.input_dir, args.output_dir, config_to_use)
            visualizer.configure_from_args(args)
        elif args.mode == 'separated':
            visualizer = SeparatedVisualizerMode(args.input_dir, args.output_dir, config_to_use)
            visualizer.configure_from_args(args)
        else:
            print(f"Unknown mode: {args.mode}")
            return
        
        # 경로 확인 및 정보 출력
        print(f"Input Directory: {visualizer.input_dir}")
        print(f"Output Directory: {visualizer.output_dir}")
        
        if not os.path.exists(visualizer.input_dir):
            print(f"Warning: Input directory not found: {visualizer.input_dir}")
        
        if not os.path.exists(visualizer.output_dir):
            print(f"Info: Output directory will be created: {visualizer.output_dir}")
        
        print()
        
        # 시각화 실행
        visualizer.run()
        
        print(f"\n{args.mode.capitalize()} visualization completed successfully!")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()