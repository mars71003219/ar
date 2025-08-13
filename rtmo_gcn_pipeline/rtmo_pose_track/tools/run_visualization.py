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

from visualization.visualizer import EnhancedVisualizer

# 설정 파일 로드
try:
    from configs.visualizer_config import config as global_config
    print(f"Loaded visualizer config successfully.")
except ImportError:
    print("Warning: Could not import visualizer config. Using default settings.")
    # Fallback or exit if config is critical
    class MockConfig:
        def __init__(self):
            self.default_input_dir = './test_data'
            self.default_output_dir = './output'
            self.max_persons = 2
            self.confidence_threshold = 0.3
            self.verbose = True
            self.fourcc_codec = 'mp4v'
            self.colors = [(255,0,0), (0,255,0), (0,0,255)]
            self.fight_color = (0,0,255)
            self.nonfight_color = (0,255,0)
            self.text_color = (255,255,255)
            self.overlap_color = (255,0,255)
            self.font_scale = 0.7
            self.font_thickness = 2
            self.title_font_scale = 0.8
            self.title_font_thickness = 2
            self.box_padding = 10
            self.line_thickness = 2
            self.keypoint_radius = 3
            self.skeleton_connections = []
            self.SAVE_OVERLAY_VIDEO = True
            self.OVERLAY_SUB_DIR = 'overlay'
            self.debug_mode = False
            self.consecutive_threshold = 3
            self.window_info_x = 10
            self.window_info_y_start = 20
            self.window_info_y_step = 15
            self.frame_info_margin = 5
            self.final_result_margin = 5
    
    global_config = MockConfig()


def create_main_parser():
    """메인 명령행 파서 생성"""
    parser = argparse.ArgumentParser(
        description='Enhanced RTMO Pose Tracking Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 추론 모드로 시각화 (비디오와 PKL이 같은 폴더)
  python run_visualization.py inference --input-dir /path/to/data --output-dir /path/to/output
  
  # 분리된 파이프라인 모드로 시각화 (비디오와 PKL이 다른 폴더)
  python run_visualization.py separated --video-dir /path/to/videos --pkl-dir /path/to/pkls --output-dir /path/to/output
  
  # 스켈레톤 없이 시각화
  python run_visualization.py inference --input-dir /path/to/data --no-skeleton
        """)
    
    # 서브 명령
    subparsers = parser.add_subparsers(dest='mode', help='Visualization modes')
    
    # 공통 인수를 각 서브파서에 추가하는 함수
    def add_common_args(subparser):
        subparser.add_argument('--input-dir', type=str, default=None,
                             help='Input directory containing videos and PKL files')
        subparser.add_argument('--video-dir', type=str, default=None,
                             help='Directory containing video files (if different from input-dir)')
        subparser.add_argument('--pkl-dir', type=str, default=None,
                             help='Directory containing PKL files (if different from input-dir)')
        subparser.add_argument('--output-dir', type=str, default=None,
                             help='Output directory for visualization results')
        subparser.add_argument('--config-file', type=str,
                             default="/workspace/rtmo_gcn_pipeline/rtmo_pose_track/configs/visualizer_config.py",
                             help='Path to custom configuration file')
        subparser.add_argument('--no-skeleton', action='store_true',
                             help='Disable skeleton visualization')
        subparser.add_argument('--no-predictions', action='store_true',
                             help='Disable prediction information display')
        subparser.add_argument('--num-person', type=int, default=2,
                             help='Number of top persons to display (default: 2)')
        subparser.add_argument('--save', action='store_true',
                             help='Save overlay video to file')
    
    # 추론 모드 서브파서
    inference_parser = subparsers.add_parser('inference', 
                                           help='Inference mode visualization')
    add_common_args(inference_parser)
    
    # 분리된 모드 서브파서
    separated_parser = subparsers.add_parser('separated',
                                           help='Separated pipeline mode visualization')
    add_common_args(separated_parser)
    separated_parser.add_argument('--stage', type=str, 
                                choices=['stage1', 'stage2', 'step1', 'step2'], 
                                default='stage1',
                                help='Pipeline stage: stage1/step1 (poses), stage2/step2 (tracking)')
    
    # 메인 파서에 버전 정보 추가
    parser.add_argument('--version', action='version', version='Enhanced Visualization Tool v1.0')
    
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
        
        # 경로 처리 로직
        input_dir = args.input_dir
        video_dir = args.video_dir or args.input_dir
        pkl_dir = args.pkl_dir or args.input_dir
        
        # 경로 검증
        if not input_dir and not (args.video_dir and args.pkl_dir):
            print("Error: Either --input-dir or both --video-dir and --pkl-dir must be specified")
            return
        
        # EnhancedVisualizer 모드 설정
        if args.mode == 'inference':
            mode = 'inference_overlay'
        elif args.mode == 'separated':
            mode = 'separated_overlay'
        else:
            print(f"Unknown mode: {args.mode}")
            return
        
        # visualizer_config.py의 SAVE_OVERLAY_VIDEO 설정 확인
        should_save = getattr(args, 'save', False) or config_to_use.SAVE_OVERLAY_VIDEO
        
        # 해당 모드의 시각화 도구 생성 (설정 파일 적용)
        visualizer = EnhancedVisualizer(
            mode=mode,
            video_dir=video_dir,
            pkl_dir=pkl_dir,
            save=should_save,
            save_dir=args.output_dir,
            num_person=getattr(args, 'num_person', 2),
            config=config_to_use,  # 설정 파일 전달
            stage=getattr(args, 'stage', 'stage1')  # stage 정보 전달
        )
        
        # 경로 및 설정 확인 및 정보 출력
        print(f"Video Directory: {visualizer.video_dir}")
        print(f"PKL Directory: {visualizer.pkl_dir}")
        print(f"Save Directory: {visualizer.save_dir}")
        print(f"Mode: {visualizer.mode}")
        print(f"Number of persons: {visualizer.num_persons}")
        print(f"Save enabled: {visualizer.save}")
        print(f"Confidence threshold: {visualizer.config.confidence_threshold}")
        if visualizer.config.SAVE_OVERLAY_VIDEO:
            final_save_path = os.path.join(visualizer.save_dir, visualizer.config.OVERLAY_SUB_DIR)
            print(f"Final save path (with overlay subdir): {final_save_path}")
        
        # 경로 존재 확인
        if not os.path.exists(visualizer.video_dir):
            print(f"Warning: Video directory not found: {visualizer.video_dir}")
        
        if not os.path.exists(visualizer.pkl_dir):
            print(f"Warning: PKL directory not found: {visualizer.pkl_dir}")
        
        if visualizer.save:
            if not os.path.exists(visualizer.save_dir):
                print(f"Info: Save directory will be created: {visualizer.save_dir}")
            if config_to_use.SAVE_OVERLAY_VIDEO and config_to_use.OVERLAY_SUB_DIR:
                overlay_dir = os.path.join(visualizer.save_dir, config_to_use.OVERLAY_SUB_DIR)
                if not os.path.exists(overlay_dir):
                    print(f"Info: Overlay directory will be created: {overlay_dir}")
        
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