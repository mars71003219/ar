#!/usr/bin/env python3
"""
Step1 vs Step2 포즈 분석 실행 스크립트
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
    class MockConfig:
        def __init__(self):
            self.default_input_dir = './test_data'
            self.default_output_dir = './output'
            self.confidence_threshold = 0.3
            self.colors = [(255,0,0), (0,255,0), (0,0,255)]
            self.SAVE_OVERLAY_VIDEO = True
    
    global_config = MockConfig()


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Step1 vs Step2 Pose Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Step1과 Step2 PKL 파일 비교 분석
  python run_pose_analysis.py --step1-pkl /path/to/step1_poses.pkl --step2-pkl /path/to/step2_windows.pkl --video /path/to/video.mp4 --output-dir /path/to/analysis

  # 분석 결과는 analysis_logs 폴더에 저장됩니다
        """)
    
    parser.add_argument('--step1-pkl', type=str, required=True,
                       help='Step1 poses PKL file path')
    parser.add_argument('--step2-pkl', type=str, required=True,
                       help='Step2 tracking PKL file path')
    parser.add_argument('--video', type=str, required=True,
                       help='Video file path')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # 경로 검증
    if not os.path.exists(args.step1_pkl):
        print(f"Error: Step1 PKL file not found: {args.step1_pkl}")
        return
    
    if not os.path.exists(args.step2_pkl):
        print(f"Error: Step2 PKL file not found: {args.step2_pkl}")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Step1 vs Step2 포즈 분석 ===")
    print(f"Step1 PKL: {args.step1_pkl}")
    print(f"Step2 PKL: {args.step2_pkl}")
    print(f"Video: {args.video}")
    print(f"Output: {args.output_dir}")
    print()
    
    try:
        # EnhancedVisualizer 생성 (분석용)
        visualizer = EnhancedVisualizer(
            mode='separated_overlay',
            video_dir=None,
            pkl_dir=None,
            save=True,
            save_dir=args.output_dir,
            num_person=2,
            config=global_config,
            stage='analysis'  # 분석 모드
        )
        
        # 분석 실행
        success = visualizer.run_analysis(args.step1_pkl, args.step2_pkl, args.video)
        
        if success:
            print("\n=== 분석 성공 ===")
            print(f"결과 확인: {args.output_dir}/analysis_logs/")
        else:
            print("\n=== 분석 실패 ===")
            return 1
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())