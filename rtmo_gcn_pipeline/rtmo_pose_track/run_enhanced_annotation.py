#!/usr/bin/env python3
"""
Enhanced STGCN++ Annotation Generator - Main Execution Script
개선된 STGCN++ 데이터셋 어노테이션 생성기 실행 스크립트

사용법:
    python run_enhanced_annotation.py [mode] [options]

모드:
    single     - 단일 프로세스 처리 (기본값)
    parallel   - 병렬 프로세스 처리 (고속)
    demo       - 데모용 소량 처리
    analyze    - 결과 분석만 수행

예시:
    # 기본 단일 처리
    python run_enhanced_annotation.py single config.py checkpoint.pth --input /path/to/videos
    
    # 병렬 처리 (권장)
    python run_enhanced_annotation.py parallel config.py checkpoint.pth --input /path/to/videos --num-workers 4
    
    # 데모 모드 (처음 10개 비디오만)
    python run_enhanced_annotation.py demo config.py checkpoint.pth --input /path/to/videos
"""

import os
import sys
import argparse
import time
import multiprocessing as mp
from datetime import datetime

# CUDA 멀티프로세싱 호환성을 위해 spawn 방법 사용 (안전한 방식)
try:
    mp.set_start_method('spawn')
except RuntimeError:
    # 이미 설정된 경우 무시
    pass

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_rtmo_bytetrack_pose_extraction import main as single_main, find_video_files
from parallel_processor import run_parallel_processing


def print_banner():
    """시작 배너 출력"""
    banner = """
════════════════════════════════════════════════════════════════
              Enhanced STGCN++ Annotation Generator             
                  개선된 싸움 분류기 데이터셋 생성기                  
════════════════════════════════════════════════════════════════
    주요 개선사항:                                                   
    • 5영역 분할 기반 위치 점수  시스템                                
    • 복합 점수 계산  (움직임+위치+상호작용+시간적일관성+지속성)           
    • 적응적 영역 가중치 학습                                         
    • 모든 객체 랭킹 및 저장                                          
    • 실패 케이스 체계적 로깅                                         
    • 병렬 처리 및 성능 최적화                                        
════════════════════════════════════════════════════════════════
"""
    print(banner)


def parse_main_args():
    """메인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Enhanced STGCN++ Annotation Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s single config.py checkpoint.pth --input /data/videos
  %(prog)s parallel config.py checkpoint.pth --input /data/videos --num-workers 4
  %(prog)s parallel config.py checkpoint.pth --input /data/videos --save-overlayfile
  %(prog)s demo config.py checkpoint.pth --input /data/videos
  %(prog)s analyze --output-root /results
        """
    )
    
    parser.add_argument('mode', choices=['single', 'parallel', 'demo', 'analyze'],
                       help='실행 모드')
    parser.add_argument('config', nargs='?', help='RTMO config file')
    parser.add_argument('checkpoint', nargs='?', help='RTMO checkpoint file')
    
    # 입출력 경로
    parser.add_argument('--input', type=str,
                       default='/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000',
                       help='입력 비디오 경로 또는 디렉토리')
    parser.add_argument('--output-root', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='출력 디렉토리')
    parser.add_argument('--save-overlayfile', action='store_true', default=True,
                       help='오버레이 시각화 비디오 파일 저장')
    
    # 모델 설정
    parser.add_argument('--device', default='cuda:0', help='추론 디바이스')
    parser.add_argument('--score-thr', type=float, default=0.3, help='검출 점수 임계값')
    parser.add_argument('--nms-thr', type=float, default=0.35, help='NMS 임계값')
    
    # 개선된 설정
    parser.add_argument('--min-track-length', type=int, default=10,
                       help='포함할 최소 트랙 길이')
    parser.add_argument('--quality-threshold', type=float, default=0.3,
                       help='최소 트랙 품질 임계값')
    
    # ByteTrack 설정
    parser.add_argument('--track-high-thresh', type=float, default=0.6)
    parser.add_argument('--track-low-thresh', type=float, default=0.1)
    parser.add_argument('--track-max-disappeared', type=int, default=30)
    parser.add_argument('--track-min-hits', type=int, default=3)
    
    # 병렬 처리 설정
    parser.add_argument('--num-workers', type=int, default=None,
                       help='병렬 처리 워커 수 (기본값: 자동)')
    
    # 데모 모드 설정
    parser.add_argument('--demo-count', type=int, default=10,
                       help='데모 모드에서 처리할 비디오 수')
    
    return parser.parse_args()


def validate_args(args):
    """인자 유효성 검사"""
    if args.mode in ['single', 'parallel', 'demo']:
        if not args.config or not args.checkpoint:
            print("ERROR: config와 checkpoint 파일이 필요합니다.")
            return False
        
        if not os.path.exists(args.config):
            print(f"ERROR: Config 파일을 찾을 수 없습니다: {args.config}")
            return False
        
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint 파일을 찾을 수 없습니다: {args.checkpoint}")
            return False
        
        if not os.path.exists(args.input):
            print(f"ERROR: 입력 경로를 찾을 수 없습니다: {args.input}")
            return False
    
    return True


def run_single_mode(args):
    """단일 프로세스 모드 실행"""
    print(" Single Process Mode - 단일 프로세스 처리를 시작합니다...")
    
    # 기존 main 함수 사용
    original_argv = sys.argv
    try:
        # sys.argv를 임시로 수정
        sys.argv = [
            'enhanced_rtmo_bytetrack_pose_extraction.py',
            args.config,
            args.checkpoint,
            '--input', args.input,
            '--output-root', args.output_root,
            '--device', args.device,
            '--score-thr', str(args.score_thr),
            '--nms-thr', str(args.nms_thr),
            '--min-track-length', str(args.min_track_length),
            '--quality-threshold', str(args.quality_threshold),
            '--track-high-thresh', str(args.track_high_thresh),
            '--track-low-thresh', str(args.track_low_thresh),
            '--track-max-disappeared', str(args.track_max_disappeared),
            '--track-min-hits', str(args.track_min_hits)
        ]
        
        single_main()
        
    finally:
        sys.argv = original_argv


def run_parallel_mode(args):
    """병렬 프로세스 모드 실행"""
    print(" Parallel Process Mode - 병렬 처리를 시작합니다...")
    print(f"워커 수: {args.num_workers or 'auto'}")
    
    run_parallel_processing(args)


def run_demo_mode(args):
    """데모 모드 실행"""
    print(f" Demo Mode - 처음 {args.demo_count}개 비디오로 데모를 실행합니다...")
    
    # 비디오 파일 목록 가져오기
    video_files = find_video_files(args.input)
    if not video_files:
        print(f"ERROR: {args.input}에서 비디오 파일을 찾을 수 없습니다.")
        return
    
    # 데모용으로 제한
    demo_videos = video_files[:args.demo_count]
    print(f"전체 {len(video_files)}개 중 {len(demo_videos)}개 비디오로 데모 실행")
    
    # 임시 디렉토리에 심볼릭 링크 생성
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_input_dir = os.path.join(temp_dir, 'demo_videos')
        os.makedirs(demo_input_dir)
        
        for i, video_path in enumerate(demo_videos):
            link_path = os.path.join(demo_input_dir, f"demo_{i:03d}_{os.path.basename(video_path)}")
            try:
                os.symlink(video_path, link_path)
            except OSError:
                # 심볼릭 링크 실패 시 복사
                shutil.copy2(video_path, link_path)
        
        # args 수정하여 데모 디렉토리 사용
        demo_args = args
        demo_args.input = demo_input_dir
        demo_args.output_root = os.path.join(args.output_root, 'demo_results')
        
        # 단일 모드로 실행 (데모이므로)
        run_single_mode(demo_args)


def run_analyze_mode(args):
    """분석 모드 실행"""
    print(" Analyze Mode - 결과 분석을 시작합니다...")
    
    if not os.path.exists(args.output_root):
        print(f"ERROR: 출력 디렉토리가 존재하지 않습니다: {args.output_root}")
        return
    
    # PKL 파일 찾기
    import glob
    pkl_files = glob.glob(os.path.join(args.output_root, '**/*enhanced_stgcn_annotation.pkl'), recursive=True)
    
    if not pkl_files:
        print("ERROR: 분석할 어노테이션 파일이 없습니다.")
        return
    
    print(f"총 {len(pkl_files)}개의 어노테이션 파일을 발견했습니다.")
    
    # 분석 수행
    analyze_results(pkl_files, args.output_root)


def analyze_results(pkl_files, output_root):
    """결과 분석 수행"""
    import pickle
    import numpy as np
    from collections import defaultdict
    
    stats = {
        'total_files': len(pkl_files),
        'total_persons': 0,
        'score_distribution': defaultdict(int),
        'region_distribution': defaultdict(float),
        'quality_distribution': defaultdict(int),
        'track_length_distribution': defaultdict(int)
    }
    
    print("분석 중...")
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            stats['total_persons'] += data['total_persons']
            
            # 점수 분포 분석
            for person_key, person_data in data['persons'].items():
                score = person_data['composite_score']
                score_range = f"{int(score*10)/10:.1f}-{int(score*10)/10+0.1:.1f}"
                stats['score_distribution'][score_range] += 1
                
                # 품질 분포
                quality = person_data['track_quality']
                quality_range = f"{int(quality*10)/10:.1f}-{int(quality*10)/10+0.1:.1f}"
                stats['quality_distribution'][quality_range] += 1
                
                # 영역 분포 (최고 점수 영역)
                region_scores = person_data['region_breakdown']
                if region_scores:
                    best_region = max(region_scores.items(), key=lambda x: x[1])[0]
                    stats['region_distribution'][best_region] += 1
        
        except Exception as e:
            print(f"Warning: {pkl_file} 분석 실패: {e}")
    
    # 결과 출력
    print("\n" + "="*60)
    print("***** ANALYSIS RESULTS *****")
    print("="*60)
    print(f"전체 파일 수: {stats['total_files']}")
    print(f"전체 인물 수: {stats['total_persons']}")
    print(f"파일당 평균 인물 수: {stats['total_persons']/stats['total_files']:.1f}")
    
    print(f"\n1. 점수 분포:")
    for score_range, count in sorted(stats['score_distribution'].items()):
        percentage = count / stats['total_persons'] * 100
        print(f"  {score_range}: {count:4d}명 ({percentage:5.1f}%)")
    
    print(f"\n2. 영역 분포:")
    total_regions = sum(stats['region_distribution'].values())
    for region, count in sorted(stats['region_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_regions * 100 if total_regions > 0 else 0
        print(f"  {region:15s}: {count:4.0f}명 ({percentage:5.1f}%)")
    
    print(f"\n3. 품질 분포:")
    for quality_range, count in sorted(stats['quality_distribution'].items()):
        percentage = count / stats['total_persons'] * 100
        print(f"  {quality_range}: {count:4d}명 ({percentage:5.1f}%)")
    
    # 분석 결과를 파일로 저장
    analysis_file = os.path.join(output_root, 'analysis_report.txt')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(f"Enhanced STGCN++ Annotation Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total files: {stats['total_files']}\n")
        f.write(f"Total persons: {stats['total_persons']}\n")
        f.write(f"Average persons per file: {stats['total_persons']/stats['total_files']:.1f}\n\n")
        
        f.write("Score Distribution:\n")
        for score_range, count in sorted(stats['score_distribution'].items()):
            percentage = count / stats['total_persons'] * 100
            f.write(f"  {score_range}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nRegion Distribution:\n")
        for region, count in sorted(stats['region_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_regions * 100 if total_regions > 0 else 0
            f.write(f"  {region}: {count:.0f} ({percentage:.1f}%)\n")
    
    print(f"\n 분석 보고서가 저장되었습니다: {analysis_file}")


def main():
    """메인 실행 함수"""
    print_banner()
    
    args = parse_main_args()
    
    if not validate_args(args):
        sys.exit(1)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_root, exist_ok=True)
    
    print(f"- 출력 디렉토리: {args.output_root}")
    print(f"- 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        if args.mode == 'single':
            run_single_mode(args)
        elif args.mode == 'parallel':
            run_parallel_mode(args)
        elif args.mode == 'demo':
            run_demo_mode(args)
        elif args.mode == 'analyze':
            run_analyze_mode(args)
            
    except KeyboardInterrupt:
        print("\nSuspend:  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    print(f"\n***** 처리 완료 *****")
    print(f"1. 총 소요 시간: {elapsed_time/3600:.2f}시간")
    print(f"2. 결과 디렉토리: {args.output_root}")


if __name__ == '__main__':
    main()