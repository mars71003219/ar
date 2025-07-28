#!/usr/bin/env python3
"""
Inference Runner Script
추론 실행 스크립트 - 엔드투엔드 파이프라인 실행을 위한 메인 스크립트
"""

import argparse
import logging
import os
import os.path as osp
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

# 현재 디렉토리를 Python 경로에 추가
current_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, current_dir)

from config import POSE_CONFIG, POSE_CHECKPOINT, GCN_CONFIG, GCN_CHECKPOINT
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, validate_config, check_gpu_availability
from main_pipeline import EndToEndPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="STGCN++ Violence Detection End-to-End Inference Pipeline"
    )
    
    # 입력 설정
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help='입력 비디오 파일 또는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--annotations', '-a',
        type=str,
        help='어노테이션 파일 경로 (video_name,label 형식)'
    )
    
    parser.add_argument(
        '--label-map',
        type=str,
        help='라벨 매핑 파일 경로 (Fight: 1, NonFight: 0 형식)'
    )
    
    # 출력 설정
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='출력 디렉토리 경로'
    )
    
    # 모델 설정
    parser.add_argument(
        '--pose-config',
        type=str,
        default=POSE_CONFIG,
        help='RTMO 모델 설정 파일 경로'
    )
    
    parser.add_argument(
        '--pose-checkpoint',
        type=str,
        default=POSE_CHECKPOINT,
        help='RTMO 모델 체크포인트 경로'
    )
    
    parser.add_argument(
        '--gcn-config',
        type=str,
        default=GCN_CONFIG,
        help='STGCN++ 모델 설정 파일 경로'
    )
    
    parser.add_argument(
        '--gcn-checkpoint',
        type=str,
        default=GCN_CHECKPOINT,
        help='STGCN++ 모델 체크포인트 경로'
    )
    
    # 실행 모드
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch', 'benchmark'],
        default='batch',
        help='실행 모드: single(단일 비디오), batch(배치 처리), benchmark(성능 평가)'
    )
    
    # 처리 옵션
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='배치 크기'
    )
    
    parser.add_argument(
        '--generate-overlay',
        action='store_true',
        help='오버레이 비디오 생성'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='추론 디바이스 (cuda:0, cpu 등)'
    )
    
    # 디버그 옵션
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 실행 없이 설정만 확인'
    )
    
    return parser.parse_args()

def load_annotations(annotation_file: str) -> Dict[str, int]:
    """어노테이션 파일 로드"""
    annotations = {}
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        video_name = parts[0].strip()
                        label = int(parts[1].strip())
                        annotations[video_name] = label
        
        logger.info(f"어노테이션 로드 완료: {len(annotations)}개 항목")
        return annotations
        
    except Exception as e:
        logger.error(f"어노테이션 파일 로드 실패: {e}")
        return {}

def load_label_map(label_map_file: str) -> Dict[str, int]:
    """라벨 매핑 파일 로드"""
    label_map = {}
    
    try:
        with open(label_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        label_name = parts[0].strip()
                        label_value = int(parts[1].strip())
                        label_map[label_name] = label_value
        
        logger.info(f"라벨 매핑 로드 완료: {label_map}")
        return label_map
        
    except Exception as e:
        logger.error(f"라벨 매핑 파일 로드 실패: {e}")
        return {}

def collect_video_files(input_path: str) -> List[str]:
    """비디오 파일 수집"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = []
    
    if osp.isfile(input_path):
        if Path(input_path).suffix.lower() in video_extensions:
            video_files.append(input_path)
    elif osp.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    video_files.append(osp.join(root, file))
    
    video_files.sort()
    logger.info(f"비디오 파일 수집 완료: {len(video_files)}개")
    
    return video_files

def match_annotations(video_files: List[str], annotations: Dict[str, int]) -> List[int]:
    """비디오 파일과 어노테이션 매칭"""
    matched_labels = []
    
    for video_path in video_files:
        video_name = osp.basename(video_path)
        
        # 확장자 제거한 이름으로도 시도
        video_name_no_ext = osp.splitext(video_name)[0]
        
        if video_name in annotations:
            matched_labels.append(annotations[video_name])
        elif video_name_no_ext in annotations:
            matched_labels.append(annotations[video_name_no_ext])
        else:
            logger.warning(f"어노테이션을 찾을 수 없습니다: {video_name}")
            matched_labels.append(None)
    
    valid_labels = [l for l in matched_labels if l is not None]
    logger.info(f"어노테이션 매칭 완료: {len(valid_labels)}/{len(video_files)}개")
    
    return matched_labels

def run_single_mode(pipeline: EndToEndPipeline, video_path: str, 
                   ground_truth: int = None, output_dir: str = "./results",
                   generate_overlay: bool = False) -> Dict:
    """단일 비디오 모드 실행"""
    logger.info("=== 단일 비디오 모드 실행 ===")
    
    result = pipeline.process_single_video(
        video_path, ground_truth, generate_overlay
    )
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    video_name = osp.splitext(osp.basename(video_path))[0]
    result_path = osp.join(output_dir, f"{video_name}_result.json")
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"단일 비디오 처리 완료: {result_path}")
    return result

def run_batch_mode(pipeline: EndToEndPipeline, video_files: List[str],
                  ground_truths: List[int] = None, output_dir: str = "./results",
                  generate_overlay: bool = False) -> Dict:
    """배치 처리 모드 실행"""
    logger.info("=== 배치 처리 모드 실행 ===")
    
    batch_result = pipeline.process_batch_videos(
        video_files, ground_truths, generate_overlay, 
        save_individual_results=True, output_dir=output_dir
    )
    
    # 종합 보고서 생성
    if batch_result.get('performance_metrics'):
        pipeline.generate_comprehensive_report(batch_result, output_dir)
    
    logger.info("배치 처리 완료")
    return batch_result

def run_benchmark_mode(pipeline: EndToEndPipeline, video_files: List[str],
                      ground_truths: List[int], output_dir: str = "./results",
                      generate_overlay: bool = False) -> Dict:
    """벤치마크 모드 실행"""
    logger.info("=== 벤치마크 모드 실행 ===")
    
    if not ground_truths or None in ground_truths:
        logger.error("벤치마크 모드는 모든 비디오에 대한 실제 라벨이 필요합니다")
        return {}
    
    # 실제 라벨이 있는 비디오만 필터링
    valid_pairs = [(v, gt) for v, gt in zip(video_files, ground_truths) if gt is not None]
    valid_videos = [pair[0] for pair in valid_pairs]
    valid_labels = [pair[1] for pair in valid_pairs]
    
    logger.info(f"벤치마크 대상: {len(valid_videos)}개 비디오")
    
    batch_result = pipeline.process_batch_videos(
        valid_videos, valid_labels, generate_overlay,
        save_individual_results=True, output_dir=output_dir
    )
    
    # 상세 성능 분석
    if batch_result.get('performance_metrics'):
        pipeline.generate_comprehensive_report(batch_result, output_dir)
        
        # 벤치마크 요약 출력
        metrics = batch_result['performance_metrics']['metrics']
        logger.info("=== 벤치마크 결과 요약 ===")
        logger.info(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
        logger.info(f"정밀도 (Precision): {metrics['precision']:.4f}")
        logger.info(f"재현율 (Recall): {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return batch_result

def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== STGCN++ Violence Detection Inference Pipeline ===")
    
    # 1. 설정 검증
    logger.info("1. 설정 검증 중...")
    if not validate_config():
        logger.error("설정 검증 실패")
        return 1
    
    check_gpu_availability()
    
    if args.dry_run:
        logger.info("Dry-run 모드: 설정 검증만 수행")
        return 0
    
    # 2. 입력 데이터 준비
    logger.info("2. 입력 데이터 준비 중...")
    video_files = collect_video_files(args.input)
    
    if not video_files:
        logger.error("처리할 비디오 파일이 없습니다")
        return 1
    
    # 어노테이션 로드
    annotations = {}
    ground_truths = None
    
    if args.annotations:
        annotations = load_annotations(args.annotations)
        if annotations:
            ground_truths = match_annotations(video_files, annotations)
    
    # 라벨 매핑 로드
    if args.label_map:
        label_map = load_label_map(args.label_map)
        logger.info(f"라벨 매핑: {label_map}")
    
    # 3. 파이프라인 초기화
    logger.info("3. 파이프라인 초기화 중...")
    try:
        pipeline = EndToEndPipeline(
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            gcn_config=args.gcn_config,
            gcn_checkpoint=args.gcn_checkpoint,
            device=args.device
        )
    except Exception as e:
        logger.error(f"파이프라인 초기화 실패: {e}")
        return 1
    
    # 4. 모드별 실행
    try:
        if args.mode == 'single':
            if len(video_files) > 1:
                logger.warning("단일 모드에서는 첫 번째 비디오만 처리됩니다")
            
            gt = ground_truths[0] if ground_truths else None
            result = run_single_mode(
                pipeline, video_files[0], gt, args.output, args.generate_overlay
            )
            
        elif args.mode == 'batch':
            result = run_batch_mode(
                pipeline, video_files, ground_truths, args.output, args.generate_overlay
            )
            
        elif args.mode == 'benchmark':
            result = run_benchmark_mode(
                pipeline, video_files, ground_truths, args.output, args.generate_overlay
            )
        
        logger.info("=== 파이프라인 실행 완료 ===")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        return 1
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {e}")
        return 1
    finally:
        # 5. 리소스 정리
        logger.info("5. 리소스 정리 중...")
        pipeline.cleanup()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)