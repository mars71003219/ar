#!/usr/bin/env python3
"""
Optimized STGCN++ Violence Detection Pipeline - Test Script
최적화된 싸움 분류 파이프라인 테스트 스크립트

Usage:
    python test_optimized_pipeline.py --mode single --video /path/to/video.mp4
    python test_optimized_pipeline.py --mode batch --input /path/to/videos/
    python test_optimized_pipeline.py --mode benchmark --input /path/to/test_videos/
"""

import sys
import os
import os.path as osp
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_violence_pipeline import OptimizedSTGCNPipeline
from pipeline_config import *


def single_video_test(pipeline: OptimizedSTGCNPipeline, video_path: str, output_dir: str) -> Dict:
    """단일 비디오 테스트"""
    print(f"🎬 단일 비디오 테스트: {osp.basename(video_path)}")
    
    start_time = time.time()
    result = pipeline.process_video_optimized(video_path, output_dir)
    total_time = time.time() - start_time
    
    if 'error' not in result:
        print(f"""
✅ 테스트 결과:
   - 비디오: {osp.basename(video_path)}
   - 예측: {result['prediction_label']}
   - 신뢰도: {result['confidence']:.4f}
   - 총 프레임: {result['total_frames']}
   - 처리 시간: {total_time:.2f}초
   - FPS: {result['fps']:.2f}
        """)
    else:
        print(f"❌ 테스트 실패: {result['error']}")
    
    return result


def batch_video_test(pipeline: OptimizedSTGCNPipeline, input_dir: str, output_dir: str, 
                    max_videos: int = 10) -> List[Dict]:
    """배치 비디오 테스트"""
    print(f"📁 배치 비디오 테스트: {input_dir}")
    
    # 비디오 파일 수집
    video_paths = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_paths.extend(Path(input_dir).rglob(ext))
    
    video_paths = [str(p) for p in video_paths[:max_videos]]
    
    if not video_paths:
        print("❌ 테스트할 비디오 파일이 없습니다.")
        return []
    
    print(f"📋 총 {len(video_paths)}개 비디오 처리 예정")
    
    start_time = time.time()
    results = pipeline.process_batch_videos(video_paths, output_dir, max_workers=4)
    total_time = time.time() - start_time
    
    # 결과 분석
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    fight_videos = [r for r in successful if r['prediction'] == 1]
    nonfight_videos = [r for r in successful if r['prediction'] == 0]
    
    avg_confidence = sum(r['confidence'] for r in successful) / len(successful) if successful else 0
    avg_fps = sum(r['fps'] for r in successful) / len(successful) if successful else 0
    
    print(f"""
📊 배치 테스트 결과:
   - 총 비디오: {len(video_paths)}개
   - 성공: {len(successful)}개
   - 실패: {len(failed)}개
   - Fight 분류: {len(fight_videos)}개
   - NonFight 분류: {len(nonfight_videos)}개
   - 평균 신뢰도: {avg_confidence:.4f}
   - 평균 FPS: {avg_fps:.2f}
   - 총 처리 시간: {total_time:.2f}초
   - 비디오당 평균 시간: {total_time/len(video_paths):.2f}초
    """)
    
    return results


def benchmark_test(pipeline: OptimizedSTGCNPipeline, test_dir: str, output_dir: str) -> Dict:
    """벤치마크 테스트 (정확도 평가)"""
    print(f"🏁 벤치마크 테스트: {test_dir}")
    
    # Fight와 NonFight 디렉토리 검색
    fight_dir = osp.join(test_dir, "Fight")
    nonfight_dir = osp.join(test_dir, "NonFight")
    
    fight_videos = []
    nonfight_videos = []
    
    if osp.exists(fight_dir):
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            fight_videos.extend(Path(fight_dir).glob(ext))
    
    if osp.exists(nonfight_dir):
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            nonfight_videos.extend(Path(nonfight_dir).glob(ext))
    
    fight_videos = [str(p) for p in fight_videos[:20]]  # 최대 20개씩
    nonfight_videos = [str(p) for p in nonfight_videos[:20]]
    
    all_videos = fight_videos + nonfight_videos
    ground_truths = [1] * len(fight_videos) + [0] * len(nonfight_videos)
    
    if not all_videos:
        print("❌ 테스트 비디오가 없습니다.")
        return {}
    
    print(f"📋 벤치마크 데이터: Fight {len(fight_videos)}개, NonFight {len(nonfight_videos)}개")
    
    # 배치 처리
    results = pipeline.process_batch_videos(all_videos, output_dir, max_workers=4)
    
    # 정확도 계산
    successful_results = [r for r in results if 'error' not in r]
    predictions = [r['prediction'] for r in successful_results]
    confidences = [r['confidence'] for r in successful_results]
    
    if len(successful_results) != len(ground_truths):
        print(f"⚠️ 일부 비디오 처리 실패: {len(ground_truths) - len(successful_results)}개")
        ground_truths = ground_truths[:len(successful_results)]
    
    # 혼동 행렬 계산
    tp = sum(1 for p, gt in zip(predictions, ground_truths) if p == 1 and gt == 1)
    tn = sum(1 for p, gt in zip(predictions, ground_truths) if p == 0 and gt == 0)
    fp = sum(1 for p, gt in zip(predictions, ground_truths) if p == 1 and gt == 0)
    fn = sum(1 for p, gt in zip(predictions, ground_truths) if p == 0 and gt == 1)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    benchmark_result = {
        'total_videos': len(all_videos),
        'successful': len(successful_results),
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        },
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
    }
    
    print(f"""
🎯 벤치마크 결과:
   - 총 비디오: {len(all_videos)}개
   - 처리 성공: {len(successful_results)}개
   
   혼동 행렬:
   - True Positive (Fight → Fight): {tp}
   - True Negative (NonFight → NonFight): {tn}
   - False Positive (NonFight → Fight): {fp}
   - False Negative (Fight → NonFight): {fn}
   
   성능 지표:
   - 정확도 (Accuracy): {accuracy:.4f}
   - 정밀도 (Precision): {precision:.4f}
   - 재현율 (Recall): {recall:.4f}
   - F1 점수: {f1_score:.4f}
   - 평균 신뢰도: {benchmark_result['avg_confidence']:.4f}
    """)
    
    # 벤치마크 결과 저장
    benchmark_path = osp.join(output_dir, 'benchmark_results.json')
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_result, f, indent=2)
    
    print(f"💾 벤치마크 결과 저장: {benchmark_path}")
    
    return benchmark_result


def compare_with_baseline(baseline_results_path: str, new_results: Dict):
    """기존 시스템과 성능 비교"""
    if not osp.exists(baseline_results_path):
        print(f"⚠️ 기준 결과 파일이 없습니다: {baseline_results_path}")
        return
    
    try:
        with open(baseline_results_path, 'r') as f:
            baseline = json.load(f)
        
        print(f"""
📈 성능 비교 (기존 vs 개선):
   - 정확도: {baseline['metrics']['accuracy']:.4f} → {new_results['metrics']['accuracy']:.4f} 
     ({new_results['metrics']['accuracy'] - baseline['metrics']['accuracy']:+.4f})
   - 정밀도: {baseline['metrics']['precision']:.4f} → {new_results['metrics']['precision']:.4f} 
     ({new_results['metrics']['precision'] - baseline['metrics']['precision']:+.4f})
   - 재현율: {baseline['metrics']['recall']:.4f} → {new_results['metrics']['recall']:.4f} 
     ({new_results['metrics']['recall'] - baseline['metrics']['recall']:+.4f})
   - F1 점수: {baseline['metrics']['f1_score']:.4f} → {new_results['metrics']['f1_score']:.4f} 
     ({new_results['metrics']['f1_score'] - baseline['metrics']['f1_score']:+.4f})
        """)
        
    except Exception as e:
        print(f"❌ 기준 결과 로드 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description='Optimized Violence Detection Pipeline Test')
    parser.add_argument('--mode', choices=['single', 'batch', 'benchmark'], required=True,
                       help='Test mode')
    parser.add_argument('--video', help='Single video path (for single mode)')
    parser.add_argument('--input', help='Input directory path (for batch/benchmark mode)')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--max-videos', type=int, default=10, 
                       help='Maximum videos to process (for batch mode)')
    parser.add_argument('--baseline', help='Baseline results file for comparison')
    
    args = parser.parse_args()
    
    # 설정 검증
    print("🔧 파이프라인 설정 검증 중...")
    if not validate_paths():
        print("❌ 필수 파일이 누락되었습니다. 설정을 확인해주세요.")
        return
    
    check_gpu_memory()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 파이프라인 초기화
    print("🚀 파이프라인 초기화 중...")
    pipeline = OptimizedSTGCNPipeline(
        pose_config=POSE_CONFIG,
        pose_checkpoint=POSE_CHECKPOINT,
        gcn_config=GCN_CONFIG,
        gcn_checkpoint=GCN_CHECKPOINT,
        device=PIPELINE_CONFIG['device'],
        sequence_length=PIPELINE_CONFIG['sequence_length']
    )
    
    try:
        if args.mode == 'single':
            if not args.video:
                print("❌ 단일 모드에서는 --video 인자가 필요합니다.")
                return
            
            single_video_test(pipeline, args.video, args.output)
            
        elif args.mode == 'batch':
            if not args.input:
                print("❌ 배치 모드에서는 --input 인자가 필요합니다.")
                return
            
            batch_video_test(pipeline, args.input, args.output, args.max_videos)
            
        elif args.mode == 'benchmark':
            if not args.input:
                print("❌ 벤치마크 모드에서는 --input 인자가 필요합니다.")
                return
            
            benchmark_results = benchmark_test(pipeline, args.input, args.output)
            
            # 기존 시스템과 비교
            if args.baseline:
                compare_with_baseline(args.baseline, benchmark_results)
        
        # 최종 성능 통계
        stats = pipeline.get_performance_stats()
        print(f"""
⚡ 최종 성능 통계:
   - 총 처리 프레임: {stats['total_frames']}
   - 평균 FPS: {stats.get('fps', 0):.2f}
   - GPU 메모리 사용량: {stats.get('gpu_memory_allocated', 0) / 1024**3:.2f} GB
        """)
        
    finally:
        # 리소스 정리
        pipeline.cleanup()


if __name__ == '__main__':
    main()