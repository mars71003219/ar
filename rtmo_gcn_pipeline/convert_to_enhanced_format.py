#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
Enhanced Annotation Format Converter
기존 enhanced_rtmo_bytetrack_pose_extraction.py 출력을 MMAction2 학습용 형태로 변환

주요 기능:
1. 개별 pkl 파일들을 통합 annotation 파일로 변환
2. Train/Val/Test split 생성
3. Enhanced format 검증 및 정제
4. 품질 통계 및 분석 리포트 생성
"""

import os
import pickle
import glob
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm


def load_individual_annotations(input_dir: str, pattern: str = "*_enhanced_stgcn_annotation.pkl") -> Dict:
    """개별 annotation 파일들을 로드"""
    
    print(f" Searching for annotation files in: {input_dir}")
    print(f" Using pattern: {pattern}")
    
    pkl_files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    print(f"Found {len(pkl_files)} enhanced annotation files")
    
    # 디버깅: 찾은 파일들 출력
    if len(pkl_files) == 0:
        print(" No annotation files found!")
        print(" Searching for any .pkl files to debug...")
        all_pkl = glob.glob(os.path.join(input_dir, "**", "*.pkl"), recursive=True)
        print(f"Found {len(all_pkl)} total .pkl files:")
        for pkl in all_pkl[:10]:  # 처음 10개만 출력
            print(f"  - {pkl}")
        if len(all_pkl) > 10:
            print(f"  ... and {len(all_pkl) - 10} more files")
        return {}
    else:
        print(" Found annotation files:")
        for pkl in pkl_files[:5]:  # 처음 5개만 출력
            print(f"  - {pkl}")
        if len(pkl_files) > 5:
            print(f"  ... and {len(pkl_files) - 5} more files")
    
    annotations = {}
    failed_files = []
    
    for pkl_file in tqdm(pkl_files, desc="Loading annotations"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 파일명에서 비디오 이름 추출
            video_name = os.path.splitext(os.path.basename(pkl_file))[0]
            video_name = video_name.replace('_enhanced_stgcn_annotation', '')
            
            annotations[video_name] = data
            
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")
            failed_files.append(pkl_file)
    
    print(f"Successfully loaded {len(annotations)} annotations")
    print(f"Failed to load {len(failed_files)} files")
    
    return annotations


def validate_and_clean_annotations(annotations: Dict) -> Tuple[Dict, Dict]:
    """Annotation 검증 및 정제"""
    
    cleaned_annotations = {}
    stats = {
        'total_videos': len(annotations),
        'valid_videos': 0,
        'invalid_videos': 0,
        'total_persons': 0,
        'avg_persons_per_video': 0.0,
        'quality_distribution': {},
        'label_distribution': {'fight': 0, 'non_fight': 0}
    }
    
    quality_scores = []
    person_counts = []
    
    for video_name, annotation in tqdm(annotations.items(), desc="Validating"):
        
        # 필수 키 확인 (수정된 구조)
        required_keys = ['video_info', 'persons', 'score_weights']
        if not all(key in annotation for key in required_keys):
            print(f"Missing required keys in {video_name}")
            stats['invalid_videos'] += 1
            continue
        
        # video_info 검증
        video_info = annotation['video_info']
        if 'label' not in video_info:
            # 파일명에서 라벨 추론
            if 'fight' in video_name.lower():
                video_info['label'] = 1
                stats['label_distribution']['fight'] += 1
            else:
                video_info['label'] = 0
                stats['label_distribution']['non_fight'] += 1
        else:
            if video_info['label'] == 1:
                stats['label_distribution']['fight'] += 1
            else:
                stats['label_distribution']['non_fight'] += 1
        
        # 품질 정보 수집
        total_persons = annotation.get('total_persons', 0)
        quality_threshold = annotation.get('quality_threshold', 0.3)
        
        if total_persons > 0:
            person_counts.append(total_persons)
            quality_scores.append(quality_threshold)
            stats['total_persons'] += total_persons
            stats['valid_videos'] += 1
            cleaned_annotations[video_name] = annotation
        else:
            print(f"No persons found in {video_name}")
            stats['invalid_videos'] += 1
    
    # 통계 계산
    if person_counts:
        stats['avg_persons_per_video'] = np.mean(person_counts)
        stats['quality_distribution'] = {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
    
    return cleaned_annotations, stats


def create_train_val_test_split(annotations: Dict, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               stratify_by_label: bool = True) -> Tuple[Dict, Dict, Dict]:
    """Train/Val/Test split 생성"""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # 라벨별로 분류
    if stratify_by_label:
        fight_videos = []
        non_fight_videos = []
        
        for video_name, annotation in annotations.items():
            label = annotation['video_info'].get('label', 0)
            if label == 1:
                fight_videos.append(video_name)
            else:
                non_fight_videos.append(video_name)
        
        # 각 클래스별로 split
        random.shuffle(fight_videos)
        random.shuffle(non_fight_videos)
        
        def split_videos(videos, train_r, val_r, test_r):
            n = len(videos)
            train_end = int(n * train_r)
            val_end = train_end + int(n * val_r)
            
            return (videos[:train_end], 
                   videos[train_end:val_end], 
                   videos[val_end:])
        
        fight_train, fight_val, fight_test = split_videos(fight_videos, train_ratio, val_ratio, test_ratio)
        non_fight_train, non_fight_val, non_fight_test = split_videos(non_fight_videos, train_ratio, val_ratio, test_ratio)
        
        # 결합
        train_videos = fight_train + non_fight_train
        val_videos = fight_val + non_fight_val
        test_videos = fight_test + non_fight_test
        
    else:
        # 전체 비디오 무작위 split
        all_videos = list(annotations.keys())
        random.shuffle(all_videos)
        
        n = len(all_videos)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_videos = all_videos[:train_end]
        val_videos = all_videos[train_end:val_end]
        test_videos = all_videos[val_end:]
    
    # Split별 annotation dict 생성
    train_annotations = {video: annotations[video] for video in train_videos}
    val_annotations = {video: annotations[video] for video in val_videos}
    test_annotations = {video: annotations[video] for video in test_videos}
    
    # Split 정보 추가
    for split_name, split_annotations in [('train', train_annotations), 
                                         ('val', val_annotations), 
                                         ('test', test_annotations)]:
        for annotation in split_annotations.values():
            annotation['split'] = split_name
    
    print(f"Dataset split completed:")
    print(f"  Train: {len(train_annotations)} videos")
    print(f"  Val: {len(val_annotations)} videos")
    print(f"  Test: {len(test_annotations)} videos")
    
    # 라벨 분포 확인
    for split_name, split_annotations in [('Train', train_annotations), 
                                         ('Val', val_annotations), 
                                         ('Test', test_annotations)]:
        fight_count = sum(1 for ann in split_annotations.values() 
                         if ann['video_info'].get('label', 0) == 1)
        total = len(split_annotations)
        print(f"  {split_name} - Fight: {fight_count}, Non-fight: {total - fight_count}")
    
    return train_annotations, val_annotations, test_annotations


def save_mmaction_format(annotations: Dict, output_path: str, split_name: str):
    """MMAction2 형태로 저장"""
    
    # MMAction2에서 기대하는 형태로 변환
    mmaction_data = {}
    
    for video_name, annotation in annotations.items():
        mmaction_data[video_name] = annotation
    
    # 메타데이터 추가
    mmaction_data['_metadata'] = {
        'dataset_type': 'Enhanced RTMO Fight Detection',
        'split': split_name,
        'total_videos': len(annotations),
        'fight_videos': sum(1 for ann in annotations.values() 
                           if ann['video_info'].get('label', 0) == 1),
        'enhanced_features': [
            'fight_prioritized_ranking',
            '5_region_spatial_analysis', 
            'composite_scoring',
            'quality_based_filtering',
            'adaptive_person_selection'
        ]
    }
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(mmaction_data, f)
    
    print(f"Saved {split_name} annotations to {output_path}")


def generate_analysis_report(annotations: Dict, stats: Dict, output_dir: str):
    """분석 리포트 생성"""
    
    report_path = os.path.join(output_dir, 'enhanced_dataset_analysis.txt')
    
    with open(report_path, 'w') as f:
        f.write("Enhanced RTMO Fight Detection Dataset Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write(f"  Total videos: {stats['total_videos']}\n")
        f.write(f"  Valid videos: {stats['valid_videos']}\n")
        f.write(f"  Invalid videos: {stats['invalid_videos']}\n")
        f.write(f"  Total persons detected: {stats['total_persons']}\n")
        f.write(f"  Average persons per video: {stats['avg_persons_per_video']:.2f}\n\n")
        
        f.write("Label Distribution:\n")
        f.write(f"  Fight videos: {stats['label_distribution']['fight']}\n")
        f.write(f"  Non-fight videos: {stats['label_distribution']['non_fight']}\n")
        
        # ZeroDivisionError 방지
        if stats['valid_videos'] > 0:
            fight_ratio = stats['label_distribution']['fight'] / stats['valid_videos']
            f.write(f"  Fight ratio: {fight_ratio:.3f}\n\n")
        else:
            f.write(f"  Fight ratio: N/A (no valid videos found)\n\n")
        
        f.write("Quality Distribution:\n")
        if stats['quality_distribution']:
            qd = stats['quality_distribution']
            f.write(f"  Mean quality: {qd['mean']:.3f}\n")
            f.write(f"  Std quality: {qd['std']:.3f}\n")
            f.write(f"  Min quality: {qd['min']:.3f}\n")
            f.write(f"  Max quality: {qd['max']:.3f}\n\n")
        
        f.write("Enhanced Features Analysis:\n")
        
        # 복합 점수 분석 (수정된 구조)
        composite_scores = []
        region_scores = defaultdict(list)
        
        for annotation in annotations.values():
            persons_dict = annotation.get('persons', {})
            
            # persons가 dict인 경우 values()로 처리
            if isinstance(persons_dict, dict):
                for person in persons_dict.values():
                    if isinstance(person, dict) and 'composite_score' in person:
                        composite_scores.append(person['composite_score'])
                    
                        region_breakdown = person.get('region_breakdown', {})
                        if isinstance(region_breakdown, dict):
                            for region, score in region_breakdown.items():
                                region_scores[region].append(score)
        
        if composite_scores:
            f.write(f"  Composite Score Distribution:\n")
            f.write(f"    Mean: {np.mean(composite_scores):.3f}\n")
            f.write(f"    Std: {np.std(composite_scores):.3f}\n")
            f.write(f"    Min: {np.min(composite_scores):.3f}\n")
            f.write(f"    Max: {np.max(composite_scores):.3f}\n\n")
        
        if region_scores:
            f.write(f"  5-Region Score Analysis:\n")
            for region, scores in region_scores.items():
                f.write(f"    {region}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}\n")
        
        f.write("\nDataset is ready for MMAction2 training with enhanced features!\n")
    
    print(f"Analysis report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert enhanced annotations to MMAction2 format')
    parser.add_argument('--input-dir', type=str, 
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Directory containing enhanced annotation pkl files')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Output directory for MMAction2 format files')
    parser.add_argument('--pattern', type=str,
                       default='*_enhanced_stgcn_annotation.pkl',
                       help='Pattern for annotation files')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--no-stratify', action='store_true',
                       help='Disable stratified sampling by label')
    
    args = parser.parse_args()
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("Enhanced Annotation Format Converter")
    print("=" * 40)
    
    # 1. 개별 annotation 파일들 로드
    print("\n1. Loading individual annotation files...")
    annotations = load_individual_annotations(args.input_dir, args.pattern)
    
    if not annotations:
        print("No valid annotations found!")
        return
    
    # 2. 검증 및 정제
    print("\n2. Validating and cleaning annotations...")
    cleaned_annotations, stats = validate_and_clean_annotations(annotations)
    
    # 3. Train/Val/Test split
    print("\n3. Creating train/val/test splits...")
    train_annotations, val_annotations, test_annotations = create_train_val_test_split(
        cleaned_annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by_label=not args.no_stratify
    )
    
    # 4. MMAction2 형태로 저장
    print("\n4. Saving in MMAction2 format...")
    
    train_path = os.path.join(args.output_dir, 'rwf2000_enhanced_train.pkl')
    val_path = os.path.join(args.output_dir, 'rwf2000_enhanced_val.pkl')
    test_path = os.path.join(args.output_dir, 'rwf2000_enhanced_test.pkl')
    
    save_mmaction_format(train_annotations, train_path, 'train')
    save_mmaction_format(val_annotations, val_path, 'val')
    save_mmaction_format(test_annotations, test_path, 'test')
    
    # 5. 분석 리포트 생성
    print("\n5. Generating analysis report...")
    generate_analysis_report(cleaned_annotations, stats, args.output_dir)
    
    print("\n" + "=" * 40)
    
    if stats['valid_videos'] > 0:
        print(" Conversion completed successfully!")
        print(f"Training file: {train_path}")
        print(f"Validation file: {val_path}")
        print(f"Test file: {test_path}")
        print("\nReady for MMAction2 training with:")
        print("  python tools/train.py configs/skeleton/stgcnpp/stgcnpp_enhanced_fight_detection.py")
    else:
        print(" Conversion failed: No valid annotations found!")
        print("\n Troubleshooting suggestions:")
        print("1. Check if enhanced annotation files exist in the input directory")
        print("2. Verify the file naming pattern matches '*_enhanced_stgcn_annotation.pkl'")
        print("3. Run enhanced_rtmo_bytetrack_pose_extraction.py first to generate annotations")
        print("4. Check file permissions and paths")
        
        print(f"\n Input directory checked: {args.input_dir}")
        print(f" Pattern used: {args.pattern}")
        
        # Additional debugging info
        if os.path.exists(args.input_dir):
            print(f" Input directory exists")
            dir_contents = os.listdir(args.input_dir)
            print(f" Directory contains {len(dir_contents)} items")
        else:
            print(f" Input directory does not exist: {args.input_dir}")


if __name__ == '__main__':
    main()