#!/usr/bin/env python3
"""
Enhanced annotation을 표준 MMAction2 PoseDataset 형식으로 변환하는 스크립트
Fight-prioritized ranking은 유지하면서 표준 포맷으로 변환
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

def select_best_person_enhanced(persons_dict, strategy='top_score'):
    """Enhanced annotation에서 최적의 person 선택 (Fight-prioritized)"""
    
    if not persons_dict:
        return None
    
    best_person = None
    best_score = -1
    
    for person_key, person_data in persons_dict.items():
        if person_key.startswith('person_'):
            # Enhanced composite score 활용
            enhanced_info = person_data.get('enhanced_info', {})
            
            if strategy == 'top_score':
                # 전체 composite score 사용
                score = enhanced_info.get('composite_score', 0.0)
            elif strategy == 'fight_priority':
                # Fight 관련 지표 우선
                movement = enhanced_info.get('movement_intensity', 0.0)
                interaction = enhanced_info.get('interaction_score', 0.0)
                position = enhanced_info.get('position_5region_score', 0.0)
                score = movement * 0.4 + interaction * 0.4 + position * 0.2
            else:
                # 기본적으로 track quality 사용
                score = enhanced_info.get('track_quality', 0.0)
            
            if score > best_score:
                best_score = score
                best_person = person_data
    
    return best_person

def convert_enhanced_to_standard_pose_format(input_file, output_file, strategy='fight_priority'):
    """Enhanced format을 표준 PoseDataset format으로 변환"""
    
    print(f"Converting {input_file} to standard PoseDataset format...")
    
    with open(input_file, 'rb') as f:
        enhanced_data = pickle.load(f)
    
    # 표준 포맷 데이터 구조
    standard_data = []
    
    total_videos = len([k for k in enhanced_data.keys() if k != '_metadata'])
    converted_count = 0
    skipped_count = 0
    
    for video_name, video_data in tqdm(enhanced_data.items(), desc="Converting videos"):
        if video_name == '_metadata':
            continue
        
        try:
            # Video info 추출
            video_info = video_data['video_info']
            persons_dict = video_data['persons']
            
            # Fight-prioritized person 선택
            best_person = select_best_person_enhanced(persons_dict, strategy=strategy)
            
            if best_person is None:
                print(f"Warning: No valid person found for {video_name}")
                skipped_count += 1
                continue
            
            # Annotation에서 keypoint 데이터 추출
            annotation = best_person['annotation']
            keypoints = annotation['keypoint']  # (T, V, C)
            keypoint_scores = annotation['keypoint_score']  # (T, V)
            
            # 표준 포맷으로 변환
            sample = {
                'frame_dir': video_info['frame_dir'],
                'total_frames': video_info['total_frames'],
                'img_shape': video_info['img_shape'],
                'original_shape': video_info['original_shape'],
                'label': video_info['label'],
                'keypoint': keypoints,  # (T, V, C)
                'keypoint_score': keypoint_scores,  # (T, V)
                
                # Enhanced metadata 보존 (optional)
                'enhanced_metadata': {
                    'strategy_used': strategy,
                    'composite_score': best_person.get('enhanced_info', {}).get('composite_score', 0.0),
                    'movement_intensity': best_person.get('enhanced_info', {}).get('movement_intensity', 0.0),
                    'interaction_score': best_person.get('enhanced_info', {}).get('interaction_score', 0.0),
                    'original_persons_count': video_data.get('total_persons', 0),
                }
            }
            
            standard_data.append(sample)
            converted_count += 1
            
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            skipped_count += 1
            continue
    
    # 표준 포맷으로 저장
    with open(output_file, 'wb') as f:
        pickle.dump(standard_data, f)
    
    print(f"\nConversion completed:")
    print(f"  - Total videos: {total_videos}")
    print(f"  - Converted: {converted_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Output: {output_file}")
    
    # 라벨 분포 확인
    label_counts = defaultdict(int)
    for sample in standard_data:
        label_counts[sample['label']] += 1
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        label_name = "Fight" if label == 1 else "NonFight"
        print(f"  - {label_name} ({label}): {count}")
    
    return converted_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description='Convert Enhanced annotations to standard PoseDataset format')
    parser.add_argument('--input-dir', type=str, 
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Directory containing enhanced annotation files')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Output directory for standard format files')
    parser.add_argument('--strategy', type=str, 
                       choices=['top_score', 'fight_priority', 'track_quality'],
                       default='fight_priority',
                       help='Person selection strategy')
    
    args = parser.parse_args()
    
    # 입력 파일들
    input_files = [
        ('rwf2000_enhanced_train.pkl', 'rwf2000_standard_train.pkl'),
        ('rwf2000_enhanced_val.pkl', 'rwf2000_standard_val.pkl'),
        ('rwf2000_enhanced_test.pkl', 'rwf2000_standard_test.pkl')
    ]
    
    total_converted = 0
    total_skipped = 0
    
    print(f"=== Enhanced to Standard PoseDataset Conversion ===")
    print(f"Strategy: {args.strategy}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    for input_name, output_name in input_files:
        input_path = os.path.join(args.input_dir, input_name)
        output_path = os.path.join(args.output_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"Skipping {input_name} - file not found")
            continue
        
        converted, skipped = convert_enhanced_to_standard_pose_format(
            input_path, output_path, strategy=args.strategy
        )
        
        total_converted += converted
        total_skipped += skipped
        print()
    
    print(f"=== Conversion Summary ===")
    print(f"Total converted: {total_converted}")
    print(f"Total skipped: {total_skipped}")
    print(f"Strategy used: {args.strategy}")
    
    # Config 파일 업데이트 안내
    print(f"\n=== Next Steps ===")
    print(f"Update your config file to use the new annotation files:")
    print(f"  ann_file_train = '{args.output_dir}/rwf2000_standard_train.pkl'")
    print(f"  ann_file_val = '{args.output_dir}/rwf2000_standard_val.pkl'")
    print(f"  ann_file_test = '{args.output_dir}/rwf2000_standard_test.pkl'")

if __name__ == "__main__":
    main()