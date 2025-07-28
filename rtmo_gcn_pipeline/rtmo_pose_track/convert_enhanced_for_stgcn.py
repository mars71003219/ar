#!/usr/bin/env python3
"""
Enhanced annotation을 STGCN용 MMAction2 형식으로 변환
Fight-prioritized ranking 정보를 모두 유지하면서 STGCN 파이프라인과 호환
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

def process_enhanced_annotation_for_stgcn(input_dir, output_dir, train_split=0.7, val_split=0.2):
    """Enhanced annotation을 STGCN용 MMAction2 형식으로 통합 변환"""
    
    print("=== Enhanced Annotation to STGCN Format Conversion ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 모든 enhanced annotation 파일 수집
    all_annotations = []
    fight_files = []
    nonfight_files = []
    
    # Fight 파일들 수집
    fight_dir = os.path.join(input_dir, "train/Fight")
    if os.path.exists(fight_dir):
        for file in os.listdir(fight_dir):
            if file.endswith('_enhanced_stgcn_annotation.pkl'):
                fight_files.append(os.path.join(fight_dir, file))
    
    # NonFight 파일들 수집
    nonfight_dir = os.path.join(input_dir, "train/NonFight")
    if os.path.exists(nonfight_dir):
        for file in os.listdir(nonfight_dir):
            if file.endswith('_enhanced_stgcn_annotation.pkl'):
                nonfight_files.append(os.path.join(nonfight_dir, file))
    
    print(f"Found {len(fight_files)} Fight files")
    print(f"Found {len(nonfight_files)} NonFight files")
    
    # 모든 파일 처리
    all_files = fight_files + nonfight_files
    converted_samples = []
    skipped_count = 0
    
    for file_path in tqdm(all_files, desc="Processing enhanced annotations"):
        try:
            with open(file_path, 'rb') as f:
                enhanced_data = pickle.load(f)
            
            # STGCN용 샘플로 변환
            stgcn_sample = convert_single_enhanced_to_stgcn(enhanced_data, file_path)
            
            if stgcn_sample is not None:
                converted_samples.append(stgcn_sample)
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"\nConversion completed:")
    print(f"- Total files processed: {len(all_files)}")
    print(f"- Successfully converted: {len(converted_samples)}")
    print(f"- Skipped: {skipped_count}")
    
    # 라벨 분포 확인
    label_counts = defaultdict(int)
    for sample in converted_samples:
        label_counts[sample['label']] += 1
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        label_name = "Fight" if label == 1 else "NonFight"
        print(f"  - {label_name} ({label}): {count}")
    
    # Train/Val/Test 분할
    np.random.seed(42)  # 재현성을 위한 시드 고정
    np.random.shuffle(converted_samples)
    
    total_samples = len(converted_samples)
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    
    train_samples = converted_samples[:train_size]
    val_samples = converted_samples[train_size:train_size + val_size]
    test_samples = converted_samples[train_size + val_size:]
    
    # 각 분할별 라벨 분포 확인
    for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
        split_labels = defaultdict(int)
        for sample in samples:
            split_labels[sample['label']] += 1
        
        print(f"\n{split_name} split:")
        print(f"  - Total: {len(samples)}")
        for label, count in split_labels.items():
            label_name = "Fight" if label == 1 else "NonFight"
            print(f"  - {label_name}: {count}")
    
    # 저장
    output_files = [
        (train_samples, 'rwf2000_enhanced_train.pkl'),
        (val_samples, 'rwf2000_enhanced_val.pkl'),
        (test_samples, 'rwf2000_enhanced_test.pkl')
    ]
    
    for samples, filename in output_files:
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved {len(samples)} samples to {output_path}")
    
    return len(converted_samples), skipped_count

def convert_single_enhanced_to_stgcn(enhanced_data, file_path):
    """단일 Enhanced annotation을 STGCN용 형식으로 변환"""
    
    try:
        # 기본 정보 추출
        video_info = enhanced_data['video_info']
        persons_dict = enhanced_data['persons']
        score_weights = enhanced_data.get('score_weights', {})
        
        if not persons_dict:
            return None
        
        # Fight-prioritized 순서로 모든 person 정렬
        person_rankings = []
        
        for person_key, person_data in persons_dict.items():
            if person_key.startswith('person_'):
                enhanced_info = person_data.get('enhanced_info', {})
                composite_score = enhanced_info.get('composite_score', 0.0)
                
                person_rankings.append({
                    'person_key': person_key,
                    'person_data': person_data,
                    'composite_score': composite_score,
                    'enhanced_info': enhanced_info
                })
        
        # Composite score 기준으로 내림차순 정렬 (Fight-prioritized)
        person_rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        if not person_rankings:
            return None
        
        # 모든 person의 keypoint 데이터 수집 (ranking 순서 유지)
        all_keypoints = []
        all_scores = []
        person_metadata = []
        
        for ranking_info in person_rankings[:5]:  # 최대 5명까지 보존
            person_data = ranking_info['person_data']
            annotation = person_data['annotation']
            
            keypoints = annotation['keypoint']    # 원본 shape 확인 필요
            scores = annotation['keypoint_score'] # 원본 shape 확인 필요
            
            # Enhanced annotation은 (1, T, V, C) 형태이므로 squeeze 필요
            if keypoints.ndim == 4 and keypoints.shape[0] == 1:
                keypoints = keypoints.squeeze(0)  # (T, V, C)로 변환
            if scores.ndim == 3 and scores.shape[0] == 1:
                scores = scores.squeeze(0)  # (T, V)로 변환
            
            all_keypoints.append(keypoints)
            all_scores.append(scores)
            
            # Enhanced metadata 보존
            person_metadata.append({
                'person_key': ranking_info['person_key'],
                'composite_score': ranking_info['composite_score'],
                'enhanced_info': ranking_info['enhanced_info']
            })
        
        # NumPy 배열로 변환 - STGCN이 기대하는 (M, T, V, C) 형태
        keypoints_array = np.array(all_keypoints)      # (M, T, V, C)
        scores_array = np.array(all_scores)            # (M, T, V)
        
        # STGCN용 샘플 구성
        stgcn_sample = {
            # MMAction2 PoseDataset 필수 필드
            'frame_dir': video_info['frame_dir'],
            'total_frames': video_info['total_frames'],
            'img_shape': video_info['img_shape'],
            'original_shape': video_info['original_shape'],
            'label': video_info['label'],
            
            # STGCN 키포인트 데이터 (다중 person 지원)
            'keypoint': keypoints_array,               # (M, T, V, C)
            'keypoint_score': scores_array,            # (M, T, V)
            
            # Enhanced 메타데이터 (Fight-prioritized 정보 보존)
            'enhanced_metadata': {
                'total_persons': len(person_rankings),
                'selected_persons': len(all_keypoints),
                'score_weights': score_weights,
                'person_rankings': person_metadata,
                'source_file': os.path.basename(file_path),
                
                # 최고 ranking person의 주요 지표
                'best_person_score': person_rankings[0]['composite_score'],
                'best_person_movement': person_rankings[0]['enhanced_info'].get('movement_intensity', 0.0),
                'best_person_interaction': person_rankings[0]['enhanced_info'].get('interaction_score', 0.0),
                'best_person_position': person_rankings[0]['enhanced_info'].get('position_5region_score', 0.0),
            }
        }
        
        return stgcn_sample
        
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert Enhanced annotations to STGCN-compatible MMAction2 format')
    parser.add_argument('--input-dir', type=str, 
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output/RWF-2000',
                       help='Directory containing enhanced annotation files')
    parser.add_argument('--output-dir', type=str,
                       default='/workspace/rtmo_gcn_pipeline/rtmo_pose_track/output',
                       help='Output directory for STGCN-compatible files')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 변환 실행
    converted_count, skipped_count = process_enhanced_annotation_for_stgcn(
        args.input_dir, 
        args.output_dir, 
        args.train_split, 
        args.val_split
    )
    
    print(f"\n=== Conversion Summary ===")
    print(f"Total converted: {converted_count}")
    print(f"Total skipped: {skipped_count}")
    print(f"Success rate: {converted_count/(converted_count+skipped_count)*100:.1f}%")
    
    print(f"\n=== Usage Instructions ===")
    print("The converted files are ready for STGCN training with MMAction2:")
    print(f"  ann_file_train = '{args.output_dir}/rwf2000_enhanced_train.pkl'")
    print(f"  ann_file_val = '{args.output_dir}/rwf2000_enhanced_val.pkl'")
    print(f"  ann_file_test = '{args.output_dir}/rwf2000_enhanced_test.pkl'")
    print()
    print("Features preserved:")
    print("  ✅ Fight-prioritized person ranking")
    print("  ✅ Multi-person keypoint data (up to 5 persons)")
    print("  ✅ Enhanced metadata (composite scores, movement, interaction)")
    print("  ✅ STGCN pipeline compatibility")
    print("  ✅ Flexible num_person configuration (1, 2, 3, 4, 5)")
    print()
    print("STGCN config settings:")
    print("  dataset_type = 'PoseDataset'")
    print("  dict(type='FormatGCNInput', num_person=2)  # or 1,3,4,5")

if __name__ == "__main__":
    main()