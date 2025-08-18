#!/usr/bin/env python3
"""
RTMO 포즈 데이터 분석 도구

매 프레임에서 탐지된 사람 수와 키포인트 신뢰도를 분석합니다.
"""

import pickle
import sys
import numpy as np
from pathlib import Path

# recognizer 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_pose_data(poses_file_path: str):
    """포즈 데이터 분석"""
    
    # 포즈 데이터 로드
    try:
        with open(poses_file_path, 'rb') as f:
            frame_poses_results = pickle.load(f)
        print(f"로드된 총 프레임 수: {len(frame_poses_results)}")
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return
    
    # 프레임별 분석
    total_frames = len(frame_poses_results)
    frames_with_persons = 0
    frames_without_persons = 0
    person_count_stats = {}
    keypoint_confidence_stats = []
    
    print("\n=== 프레임별 탐지된 사람 수 분석 ===")
    print("프레임번호 | 탐지된 사람 수 | 키포인트 신뢰도 (평균)")
    print("-" * 60)
    
    for i, frame_poses in enumerate(frame_poses_results[:50]):  # 처음 50프레임만 출력
        person_count = len(frame_poses.persons) if frame_poses.persons else 0
        
        if person_count > 0:
            frames_with_persons += 1
            
            # 키포인트 신뢰도 계산
            avg_confidences = []
            for person in frame_poses.persons:
                if hasattr(person, 'keypoints') and len(person.keypoints) > 0:
                    confidences = [kpt[2] for kpt in person.keypoints if len(kpt) >= 3]
                    if confidences:
                        avg_confidences.append(np.mean(confidences))
            
            frame_avg_conf = np.mean(avg_confidences) if avg_confidences else 0.0
            keypoint_confidence_stats.append(frame_avg_conf)
            
            print(f"  {i:3d}      |      {person_count:2d}       |        {frame_avg_conf:.3f}")
        else:
            frames_without_persons += 1
            print(f"  {i:3d}      |       0       |        N/A")
        
        # 사람 수 통계
        if person_count in person_count_stats:
            person_count_stats[person_count] += 1
        else:
            person_count_stats[person_count] = 1
    
    if total_frames > 50:
        print(f"... (총 {total_frames}프레임 중 처음 50프레임만 표시)")
        
        # 나머지 프레임들도 통계에 포함
        for i, frame_poses in enumerate(frame_poses_results[50:], 50):
            person_count = len(frame_poses.persons) if frame_poses.persons else 0
            
            if person_count > 0:
                frames_with_persons += 1
                
                # 키포인트 신뢰도 계산
                avg_confidences = []
                for person in frame_poses.persons:
                    if hasattr(person, 'keypoints') and len(person.keypoints) > 0:
                        confidences = [kpt[2] for kpt in person.keypoints if len(kpt) >= 3]
                        if confidences:
                            avg_confidences.append(np.mean(confidences))
                
                frame_avg_conf = np.mean(avg_confidences) if avg_confidences else 0.0
                keypoint_confidence_stats.append(frame_avg_conf)
            else:
                frames_without_persons += 1
            
            # 사람 수 통계
            if person_count in person_count_stats:
                person_count_stats[person_count] += 1
            else:
                person_count_stats[person_count] = 1
    
    print("\n=== 전체 통계 ===")
    print(f"총 프레임 수: {total_frames}")
    print(f"사람이 탐지된 프레임: {frames_with_persons} ({frames_with_persons/total_frames*100:.1f}%)")
    print(f"사람이 탐지되지 않은 프레임: {frames_without_persons} ({frames_without_persons/total_frames*100:.1f}%)")
    
    print("\n=== 프레임별 사람 수 분포 ===")
    for person_count in sorted(person_count_stats.keys()):
        count = person_count_stats[person_count]
        percentage = count / total_frames * 100
        print(f"{person_count}명: {count}프레임 ({percentage:.1f}%)")
    
    if keypoint_confidence_stats:
        print(f"\n=== 키포인트 신뢰도 통계 ===")
        print(f"평균 신뢰도: {np.mean(keypoint_confidence_stats):.3f}")
        print(f"최소 신뢰도: {np.min(keypoint_confidence_stats):.3f}")
        print(f"최대 신뢰도: {np.max(keypoint_confidence_stats):.3f}")
    
    # 연속된 빈 프레임 구간 분석
    print("\n=== 연속된 빈 프레임 구간 분석 ===")
    empty_sequences = []
    current_empty_start = None
    
    for i, frame_poses in enumerate(frame_poses_results):
        person_count = len(frame_poses.persons) if frame_poses.persons else 0
        
        if person_count == 0:
            if current_empty_start is None:
                current_empty_start = i
        else:
            if current_empty_start is not None:
                empty_sequences.append((current_empty_start, i - 1))
                current_empty_start = None
    
    # 마지막 시퀀스 처리
    if current_empty_start is not None:
        empty_sequences.append((current_empty_start, total_frames - 1))
    
    if empty_sequences:
        print("연속된 빈 프레임 구간들:")
        for start, end in empty_sequences:
            length = end - start + 1
            print(f"  프레임 {start}-{end}: {length}프레임")
    else:
        print("연속된 빈 프레임 구간 없음")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python analyze_pose_data.py <poses_file_path>")
        sys.exit(1)
    
    poses_file = sys.argv[1]
    if not Path(poses_file).exists():
        print(f"파일이 존재하지 않습니다: {poses_file}")
        sys.exit(1)
    
    analyze_pose_data(poses_file)