#!/usr/bin/env python3
"""
트래킹 ID 일관성 분석 도구
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict, Counter

# recognizer 모듈 경로 추가
sys.path.insert(0, '/workspace/recognizer')

def analyze_tracking_ids(poses_file_path: str):
    """트래킹 ID 일관성 분석"""
    
    with open(poses_file_path, 'rb') as f:
        frame_poses_results = pickle.load(f)
    
    print(f"로드된 총 프레임 수: {len(frame_poses_results)}")
    print()
    
    # ID별 등장 프레임 추적
    id_frames = defaultdict(list)
    frame_id_count = []
    
    for i, frame_poses in enumerate(frame_poses_results):
        frame_ids = []
        if frame_poses and hasattr(frame_poses, 'persons') and frame_poses.persons:
            for person in frame_poses.persons:
                if hasattr(person, 'track_id') and person.track_id is not None:
                    track_id = person.track_id
                    id_frames[track_id].append(i)
                    frame_ids.append(track_id)
        
        frame_id_count.append((i, len(frame_ids), frame_ids))
    
    print("=== ID 일관성 분석 ===")
    
    # 총 ID 개수
    total_ids = len(id_frames)
    print(f"총 사용된 트래킹 ID 수: {total_ids}")
    
    # ID별 지속 시간 분석
    id_durations = {}
    for track_id, frames in id_frames.items():
        if frames:
            duration = max(frames) - min(frames) + 1
            active_frames = len(frames)
            consistency = active_frames / duration if duration > 0 else 0
            id_durations[track_id] = {
                'start': min(frames),
                'end': max(frames),
                'duration': duration,
                'active_frames': active_frames,
                'consistency': consistency
            }
    
    # 지속 시간별 ID 분류
    long_term_ids = []  # 30프레임 이상
    medium_term_ids = [] # 10-29프레임
    short_term_ids = []  # 1-9프레임
    
    for track_id, stats in id_durations.items():
        if stats['duration'] >= 30:
            long_term_ids.append((track_id, stats))
        elif stats['duration'] >= 10:
            medium_term_ids.append((track_id, stats))
        else:
            short_term_ids.append((track_id, stats))
    
    print(f"\n=== ID 지속성 분석 ===")
    print(f"장기 ID (30+ 프레임): {len(long_term_ids)}개")
    print(f"중기 ID (10-29 프레임): {len(medium_term_ids)}개")
    print(f"단기 ID (1-9 프레임): {len(short_term_ids)}개")
    
    # 장기 ID 상세 분석
    if long_term_ids:
        print(f"\n=== 장기 ID 상세 정보 ===")
        for track_id, stats in sorted(long_term_ids, key=lambda x: x[1]['duration'], reverse=True):
            print(f"ID {track_id}: {stats['start']}-{stats['end']}프레임 "
                  f"(지속:{stats['duration']}, 활성:{stats['active_frames']}, "
                  f"일관성:{stats['consistency']:.2f})")
    
    # 주요 ID (가장 오래 지속된 2개) 추적
    if len(long_term_ids) >= 2:
        main_ids = sorted(long_term_ids, key=lambda x: x[1]['duration'], reverse=True)[:2]
        print(f"\n=== 주요 ID 추적 패턴 ===")
        
        for track_id, stats in main_ids:
            frames = id_frames[track_id]
            print(f"\nID {track_id} 등장 패턴:")
            
            # 연속 구간 찾기
            if frames:
                consecutive_groups = []
                current_group = [frames[0]]
                
                for i in range(1, len(frames)):
                    if frames[i] == frames[i-1] + 1:
                        current_group.append(frames[i])
                    else:
                        consecutive_groups.append(current_group)
                        current_group = [frames[i]]
                consecutive_groups.append(current_group)
                
                print(f"  연속 구간 수: {len(consecutive_groups)}")
                for j, group in enumerate(consecutive_groups):
                    if len(group) > 1:
                        print(f"  구간 {j+1}: {group[0]}-{group[-1]}프레임 ({len(group)}프레임)")
                    else:
                        print(f"  구간 {j+1}: {group[0]}프레임 (1프레임)")
    
    # 프레임별 ID 변화 분석
    print(f"\n=== 프레임별 ID 변화 분석 (처음 30프레임) ===")
    print("프레임 | ID 개수 | 트래킹 ID들")
    print("-" * 40)
    
    for i, count, ids in frame_id_count[:30]:
        ids_str = str(sorted(ids)) if ids else "[]"
        print(f"{i:6} | {count:7} | {ids_str}")
    
    # ID 전환 횟수 계산
    id_changes = 0
    prev_ids = set()
    
    for i, count, ids in frame_id_count:
        current_ids = set(ids)
        if i > 0 and prev_ids != current_ids:
            id_changes += 1
        prev_ids = current_ids
    
    print(f"\n=== ID 안정성 지표 ===")
    print(f"총 ID 변화 횟수: {id_changes}")
    print(f"평균 ID 변화 간격: {len(frame_poses_results)/id_changes:.1f}프레임" if id_changes > 0 else "ID 변화 없음")
    print(f"ID 안정성 점수: {(1 - id_changes/len(frame_poses_results))*100:.1f}%" if len(frame_poses_results) > 0 else "0%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python analyze_tracking_ids.py <poses_file.pkl>")
        sys.exit(1)
    
    poses_file = sys.argv[1]
    if not Path(poses_file).exists():
        print(f"파일을 찾을 수 없습니다: {poses_file}")
        sys.exit(1)
    
    analyze_tracking_ids(poses_file)