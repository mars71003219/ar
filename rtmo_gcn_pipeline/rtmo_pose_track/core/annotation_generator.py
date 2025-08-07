#!/usr/bin/env python3
"""
어노테이션 생성 모듈 - 트래킹된 포즈 데이터에서 MMAction2 호환 어노테이션 생성
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from .scoring_system import EnhancedFightInvolvementScorer


def create_enhanced_annotation(pose_results, video_path, pose_model, 
                             min_track_length=10, quality_threshold=0.3, weights=None):
    """개선된 어노테이션 생성 (모든 객체 랭킹)"""
    if not pose_results:
        return None, "No pose results"
    
    # 1. 모든 Track ID 수집 및 기본 필터링
    all_tracks_data = collect_all_tracks_data(pose_results, min_track_length)
    
    if len(all_tracks_data) == 0:
        return None, f"No tracks with minimum length {min_track_length}"
    
    # 2. 이미지 크기 추출
    img_shape = pose_results[0].img_shape if hasattr(pose_results[0], 'img_shape') else (720, 1280)
    
    # 3. 개선된 점수 계산기 초기화 (가중치 전달)
    scorer = EnhancedFightInvolvementScorer(img_shape, enable_adaptive=True, weights=weights)
    
    # 4. 각 Track ID에 대해 복합 점수 계산
    scored_tracks = []
    for track_id, track_data in all_tracks_data.items():
        score_info = scorer.calculate_enhanced_fight_score(track_data, all_tracks_data)
        scored_tracks.append((track_id, score_info, track_data))
    
    # 5. 점수순 정렬 (내림차순)
    scored_tracks.sort(key=lambda x: x[1]['composite_score'], reverse=True)
    
    # 6. 품질 필터링
    quality_tracks = []
    for track_id, score_info, track_data in scored_tracks:
        track_quality = _calculate_track_quality(track_data)
        
        if track_quality >= quality_threshold:
            quality_tracks.append((track_id, score_info, track_data, track_quality))
        else:
            print(f"Track {track_id} filtered out (quality: {track_quality:.3f} < {quality_threshold})")
    
    if len(quality_tracks) == 0:
        return None, f"No tracks passed quality threshold {quality_threshold}"
    
    # 7. 어노테이션 구성 - 모든 품질 트랙 포함
    total_frames = len(pose_results)
    persons = {}
    
    for rank, (track_id, score_info, track_data, track_quality) in enumerate(quality_tracks):
        try:
            # MMAction2 호환 키포인트 시퀀스 생성
            keypoint_sequence = _create_keypoint_sequence(track_data, total_frames)
            
            # keypoint_score 시퀀스 생성 (키포인트 신뢰도)
            keypoint_score_sequence = _create_keypoint_score_sequence(track_data, total_frames)
            
            persons[str(track_id)] = {
                'keypoint': keypoint_sequence,
                'keypoint_score': keypoint_score_sequence,  
                'num_keypoints': 17,  # COCO format
                'track_id': track_id,
                'composite_score': float(score_info['composite_score']),
                'rank': rank + 1,
                'quality_score': float(track_quality),
                'num_frames': len(track_data),
                'score_breakdown': {
                    'movement': float(score_info['breakdown']['movement']),
                    'position': float(score_info['breakdown']['position']),
                    'interaction': float(score_info['breakdown']['interaction']),
                    'temporal_consistency': float(score_info['breakdown']['temporal_consistency']),
                    'persistence': float(score_info['breakdown']['persistence'])
                },
                'region_breakdown': score_info.get('region_breakdown', {})
            }
        except Exception as e:
            print(f"Error creating keypoint sequence for track {track_id}: {str(e)}")
            continue
    
    if not persons:
        return None, "Failed to create keypoint sequences"
    
    # 8. 최종 어노테이션 구성
    annotation = {
        'frame_ind': 0,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'persons': persons,
        'total_persons': len(persons),
        'total_frames': total_frames
    }
    
    print(f"Created annotation with {len(persons)} persons, top score: {quality_tracks[0][1]['composite_score']:.3f}")
    
    return annotation, "Success"


def collect_all_tracks_data(pose_results, min_track_length=10):
    """모든 트랙 데이터 수집"""
    tracks_data = defaultdict(dict)
    
    for frame_idx, pose_result in enumerate(pose_results):
        if hasattr(pose_result, 'pred_instances') and pose_result.pred_instances is not None:
            instances = pose_result.pred_instances
            
            # track_ids가 있는지 확인
            if hasattr(instances, 'track_ids') and instances.track_ids is not None:
                track_ids = instances.track_ids
                keypoints = instances.keypoints
                bboxes = instances.bboxes
                scores = instances.scores if hasattr(instances, 'scores') else None
                
                # torch tensor를 numpy로 변환
                if hasattr(track_ids, 'cpu'):
                    track_ids = track_ids.cpu().numpy()
                    keypoints = keypoints.cpu().numpy()
                    bboxes = bboxes.cpu().numpy()
                    if scores is not None:
                        scores = scores.cpu().numpy()
                
                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)
                    
                    # track_id가 -1인 경우 (매칭 실패) 무시
                    if track_id < 0:
                        continue
                    
                    track_data = {
                        'keypoints': keypoints[i],
                        'bbox': bboxes[i],
                        'score': scores[i] if scores is not None else 1.0
                    }
                    
                    tracks_data[track_id][frame_idx] = track_data
    
    # 최소 길이 필터링
    filtered_tracks = {
        track_id: data for track_id, data in tracks_data.items()
        if len(data) >= min_track_length
    }
    
    return filtered_tracks


def _calculate_track_quality(track_data):
    """트랙 품질 계산"""
    if not track_data:
        return 0.0
    
    quality_scores = []
    debug_info = []
    
    for frame_idx, frame_data in track_data.items():
        keypoints = frame_data.get('keypoints', [])
        bbox_score = frame_data.get('score', 0.0)
        
        if len(keypoints) > 0:
            try:
                kpts = np.array(keypoints)
                frame_quality = 0.0
                
                if kpts.ndim == 2 and kpts.shape[1] >= 3:
                    # 키포인트 신뢰도 (visibility) 평균
                    visibility_scores = kpts[:, 2]
                    avg_visibility = np.mean(visibility_scores)
                    frame_quality = avg_visibility * bbox_score
                elif kpts.ndim == 2 and kpts.shape[1] == 2:
                    # 2D 키포인트인 경우 bbox_score만 사용
                    frame_quality = bbox_score
                else:
                    # 다른 형태인 경우 bbox_score만 사용
                    frame_quality = bbox_score
                
                quality_scores.append(frame_quality)
                debug_info.append(f"Frame {frame_idx}: kpts_shape={kpts.shape}, bbox_score={bbox_score:.3f}, quality={frame_quality:.3f}")
                
            except Exception as e:
                print(f"Error calculating quality for frame {frame_idx}: {str(e)}")
                continue
    
    final_quality = np.mean(quality_scores) if quality_scores else 0.0
    
    # 디버깅: 품질이 낮을 때만 출력
    if final_quality < 0.1 and len(debug_info) > 0:
        print(f"Low quality track debug (final: {final_quality:.3f}):")
        for info in debug_info[:3]:  # 처음 3개 프레임만
            print(f"  {info}")
    
    return final_quality


def _create_keypoint_sequence(track_data, total_frames):
    """MMAction2 호환 키포인트 시퀀스 생성"""
    # (1, total_frames, 17, 2) 형태로 생성
    keypoint_sequence = np.zeros((1, total_frames, 17, 2), dtype=np.float32)
    
    for frame_idx, frame_data in track_data.items():
        if frame_idx < total_frames:
            keypoints = frame_data.get('keypoints', [])
            
            if len(keypoints) > 0:
                try:
                    kpts = np.array(keypoints)
                    
                    # 다양한 형태 처리
                    if kpts.ndim == 1 and len(kpts) >= 51:
                        # 평면화된 경우 (51,) -> (17, 3)
                        kpts = kpts.reshape(-1, 3)
                    elif kpts.ndim == 3:
                        # (1, 17, 3) -> (17, 3)
                        kpts = kpts.squeeze(0)
                    
                    # (17, 3) 형태인지 확인
                    if kpts.ndim == 2 and kpts.shape[0] >= 17 and kpts.shape[1] >= 2:
                        # x, y 좌표만 사용
                        keypoint_sequence[0, frame_idx, :17, 0] = kpts[:17, 0]
                        keypoint_sequence[0, frame_idx, :17, 1] = kpts[:17, 1]
                        
                except Exception as e:
                    print(f"Error processing keypoints at frame {frame_idx}: {str(e)}")
                    continue
    
    return keypoint_sequence


def _create_keypoint_score_sequence(track_data, total_frames):
    """키포인트 신뢰도 시퀀스 생성"""
    # (1, total_frames, 17) 형태로 생성
    score_sequence = np.zeros((1, total_frames, 17), dtype=np.float32)
    
    for frame_idx, frame_data in track_data.items():
        if frame_idx < total_frames:
            keypoints = frame_data.get('keypoints', [])
            
            if len(keypoints) > 0:
                try:
                    kpts = np.array(keypoints)
                    
                    # 다양한 형태 처리
                    if kpts.ndim == 1 and len(kpts) >= 51:
                        # 평면화된 경우 (51,) -> (17, 3)
                        kpts = kpts.reshape(-1, 3)
                    elif kpts.ndim == 3:
                        # (1, 17, 3) -> (17, 3)
                        kpts = kpts.squeeze(0)
                    
                    # (17, 3) 형태인지 확인하고 신뢰도 추출
                    if kpts.ndim == 2 and kpts.shape[0] >= 17 and kpts.shape[1] >= 3:
                        score_sequence[0, frame_idx, :17] = kpts[:17, 2]
                        
                except Exception as e:
                    print(f"Error processing keypoint scores at frame {frame_idx}: {str(e)}")
                    continue
    
    return score_sequence