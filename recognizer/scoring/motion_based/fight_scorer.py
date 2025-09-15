"""
움직임 기반 복합점수 계산기 구현

기존 rtmo_gcn_pipeline의 RegionBasedPositionScorer와
EnhancedFightInvolvementScorer를 새로운 표준 인터페이스에 맞게 재구성한 버전입니다.
고정 영역보다 움직임 패턴에 중점을 둔 계산 방식입니다.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

try:
    from scoring.base import BaseScorer, PersonScores
    from utils.data_structure import FramePoses, ScoringConfig, PersonPose
except ImportError:
    from ..base import BaseScorer, PersonScores
    from ...utils.data_structure import FramePoses, ScoringConfig, PersonPose


class RegionBasedPositionScorer:
    """5영역 분할 기반 위치 점수 계산기"""

    def __init__(self, img_width, img_height):
        self.width = img_width
        self.height = img_height
        self.regions = self._define_regions()

        # 각 영역별 기본 활동성 가중치
        self.region_weights = {
            'top_left': 0.7,
            'top_right': 0.7,
            'bottom_left': 0.8,
            'bottom_right': 0.8,
            'center_overlap': 1.0
        }

    def _define_regions(self):
        """화면을 5개 영역으로 분할"""
        w, h = self.width, self.height

        # 중앙 겹침 영역 크기 (전체의 50%)
        center_w = int(w * 0.5)
        center_h = int(h * 0.5)
        center_x = (w - center_w) // 2
        center_y = (h - center_h) // 2

        regions = {
            'top_left': (0, 0, w//2, h//2),
            'top_right': (w//2, 0, w, h//2),
            'bottom_left': (0, h//2, w//2, h),
            'bottom_right': (w//2, h//2, w, h),
            'center_overlap': (center_x, center_y, center_x + center_w, center_y + center_h)
        }

        return regions

    def calculate_position_score(self, bbox_history):
        """5영역 기반 위치 점수 계산"""
        region_activities = {region: 0.0 for region in self.regions.keys()}
        total_frames = len(bbox_history)

        for bbox in bbox_history:
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2

            for region_name, (x1, y1, x2, y2) in self.regions.items():
                if x1 <= bbox_center_x <= x2 and y1 <= bbox_center_y <= y2:
                    relative_score = self._calculate_relative_position_score(
                        bbox_center_x, bbox_center_y, x1, y1, x2, y2, region_name
                    )
                    region_activities[region_name] += relative_score

        # 영역별 평균 활동도 계산
        region_scores = {}
        for region, activity in region_activities.items():
            avg_activity = activity / total_frames if total_frames > 0 else 0
            weighted_score = avg_activity * self.region_weights[region]
            region_scores[region] = weighted_score

        final_score = max(region_scores.values()) if region_scores else 0.0

        return final_score, region_scores

    def _calculate_relative_position_score(self, x, y, x1, y1, x2, y2, region_name):
        """영역 내에서의 상대적 위치 기반 점수"""
        region_width = x2 - x1
        region_height = y2 - y1

        rel_x = (x - x1) / region_width
        rel_y = (y - y1) / region_height

        if region_name == 'center_overlap':
            return self._center_region_score(rel_x, rel_y)
        else:
            return self._corner_region_score(rel_x, rel_y)

    def _corner_region_score(self, rel_x, rel_y):
        """모서리 영역에서의 점수 계산"""
        distance_from_center = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
        max_distance = np.sqrt(0.5**2 + 0.5**2)

        return 1.0 - (distance_from_center / max_distance)

    def _center_region_score(self, rel_x, rel_y):
        """중앙 겹침 영역에서의 점수 계산"""
        edge_distance = min(rel_x, 1-rel_x, rel_y, 1-rel_y)
        return 0.8 + 0.2 * (edge_distance / 0.5)


class AdaptiveRegionImportance:
    """비디오별로 적응적으로 영역 중요도를 학습하는 시스템"""

    def __init__(self, convergence_threshold=0.05):
        self.convergence_threshold = convergence_threshold
        self.region_importance_history = []

    def calculate_video_specific_importance(self, all_tracks_data, img_shape, iterations=5):
        """특정 비디오에 대해 반복적으로 영역 중요도 학습"""
        region_scorer = RegionBasedPositionScorer(img_shape[1], img_shape[0])

        current_weights = region_scorer.region_weights.copy()

        for iteration in range(iterations):
            track_scores = {}
            for track_id, track_data in all_tracks_data.items():
                bbox_history = [data['bbox'] for data in track_data.values()]

                region_scorer.region_weights = current_weights
                position_score, region_breakdown = region_scorer.calculate_position_score(bbox_history)

                movement_score = self._calculate_movement_intensity(track_data)

                combined_score = position_score * 0.6 + movement_score * 0.4
                track_scores[track_id] = {
                    'combined_score': combined_score,
                    'region_breakdown': region_breakdown,
                    'movement_score': movement_score
                }

            top_tracks = self._get_top_scoring_tracks(track_scores, top_k=5)
            new_weights = self._update_weights_from_top_tracks(
                top_tracks, current_weights
            )

            weight_change = sum(abs(new_weights[k] - current_weights[k])
                              for k in current_weights.keys())

            if weight_change < self.convergence_threshold:
                break

            current_weights = new_weights
            self.region_importance_history.append(current_weights.copy())

        return current_weights, track_scores

    def _calculate_movement_intensity(self, track_data):
        """움직임 강도 계산"""
        if len(track_data) < 2:
            return 0.0

        frame_indices = sorted(track_data.keys())
        intensities = []

        for i in range(1, len(frame_indices)):
            prev_frame = frame_indices[i-1]
            curr_frame = frame_indices[i]

            if 'keypoints' in track_data[prev_frame] and 'keypoints' in track_data[curr_frame]:
                try:
                    prev_kpts = np.array(track_data[prev_frame]['keypoints'])
                    curr_kpts = np.array(track_data[curr_frame]['keypoints'])

                    # 배열 크기 확인
                    if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                        # 키포인트 차원에 따라 다르게 처리
                        if prev_kpts.ndim == 2 and prev_kpts.shape[1] >= 2:
                            # (17, 2) 또는 (17, 3) 형태 - 각 키포인트의 x,y 좌표만 사용
                            movement = np.linalg.norm(curr_kpts[:, :2] - prev_kpts[:, :2], axis=1)
                        else:
                            # 다른 형태의 경우 기존 방식
                            movement = np.linalg.norm(curr_kpts - prev_kpts, axis=-1)

                        if movement.size > 0:
                            intensities.append(float(np.mean(movement)))
                except Exception as e:
                    print(f"Error calculating movement intensity: {str(e)}")
                    continue

        return np.mean(intensities) if intensities else 0.0

    def _get_top_scoring_tracks(self, track_scores, top_k=5):
        """상위 K개 트랙 선별"""
        sorted_tracks = sorted(
            track_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        return dict(sorted_tracks[:top_k])

    def _update_weights_from_top_tracks(self, top_tracks, current_weights, alpha=0.3):
        """상위 트랙들의 영역 분포를 기반으로 가중치 업데이트"""
        region_usage = defaultdict(float)
        total_score = 0

        for track_id, track_info in top_tracks.items():
            track_score = track_info['combined_score']
            region_breakdown = track_info['region_breakdown']

            for region, region_score in region_breakdown.items():
                region_usage[region] += region_score * track_score
            total_score += track_score

        if total_score > 0:
            for region in region_usage:
                region_usage[region] /= total_score

        new_weights = {}
        for region, current_weight in current_weights.items():
            observed_importance = region_usage.get(region, 0)
            new_weights[region] = (1 - alpha) * current_weight + alpha * observed_importance

        return new_weights


class EnhancedFightInvolvementScorer:
    """개선된 싸움 참여도 점수 계산기"""

    def __init__(self, img_shape, enable_adaptive=True, weights=None):
        self.img_shape = img_shape
        self.enable_adaptive = enable_adaptive

        self.position_scorer = RegionBasedPositionScorer(img_shape[1], img_shape[0])

        if enable_adaptive:
            self.adaptive_analyzer = AdaptiveRegionImportance()

        # 가중치 설정 - 설정파일에서 4개 값만 사용 (persistence는 제거)
        if weights and len(weights) == 4:
            self.weights = {
                'movement': weights[0],
                'interaction': weights[1],
                'position': weights[2],
                'temporal_consistency': weights[3]
            }
        else:
            # 기본값: movement_based 설정
            self.weights = {
                'movement': 0.4,
                'interaction': 0.4,
                'position': 0.1,
                'temporal_consistency': 0.1
            }

    def calculate_enhanced_fight_score(self, track_data, all_tracks_data=None):
        """개선된 복합 점수 계산"""
        # 1. 움직임 강도 점수
        movement_score = self._calculate_movement_intensity(track_data)

        # 2. 개선된 5영역 기반 위치 점수
        bbox_history = [data['bbox'] for data in track_data.values()]

        if self.enable_adaptive and all_tracks_data:
            adaptive_weights, _ = self.adaptive_analyzer.calculate_video_specific_importance(
                all_tracks_data, self.img_shape
            )
            self.position_scorer.region_weights = adaptive_weights

        position_score, region_breakdown = self.position_scorer.calculate_position_score(bbox_history)

        # 3. 상호작용 점수
        interaction_score = self._calculate_enhanced_interaction(track_data, all_tracks_data)

        # 4. 시간적 일관성 점수
        temporal_consistency = self._calculate_temporal_consistency(track_data)

        # 5. 고립 상태 확인 및 점수 조정
        isolation_penalty = self._calculate_isolation_penalty(track_data, all_tracks_data)

        # 최종 가중 점수 계산 (persistence 제거)
        composite_score = (
            movement_score * self.weights['movement'] +
            position_score * self.weights['position'] +
            interaction_score * self.weights['interaction'] +
            temporal_consistency * self.weights['temporal_consistency']
        )

        # 고립된 객체의 경우 최소 점수로 조정
        if isolation_penalty > 0:
            composite_score = min(composite_score, 0.1)  # 최소 점수를 0.1로 설정

        return {
            'composite_score': composite_score,
            'breakdown': {
                'movement': movement_score,
                'position': position_score,
                'interaction': interaction_score,
                'temporal_consistency': temporal_consistency,
                'isolation_penalty': isolation_penalty
            },
            'region_breakdown': region_breakdown
        }

    def _calculate_movement_intensity(self, track_data):
        """움직임 강도 계산"""
        if len(track_data) < 2:
            return 0.0

        frame_indices = sorted(track_data.keys())
        intensities = []

        for i in range(1, len(frame_indices)):
            prev_frame = frame_indices[i-1]
            curr_frame = frame_indices[i]

            if 'keypoints' in track_data[prev_frame] and 'keypoints' in track_data[curr_frame]:
                try:
                    prev_kpts = np.array(track_data[prev_frame]['keypoints'])
                    curr_kpts = np.array(track_data[curr_frame]['keypoints'])

                    # 배열 크기 확인
                    if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                        # 키포인트 차원에 따라 다르게 처리
                        if prev_kpts.ndim == 2 and prev_kpts.shape[1] >= 2:
                            # (17, 2) 또는 (17, 3) 형태 - 각 키포인트의 x,y 좌표만 사용
                            movement = np.linalg.norm(curr_kpts[:, :2] - prev_kpts[:, :2], axis=1)
                        else:
                            # 다른 형태의 경우 기존 방식
                            movement = np.linalg.norm(curr_kpts - prev_kpts, axis=-1)

                        if movement.size > 0:
                            threshold = movement.mean() + 2*movement.std()
                            rapid_movement = np.sum(movement > threshold)
                            intensities.append(float(rapid_movement / len(movement)))
                except Exception as e:
                    print(f"Error calculating rapid movement: {str(e)}")
                    continue

        return np.mean(intensities) if intensities else 0.0

    def _calculate_enhanced_interaction(self, track_data, all_tracks_data):
        """개선된 상호작용 점수 계산"""
        if not all_tracks_data or len(all_tracks_data) <= 1:
            return 0.0

        interaction_intensity = 0.0
        interaction_count = 0

        current_track_id = None
        for tid, tdata in all_tracks_data.items():
            # 딕셔너리의 메모리 주소로 비교 (같은 객체인지 확인)
            if id(tdata) == id(track_data):
                current_track_id = tid
                break

        if current_track_id is None:
            return 0.0

        for other_id, other_data in all_tracks_data.items():
            if other_id == current_track_id:
                continue

            common_frames = set(track_data.keys()) & set(other_data.keys())

            for frame in common_frames:
                distance_score = self._calculate_proximity_interaction(
                    track_data[frame]['bbox'],
                    other_data[frame]['bbox']
                )

                sync_score = self._calculate_movement_synchronization(
                    track_data, other_data, frame
                )

                interaction_intensity += (distance_score + sync_score) / 2
                interaction_count += 1

        return interaction_intensity / interaction_count if interaction_count > 0 else 0.0

    def _calculate_proximity_interaction(self, bbox1, bbox2):
        """근접도 기반 상호작용 점수"""
        center1 = np.array([(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2])
        center2 = np.array([(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2])

        distance = np.linalg.norm(center1 - center2)

        # 바운딩박스 크기 기반 정규화
        size1 = np.sqrt((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        size2 = np.sqrt((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        avg_size = (size1 + size2) / 2

        normalized_distance = distance / avg_size if avg_size > 0 else float('inf')

        # 거리가 가까울수록 높은 점수 (임계값 3.0)
        return max(0, 1.0 - normalized_distance / 3.0)

    def _calculate_movement_synchronization(self, track_data1, track_data2, current_frame):
        """움직임 동기화 점수"""
        frame_indices1 = sorted(track_data1.keys())
        frame_indices2 = sorted(track_data2.keys())

        if current_frame not in frame_indices1 or current_frame not in frame_indices2:
            return 0.0

        idx1 = frame_indices1.index(current_frame)
        idx2 = frame_indices2.index(current_frame)

        if idx1 == 0 or idx2 == 0:
            return 0.0

        prev_frame1 = frame_indices1[idx1 - 1]
        prev_frame2 = frame_indices2[idx2 - 1]

        if prev_frame1 not in track_data1 or prev_frame2 not in track_data2:
            return 0.0

        # 움직임 벡터 계산
        movement1 = self._get_movement_vector(track_data1[prev_frame1], track_data1[current_frame])
        movement2 = self._get_movement_vector(track_data2[prev_frame2], track_data2[current_frame])

        return self._calculate_vector_similarity(movement1, movement2)

    def _calculate_temporal_consistency(self, track_data):
        """시간적 일관성 점수"""
        if len(track_data) < 3:
            return 1.0

        frame_indices = sorted(track_data.keys())
        consistency_scores = []

        for i in range(2, len(frame_indices)):
            prev_prev = frame_indices[i-2]
            prev = frame_indices[i-1]
            curr = frame_indices[i]

            movement1 = self._get_movement_vector(
                track_data[prev_prev], track_data[prev]
            )

            movement2 = self._get_movement_vector(
                track_data[prev], track_data[curr]
            )

            consistency = self._calculate_vector_similarity(movement1, movement2)
            consistency_scores.append(consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _get_movement_vector(self, prev_data, curr_data):
        """움직임 벡터 계산"""
        try:
            if 'keypoints' in prev_data and 'keypoints' in curr_data:
                prev_kpts = np.array(prev_data['keypoints'])
                curr_kpts = np.array(curr_data['keypoints'])

                # 배열 크기 확인
                if prev_kpts.shape == curr_kpts.shape and prev_kpts.size > 0:
                    # 키포인트 차원에 따라 다르게 처리
                    if prev_kpts.ndim == 2 and prev_kpts.shape[1] >= 2:
                        # (17, 2) 또는 (17, 3) 형태 - x,y 좌표만 사용
                        movement = curr_kpts[:, :2] - prev_kpts[:, :2]
                        return np.mean(movement, axis=0)  # 평균 움직임 벡터
                    else:
                        # 다른 형태의 경우 기존 방식
                        movement = curr_kpts - prev_kpts
                        return np.mean(movement, axis=0)  # 평균 움직임 벡터

            # 바운딩박스 중심점 기반 움직임
            if 'bbox' in prev_data and 'bbox' in curr_data:
                prev_center = np.array([(prev_data['bbox'][0] + prev_data['bbox'][2])/2,
                                       (prev_data['bbox'][1] + prev_data['bbox'][3])/2])
                curr_center = np.array([(curr_data['bbox'][0] + curr_data['bbox'][2])/2,
                                       (curr_data['bbox'][1] + curr_data['bbox'][3])/2])
                return curr_center - prev_center

            return np.zeros(2)  # 기본값

        except Exception as e:
            print(f"Error calculating movement vector: {str(e)}")
            return np.zeros(2)

    def _calculate_vector_similarity(self, vec1, vec2):
        """벡터 유사도 계산 (코사인 유사도)"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return max(0, float(cosine_sim))  # 음수 유사도는 0으로 처리

        except Exception as e:
            print(f"Error calculating vector similarity: {str(e)}")
            return 0.0

    def _get_total_frames(self, all_tracks_data):
        """전체 프레임 수 계산"""
        if not all_tracks_data:
            return 1

        all_frames = set()
        for track_data in all_tracks_data.values():
            all_frames.update(track_data.keys())

        return len(all_frames) if all_frames else 1

    def _calculate_isolation_penalty(self, track_data, all_tracks_data):
        """고립 상태 패널티 계산"""
        if not all_tracks_data or len(all_tracks_data) <= 1:
            return 1.0  # 다른 객체가 없으면 완전 고립

        # 현재 트랙 ID 찾기
        current_track_id = None
        for tid, tdata in all_tracks_data.items():
            if id(tdata) == id(track_data):
                current_track_id = tid
                break

        if current_track_id is None:
            return 1.0

        # 상호작용 거리 임계값 (화면 대각선의 20%)
        isolation_threshold = np.sqrt(self.img_shape[1]**2 + self.img_shape[0]**2) * 0.2

        frame_isolation_scores = []

        # 각 프레임에서 다른 객체와의 거리 확인
        for frame_idx, frame_data in track_data.items():
            curr_bbox = frame_data['bbox']
            curr_center = np.array([(curr_bbox[0] + curr_bbox[2])/2,
                                   (curr_bbox[1] + curr_bbox[3])/2])

            min_distance = float('inf')

            # 같은 프레임의 다른 모든 객체와 거리 계산
            for other_track_id, other_track_data in all_tracks_data.items():
                if other_track_id == current_track_id:
                    continue

                # 같은 프레임 데이터 찾기
                if frame_idx in other_track_data:
                    other_bbox = other_track_data[frame_idx]['bbox']
                    other_center = np.array([(other_bbox[0] + other_bbox[2])/2,
                                           (other_bbox[1] + other_bbox[3])/2])

                    distance = np.linalg.norm(curr_center - other_center)
                    min_distance = min(min_distance, distance)

            # 거리 기반 고립도 계산 (임계값보다 멀면 고립)
            if min_distance == float('inf') or min_distance > isolation_threshold:
                frame_isolation_scores.append(1.0)  # 완전 고립
            else:
                # 거리에 반비례하는 고립도 (가까울수록 고립도 낮음)
                isolation_score = min(min_distance / isolation_threshold, 1.0)
                frame_isolation_scores.append(isolation_score)

        # 전체 프레임에서 평균 고립도 계산
        avg_isolation = np.mean(frame_isolation_scores) if frame_isolation_scores else 1.0

        # 고립도가 0.8 이상이면 패널티 적용
        return avg_isolation if avg_isolation >= 0.8 else 0.0


class MotionBasedScorer(BaseScorer):
    """움직임 기반 복합점수 계산기"""
    
    def __init__(self, config: ScoringConfig, img_width: int = 640, img_height: int = 640):
        """
        Args:
            config: 점수 계산 설정
            img_width: 이미지 너비
            img_height: 이미지 높이
        """
        super().__init__(config)

        self.img_width = img_width
        self.img_height = img_height
        self.img_shape = (img_height, img_width)

        # 통합된 점수 계산기들
        self.position_scorer = None
        self.fight_scorer = None

        # 자동 초기화
        self.initialize_scorer()
    
    def initialize_scorer(self) -> bool:
        """점수 계산기 초기화"""
        try:
            if self.is_initialized:
                return True

            # 통합된 점수 계산기 초기화
            self.position_scorer = RegionBasedPositionScorer(
                self.img_width, self.img_height
            )

            # 가중치를 config에서 가져와서 설정
            weights = None
            if hasattr(self.config, 'weights') and self.config.weights:
                # 4개 가중치만 추출 (movement, interaction, position, temporal_consistency)
                weight_mapping = {
                    'movement': self.config.weights.get('movement', 0.4),
                    'interaction': self.config.weights.get('interaction', 0.4),
                    'position': self.config.weights.get('position', 0.1),
                    'temporal': self.config.weights.get('temporal', 0.1)
                }
                weights = [weight_mapping['movement'], weight_mapping['interaction'],
                          weight_mapping['position'], weight_mapping['temporal']]

            self.fight_scorer = EnhancedFightInvolvementScorer(
                img_shape=self.img_shape,
                enable_adaptive=True,
                weights=weights
            )

            self.is_initialized = True
            logging.info("Motion-based scorer initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize motion-based scorer: {str(e)}")
            self.is_initialized = False
            return False
    
    def calculate_scores(self, tracked_poses: List[FramePoses]) -> Dict[int, PersonScores]:
        """복합점수 계산"""
        if not self.is_initialized:
            raise RuntimeError("Scorer not initialized. Call initialize_scorer() first.")
        
        # 유효한 트랙 데이터 필터링
        valid_tracks = self.filter_valid_tracks(tracked_poses)
        valid_tracks = self.validate_track_data(valid_tracks)
        
        if not valid_tracks:
            return {}
        
        # 각 트랙별 점수 계산
        track_scores = {}
        
        for track_id, track_data in valid_tracks.items():
            try:
                person_scores = PersonScores(track_id)
                person_scores.frame_count = len(track_data)
                
                # 바운딩 박스와 키포인트 히스토리 추출
                bbox_history = [frame['bbox'] for frame in track_data]
                keypoint_history = [frame['keypoints'] for frame in track_data]
                quality_scores = [frame['score'] for frame in track_data]
                
                person_scores.bbox_history = bbox_history
                person_scores.keypoint_history = keypoint_history
                person_scores.quality_scores = quality_scores
                
                # 각 점수 요소 계산
                person_scores.movement_score = self.calculate_movement_score(bbox_history)
                person_scores.position_score = self.calculate_position_score(bbox_history)
                person_scores.interaction_score = self.calculate_interaction_score(
                    track_id, track_data, valid_tracks
                )
                person_scores.temporal_consistency_score = self.calculate_temporal_consistency_score(track_data)
                person_scores.persistence_score = self.calculate_persistence_score(
                    track_data, len(tracked_poses)
                )
                
                # 복합점수 계산
                person_scores.calculate_composite_score(self.weights)
                
                track_scores[track_id] = person_scores
                
            except Exception as e:
                logging.warning(f"Error calculating scores for track {track_id}: {str(e)}")
                continue
        
        # 통계 업데이트
        self.update_statistics(track_scores)
        
        return track_scores
    
    def calculate_position_score(self, bbox_history: List[List[float]]) -> float:
        """위치 점수 계산"""
        if not bbox_history:
            return 0.0

        if self.position_scorer:
            try:
                # 통합된 구현 사용
                position_score, region_scores = self.position_scorer.calculate_position_score(bbox_history)
                return position_score
            except Exception as e:
                logging.warning(f"Error in position scorer: {str(e)}")

        # 기본 구현: 화면 중앙에 가까울수록 높은 점수
        return self._basic_position_score(bbox_history)
    
    def _basic_position_score(self, bbox_history: List[List[float]]) -> float:
        """기본 위치 점수 계산 구현"""
        center_x = self.img_width / 2
        center_y = self.img_height / 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        position_scores = []
        
        for bbox in bbox_history:
            # 바운딩 박스 중심점
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # 화면 중앙으로부터의 거리
            distance = np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
            
            # 거리를 점수로 변환 (가까울수록 높은 점수)
            score = max(0.0, 1.0 - (distance / max_distance))
            position_scores.append(score)
        
        return np.mean(position_scores) if position_scores else 0.0
    
    def calculate_interaction_score(self, track_id: int, track_data: List[Dict[str, Any]], 
                                  all_tracks: Dict[int, List[Dict[str, Any]]]) -> float:
        """상호작용 점수 계산"""
        if len(all_tracks) <= 1:  # 다른 트랙이 없으면 상호작용 없음
            return 0.0
        
        interaction_scores = []
        
        # 시간별로 다른 트랙들과의 상호작용 계산
        for frame_data in track_data:
            frame_idx = frame_data['frame_idx']
            curr_bbox = frame_data['bbox']
            
            frame_interactions = []
            
            # 같은 프레임의 다른 트랙들과 비교
            for other_track_id, other_track_data in all_tracks.items():
                if other_track_id == track_id:
                    continue
                
                # 같은 프레임의 데이터 찾기
                other_frame_data = None
                for other_frame in other_track_data:
                    if other_frame['frame_idx'] == frame_idx:
                        other_frame_data = other_frame
                        break
                
                if other_frame_data:
                    other_bbox = other_frame_data['bbox']
                    
                    # IoU 계산
                    iou = self._calculate_iou(curr_bbox, other_bbox)
                    
                    # 거리 기반 상호작용 점수
                    distance_score = self._calculate_distance_interaction(curr_bbox, other_bbox)
                    
                    # 상호작용 점수 (IoU + 거리)
                    interaction = iou * 0.7 + distance_score * 0.3
                    frame_interactions.append(interaction)
            
            if frame_interactions:
                interaction_scores.append(max(frame_interactions))
        
        return np.mean(interaction_scores) if interaction_scores else 0.0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance_interaction(self, bbox1: List[float], bbox2: List[float]) -> float:
        """거리 기반 상호작용 점수"""
        # 바운딩 박스 중심점들 간의 거리
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # 상호작용 임계 거리 (바운딩 박스 크기 기반)
        bbox1_size = np.sqrt((bbox1[2] - bbox1[0])**2 + (bbox1[3] - bbox1[1])**2)
        bbox2_size = np.sqrt((bbox2[2] - bbox2[0])**2 + (bbox2[3] - bbox2[1])**2)
        avg_size = (bbox1_size + bbox2_size) / 2
        
        interaction_threshold = avg_size * 2.0  # 평균 크기의 2배를 임계값으로
        
        if distance > interaction_threshold:
            return 0.0
        
        # 가까울수록 높은 점수
        interaction_score = max(0.0, 1.0 - (distance / interaction_threshold))
        
        return interaction_score
    
    def get_scorer_info(self) -> Dict[str, Any]:
        """점수 계산기 정보 반환"""
        base_info = super().get_scorer_info()

        base_info.update({
            'scorer_type': 'motion_based',
            'image_size': (self.img_width, self.img_height),
            'unified_implementation': True,
            'position_scorer_available': self.position_scorer is not None,
            'fight_scorer_available': self.fight_scorer is not None
        })

        return base_info
    
    def set_image_size(self, width: int, height: int):
        """이미지 크기 설정

        Args:
            width: 이미지 너비
            height: 이미지 높이
        """
        if width > 0 and height > 0:
            self.img_width = width
            self.img_height = height
            self.img_shape = (height, width)

            # 위치 점수 계산기 재초기화
            try:
                self.position_scorer = RegionBasedPositionScorer(width, height)
                # Fight scorer도 새로운 이미지 크기로 재초기화
                if self.fight_scorer:
                    self.fight_scorer.img_shape = self.img_shape
                    self.fight_scorer.position_scorer = self.position_scorer
            except Exception as e:
                logging.warning(f"Failed to reinitialize scorers: {str(e)}")
    
    def score_frame_poses(self, frame_poses: FramePoses) -> FramePoses:
        """단일 프레임의 포즈에 대해 점수를 계산하여 반환
        
        파이프라인 인터페이스용 메소드
        
        Args:
            frame_poses: 점수 계산할 프레임 포즈 데이터
            
        Returns:
            점수가 계산된 프레임 포즈 데이터
        """
        if not self.is_initialized:
            if not self.initialize_scorer():
                logging.warning("Scorer not initialized, returning original frame poses")
                return frame_poses
        
        if not frame_poses.persons:
            return frame_poses
        
        # 현재 프레임에 대한 기본적인 점수 계산
        # (완전한 복합점수 계산은 multiple frames가 필요하므로 여기서는 기본 점수만 적용)
        scored_persons = []
        
        for person in frame_poses.persons:
            # 기본적인 위치 점수만 계산 (단일 프레임)
            if person.bbox and len(person.bbox) == 4:
                position_score = self._basic_position_score([person.bbox])
                
                # 메타데이터에 점수 정보 추가
                metadata = person.metadata if hasattr(person, 'metadata') and person.metadata else {}
                metadata.update({
                    'position_score': position_score,
                    'scored_by': 'motion_based_scorer',
                    'frame_scoring': True
                })
                
                # PersonPose 객체 복사 및 메타데이터 업데이트
                scored_person = PersonPose(
                    person_id=person.person_id,
                    bbox=person.bbox,
                    keypoints=person.keypoints,
                    score=person.score
                )
                scored_person.track_id = getattr(person, 'track_id', None)
                scored_person.metadata = metadata
                scored_persons.append(scored_person)
            else:
                # bbox가 유효하지 않은 경우 원본 그대로
                scored_persons.append(person)
        
        # 새로운 FramePoses 생성
        scored_frame_poses = FramePoses(
            frame_idx=frame_poses.frame_idx,
            persons=scored_persons,
            timestamp=frame_poses.timestamp,
            image_shape=frame_poses.image_shape,
            metadata={
                **frame_poses.metadata,
                'scoring_info': {
                    'scorer_type': 'motion_based',
                    'scored_persons': len(scored_persons),
                    'frame_level_scoring': True
                }
            }
        )
        
        return scored_frame_poses
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        if self.position_scorer:
            self.position_scorer = None
        
        if self.fight_scorer:
            self.fight_scorer = None