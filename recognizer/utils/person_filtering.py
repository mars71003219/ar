"""
인원수 기반 필터링 유틸리티
혼자 있는 사람 객체를 필터링하여 부정확한 fight 분류를 방지
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from utils.data_structure import PersonPose, FramePoses
except ImportError:
    from ..utils.data_structure import PersonPose, FramePoses


class PersonProximityFilter:
    """
    사람 객체 간 근접성 기반 필터링
    """
    
    def __init__(self, 
                 min_persons_for_interaction: int = 2,
                 proximity_multiplier: float = 4.5,
                 enable_filtering: bool = True):
        """
        Args:
            min_persons_for_interaction: 상호작용 행동 분류에 필요한 최소 인원수
            proximity_multiplier: 객체 너비의 배수 (근접 범위 계산용)
            enable_filtering: 필터링 활성화 여부
        """
        self.min_persons_for_interaction = min_persons_for_interaction
        self.proximity_multiplier = proximity_multiplier
        self.enable_filtering = enable_filtering
        
        self.stats = {
            'total_persons_processed': 0,
            'isolated_persons_filtered': 0,
            'windows_with_filtering': 0,
            'total_windows_processed': 0
        }
        
        logging.info(f"PersonProximityFilter initialized:")
        logging.info(f"  - Min persons for interaction: {min_persons_for_interaction}")
        logging.info(f"  - Proximity multiplier: {proximity_multiplier}x")
        logging.info(f"  - Filtering enabled: {enable_filtering}")
    
    def calculate_bbox_width(self, bbox: List[float]) -> float:
        """바운딩 박스 너비 계산"""
        if len(bbox) < 4:
            return 0.0
        return abs(bbox[2] - bbox[0])
    
    def calculate_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """바운딩 박스 중심점 계산"""
        if len(bbox) < 4:
            return (0.0, 0.0)
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        return (center_x, center_y)
    
    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """두 중심점 간 유클리드 거리 계산"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def find_nearby_persons(self, target_person: PersonPose, all_persons: List[PersonPose]) -> List[PersonPose]:
        """
        특정 객체 주변의 근접한 다른 객체들을 찾기
        
        Args:
            target_person: 기준이 되는 사람 객체
            all_persons: 전체 사람 객체 리스트
            
        Returns:
            근접 범위 내의 다른 사람 객체들 (자신 제외)
        """
        if not target_person.bbox or len(target_person.bbox) < 4:
            return []
        
        target_width = self.calculate_bbox_width(target_person.bbox)
        if target_width <= 0:
            return []
        
        target_center = self.calculate_bbox_center(target_person.bbox)
        proximity_threshold = target_width * self.proximity_multiplier
        
        nearby_persons = []
        
        for person in all_persons:
            # 자기 자신 제외
            if person is target_person:
                continue
            
            if not person.bbox or len(person.bbox) < 4:
                continue
            
            person_center = self.calculate_bbox_center(person.bbox)
            distance = self.calculate_distance(target_center, person_center)
            
            if distance <= proximity_threshold:
                nearby_persons.append(person)
        
        return nearby_persons
    
    def is_person_isolated(self, target_person: PersonPose, all_persons: List[PersonPose]) -> bool:
        """
        특정 객체가 혼자 있는지 확인
        
        Args:
            target_person: 확인할 사람 객체
            all_persons: 전체 사람 객체 리스트
            
        Returns:
            혼자 있으면 True, 근접한 다른 사람이 있으면 False
        """
        nearby_persons = self.find_nearby_persons(target_person, all_persons)
        
        # 최소 인원수 요구사항 확인
        # target_person(1명) + nearby_persons가 최소 인원수를 만족하는지 확인
        total_nearby_count = 1 + len(nearby_persons)  # 자신 + 근접한 사람들
        
        is_isolated = total_nearby_count < self.min_persons_for_interaction
        
        if is_isolated:
            logging.debug(f"Person isolated: width={self.calculate_bbox_width(target_person.bbox):.1f}, "
                         f"nearby_count={len(nearby_persons)}, total={total_nearby_count}")
        
        return is_isolated
    
    def filter_isolated_persons(self, frame_poses: FramePoses) -> FramePoses:
        """
        프레임에서 혼자 있는 사람들을 필터링
        
        Args:
            frame_poses: 필터링할 프레임 데이터
            
        Returns:
            필터링된 프레임 데이터
        """
        if not self.enable_filtering:
            return frame_poses
        
        if not frame_poses.persons or len(frame_poses.persons) == 0:
            return frame_poses
        
        # 전체 인원수가 최소 요구사항보다 적으면 모두 필터링
        if len(frame_poses.persons) < self.min_persons_for_interaction:
            logging.debug(f"Frame {frame_poses.frame_idx}: Total persons ({len(frame_poses.persons)}) "
                         f"< min required ({self.min_persons_for_interaction}), filtering all")
            filtered_frame = FramePoses(
                frame_idx=frame_poses.frame_idx,
                timestamp=frame_poses.timestamp,
                persons=[]
            )
            self.stats['isolated_persons_filtered'] += len(frame_poses.persons)
            return filtered_frame
        
        # 각 객체별로 혼자 있는지 확인하고 필터링
        filtered_persons = []
        isolated_count = 0
        
        for person in frame_poses.persons:
            if self.is_person_isolated(person, frame_poses.persons):
                isolated_count += 1
                logging.debug(f"Frame {frame_poses.frame_idx}: Filtering isolated person "
                             f"(track_id: {person.track_id})")
            else:
                filtered_persons.append(person)
        
        # 필터링된 프레임 생성
        filtered_frame = FramePoses(
            frame_idx=frame_poses.frame_idx,
            timestamp=frame_poses.timestamp,
            persons=filtered_persons
        )
        
        # 통계 업데이트
        self.stats['isolated_persons_filtered'] += isolated_count
        if isolated_count > 0:
            self.stats['windows_with_filtering'] += 1
            
        logging.debug(f"Frame {frame_poses.frame_idx}: {len(frame_poses.persons)} → {len(filtered_persons)} "
                     f"persons (filtered: {isolated_count})")
        
        return filtered_frame
    
    def filter_window_frames(self, frames: List[FramePoses]) -> List[FramePoses]:
        """
        윈도우 내 모든 프레임에 대해 필터링 적용
        
        Args:
            frames: 필터링할 프레임 리스트
            
        Returns:
            필터링된 프레임 리스트
        """
        if not self.enable_filtering:
            return frames
        
        self.stats['total_windows_processed'] += 1
        
        filtered_frames = []
        
        for frame_poses in frames:
            self.stats['total_persons_processed'] += len(frame_poses.persons) if frame_poses.persons else 0
            filtered_frame = self.filter_isolated_persons(frame_poses)
            filtered_frames.append(filtered_frame)
        
        return filtered_frames
    
    def should_skip_classification(self, frames: List[FramePoses]) -> bool:
        print(f"*** FILTERING CHECK: frames={len(frames) if frames else 0}, enable={self.enable_filtering} ***")
        """
        분류를 건너뛸지 결정
        
        Args:
            frames: 확인할 프레임 리스트
            
        Returns:
            분류를 건너뛸지 여부 (True: 건너뛰기, False: 분류 진행)
        """
        if not self.enable_filtering:
            print("*** FILTERING DISABLED ***")
            return False
        
        # 필터링 후 남은 객체 수 확인
        total_remaining_persons = 0
        for frame_poses in frames:
            if frame_poses.persons:
                total_remaining_persons += len(frame_poses.persons)
        
        print(f"*** TOTAL PERSONS: {total_remaining_persons}, MIN REQUIRED: {self.min_persons_for_interaction} ***")
        
        # 윈도우 전체에서 최소 인원수 요구사항을 만족하지 않으면 건너뛰기
        should_skip = total_remaining_persons < self.min_persons_for_interaction
        
        print(f"*** SHOULD SKIP: {should_skip} ***")
        
        if should_skip:
            logging.info(f"Skipping classification: insufficient persons after filtering "
                        f"(remaining: {total_remaining_persons}, required: {self.min_persons_for_interaction})")
        
        return should_skip
    
    def get_filter_stats(self) -> Dict:
        """필터링 통계 반환"""
        stats = self.stats.copy()
        
        if stats['total_persons_processed'] > 0:
            stats['isolation_rate'] = stats['isolated_persons_filtered'] / stats['total_persons_processed']
        else:
            stats['isolation_rate'] = 0.0
        
        if stats['total_windows_processed'] > 0:
            stats['windows_with_filtering_rate'] = stats['windows_with_filtering'] / stats['total_windows_processed']
        else:
            stats['windows_with_filtering_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_persons_processed': 0,
            'isolated_persons_filtered': 0,
            'windows_with_filtering': 0,
            'total_windows_processed': 0
        }
        
        logging.info("PersonProximityFilter stats reset")
    
    def update_config(self, 
                     min_persons_for_interaction: Optional[int] = None,
                     proximity_multiplier: Optional[float] = None,
                     enable_filtering: Optional[bool] = None):
        """설정 업데이트"""
        if min_persons_for_interaction is not None:
            self.min_persons_for_interaction = min_persons_for_interaction
            logging.info(f"Updated min_persons_for_interaction to {min_persons_for_interaction}")
        
        if proximity_multiplier is not None:
            self.proximity_multiplier = proximity_multiplier
            logging.info(f"Updated proximity_multiplier to {proximity_multiplier}")
        
        if enable_filtering is not None:
            self.enable_filtering = enable_filtering
            logging.info(f"Updated enable_filtering to {enable_filtering}")


def create_proximity_filter_from_config(config: Dict) -> PersonProximityFilter:
    """
    설정에서 PersonProximityFilter 생성
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        PersonProximityFilter 인스턴스
    """
    filter_config = config.get('person_filtering', {})
    
    return PersonProximityFilter(
        min_persons_for_interaction=filter_config.get('min_persons_for_interaction', 2),
        proximity_multiplier=filter_config.get('proximity_multiplier', 4.5),
        enable_filtering=filter_config.get('enable_filtering', True)
    )