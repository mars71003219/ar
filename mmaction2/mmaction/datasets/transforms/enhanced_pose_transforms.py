# Copyright (c) OpenMMLab. All rights reserved.
"""
Enhanced Pose Transforms for Custom RTMO Annotation Format
커스텀 RTMO 어노테이션을 위한 개선된 포즈 변환

주요 기능:
1. Enhanced annotation metadata 처리
2. Fight-prioritized data augmentation
3. 5-region position-aware transforms
4. Quality-based adaptive processing
5. Multi-person pose handling
"""

import copy
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadEnhancedPoseAnnotation(BaseTransform):
    """Enhanced annotation format에서 포즈 데이터 로드
    
    Enhanced format의 추가 메타데이터를 처리하고 
    Fight-prioritized 정보를 활용합니다.
    
    Args:
        with_enhanced_info (bool): Enhanced metadata 포함 여부
        quality_threshold (float): 품질 임계값
        use_composite_score (bool): 복합 점수 활용 여부
    """
    
    def __init__(self,
                 with_enhanced_info: bool = True,
                 quality_threshold: float = 0.3,
                 use_composite_score: bool = True):
        
        self.with_enhanced_info = with_enhanced_info
        self.quality_threshold = quality_threshold
        self.use_composite_score = use_composite_score
    
    def transform(self, results: Dict) -> Dict:
        """Enhanced annotation 변환"""
        
        # 기본 keypoint 정보는 이미 로드됨
        keypoint = results['keypoint']  # (T, V, C)
        keypoint_score = results['keypoint_score']  # (T, V)
        
        # Enhanced metadata 처리
        if self.with_enhanced_info and 'enhanced_metadata' in results:
            enhanced_meta = results['enhanced_metadata']
            
            # 품질 기반 신뢰도 조정
            quality_threshold = enhanced_meta.get('quality_threshold', self.quality_threshold)
            
            # 낮은 품질 프레임에 대한 가중치 조정
            if self.use_composite_score:
                selected_persons = enhanced_meta.get('selected_persons_info', [])
                if selected_persons:
                    person_info = selected_persons[0]  # 첫 번째 선택된 person
                    
                    # 복합 점수 기반 신뢰도 조정
                    composite_score = person_info.get('composite_score', 0.5)
                    confidence_multiplier = min(1.0, max(0.3, composite_score))
                    
                    # 전체 keypoint score에 신뢰도 반영
                    keypoint_score = keypoint_score * confidence_multiplier
            
            # Enhanced 정보를 results에 저장
            results['enhanced_info'] = enhanced_meta
        
        # 업데이트된 데이터 저장
        results['keypoint'] = keypoint.astype(np.float32)
        results['keypoint_score'] = keypoint_score.astype(np.float32)
        
        return results


@TRANSFORMS.register_module()
class EnhancedPoseNormalize(BaseTransform):
    """Enhanced pose data normalization with 5-region awareness
    
    5영역 분할 정보를 활용한 위치 인식 정규화를 수행합니다.
    
    Args:
        mean (Tuple[float, float]): Normalization mean for (x, y)
        std (Tuple[float, float]): Normalization std for (x, y)
        region_aware (bool): 5영역 인식 정규화 사용 여부
        preserve_center_region (bool): 중앙 영역 보존 여부
    """
    
    def __init__(self,
                 mean: Tuple[float, float] = (0.5, 0.5),
                 std: Tuple[float, float] = (0.5, 0.5),
                 region_aware: bool = True,
                 preserve_center_region: bool = True):
        
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.region_aware = region_aware
        self.preserve_center_region = preserve_center_region
    
    def transform(self, results: Dict) -> Dict:
        """Enhanced normalization 수행"""
        
        keypoint = results['keypoint']  # (T, V, C)
        h, w = results['img_shape']
        
        # 기본 정규화: pixel coordinates -> [0, 1]
        keypoint[..., 0] = keypoint[..., 0] / w
        keypoint[..., 1] = keypoint[..., 1] / h
        
        if self.region_aware:
            # 5영역 분할 기반 적응적 정규화
            keypoint = self._apply_region_aware_normalization(keypoint, h, w, results)
        else:
            # 표준 정규화: [0, 1] -> [-1, 1] (approximately)
            keypoint = (keypoint - self.mean) / self.std
        
        results['keypoint'] = keypoint.astype(np.float32)
        return results
    
    def _apply_region_aware_normalization(self, keypoint: np.ndarray, 
                                        h: int, w: int, results: Dict) -> np.ndarray:
        """5영역 인식 정규화 적용"""
        
        # 5영역 정의 (enhanced format과 동일)
        center_margin = 0.2
        regions = {
            'top_left': (0, 0, 0.5, 0.5),
            'top_right': (0.5, 0, 1.0, 0.5),
            'bottom_left': (0, 0.5, 0.5, 1.0),
            'bottom_right': (0.5, 0.5, 1.0, 1.0),
            'center': (0.5 - center_margin/2, 0.5 - center_margin/2,
                      0.5 + center_margin/2, 0.5 + center_margin/2)
        }
        
        normalized_keypoint = keypoint.copy()
        
        # Enhanced metadata에서 영역별 가중치 정보 활용
        if 'enhanced_info' in results:
            selected_persons = results['enhanced_info'].get('selected_persons_info', [])
            if selected_persons:
                person_info = selected_persons[0]
                region_scores = person_info.get('region_breakdown', {})
                
                # 영역별 적응적 정규화
                for region_name, (x1, y1, x2, y2) in regions.items():
                    region_weight = region_scores.get(region_name, 1.0)
                    
                    # 해당 영역에 있는 keypoint 찾기
                    in_region_mask = ((keypoint[..., 0] >= x1) & (keypoint[..., 0] <= x2) &
                                     (keypoint[..., 1] >= y1) & (keypoint[..., 1] <= y2))
                    
                    if np.any(in_region_mask):
                        # 영역별 가중치를 적용한 정규화
                        region_mean = self.mean * region_weight
                        region_std = self.std * max(0.5, region_weight)
                        
                        normalized_keypoint[in_region_mask] = (
                            (keypoint[in_region_mask] - region_mean) / region_std
                        )
        
        # 중앙 영역 보존 (Fight 중심 영역)
        if self.preserve_center_region:
            center_x1, center_y1, center_x2, center_y2 = regions['center']
            center_mask = ((keypoint[..., 0] >= center_x1) & (keypoint[..., 0] <= center_x2) &
                          (keypoint[..., 1] >= center_y1) & (keypoint[..., 1] <= center_y2))
            
            if np.any(center_mask):
                # 중앙 영역은 더 보수적으로 정규화
                conservative_std = self.std * 0.8
                normalized_keypoint[center_mask] = (
                    (keypoint[center_mask] - self.mean) / conservative_std
                )
        
        return normalized_keypoint


@TRANSFORMS.register_module()
class FightAwareAugmentation(BaseTransform):
    """Fight-aware data augmentation
    
    Fight/Non-fight 레이블에 따라 적응적 데이터 증강을 수행합니다.
    
    Args:
        fight_aug_prob (float): Fight 클래스 증강 확률
        interaction_preserve_prob (float): Interaction 보존 확률
        temporal_consistency_weight (float): 시간적 일관성 가중치
        region_aware_flip (bool): 영역 인식 flip 사용 여부
    """
    
    def __init__(self,
                 fight_aug_prob: float = 0.8,
                 interaction_preserve_prob: float = 0.7,
                 temporal_consistency_weight: float = 0.3,
                 region_aware_flip: bool = True):
        
        self.fight_aug_prob = fight_aug_prob
        self.interaction_preserve_prob = interaction_preserve_prob
        self.temporal_consistency_weight = temporal_consistency_weight
        self.region_aware_flip = region_aware_flip
    
    def transform(self, results: Dict) -> Dict:
        """Fight-aware augmentation 적용"""
        
        label = results.get('label', 0)
        is_fight = (label == 1)
        
        # Fight 영상에 대해 더 적극적인 augmentation
        if is_fight and random.random() < self.fight_aug_prob:
            results = self._apply_fight_augmentation(results)
        
        # Interaction 보존을 위한 constraint
        if (is_fight and 'enhanced_info' in results and 
            random.random() < self.interaction_preserve_prob):
            results = self._preserve_interaction_context(results)
        
        return results
    
    def _apply_fight_augmentation(self, results: Dict) -> Dict:
        """Fight 특화 증강 적용"""
        
        keypoint = results['keypoint']  # (T, V, C)
        
        # 1. 시간적 변형 (Fight에서 중요한 동작 패턴 보존)
        if random.random() < 0.4:
            keypoint = self._apply_temporal_deformation(keypoint)
        
        # 2. 공간적 변형 (상호작용 영역 중심)
        if random.random() < 0.5:
            keypoint = self._apply_interaction_aware_spatial_aug(keypoint, results)
        
        # 3. 영역별 적응적 노이즈
        if random.random() < 0.3:
            keypoint = self._apply_region_adaptive_noise(keypoint, results)
        
        results['keypoint'] = keypoint
        return results
    
    def _apply_temporal_deformation(self, keypoint: np.ndarray) -> np.ndarray:
        """시간적 변형 적용 (Fight 동작 패턴 보존)"""
        
        # keypoint shape: (M, T, V, C) 또는 (T, V, C)
        if keypoint.ndim == 4:
            M, T, V, C = keypoint.shape
            is_multi_person = True
        else:
            T, V, C = keypoint.shape
            is_multi_person = False
        
        # 비선형 시간 변형 (중간 프레임 강조)
        time_indices = np.arange(T)
        center_frame = T // 2
        
        # 중앙 프레임 주변에 더 많은 가중치
        time_weights = np.exp(-0.5 * ((time_indices - center_frame) / (T * 0.3)) ** 2)
        time_weights = time_weights / np.sum(time_weights) * T
        
        # 가중치 기반 프레임 선택 (중복 허용)
        selected_indices = np.random.choice(T, size=T, p=time_weights/np.sum(time_weights))
        selected_indices = np.sort(selected_indices)
        
        # Multi-person 처리
        if is_multi_person:
            return keypoint[:, selected_indices]  # (M, T_new, V, C)
        else:
            return keypoint[selected_indices]  # (T_new, V, C)
    
    def _apply_interaction_aware_spatial_aug(self, keypoint: np.ndarray, 
                                           results: Dict) -> np.ndarray:
        """상호작용 인식 공간 증강"""
        
        # Enhanced metadata에서 interaction 정보 활용
        if 'enhanced_info' in results:
            selected_persons = results['enhanced_info'].get('selected_persons_info', [])
            if selected_persons:
                person_info = selected_persons[0]
                interaction_score = person_info.get('interaction_score', 0.5)
                
                # Interaction 점수에 따른 적응적 변형
                deformation_strength = min(0.1, interaction_score * 0.2)
                
                # 중심 영역 보존하면서 변형 적용
                center_x, center_y = 0.5, 0.5
                
                for t in range(keypoint.shape[0]):
                    for v in range(keypoint.shape[1]):
                        if keypoint[t, v, 0] > 0 and keypoint[t, v, 1] > 0:  # 유효한 keypoint
                            # 중심에서의 거리에 따른 변형
                            dist_from_center = np.sqrt(
                                (keypoint[t, v, 0] - center_x) ** 2 + 
                                (keypoint[t, v, 1] - center_y) ** 2
                            )
                            
                            # 중심에서 멀수록 더 강한 변형
                            local_strength = deformation_strength * (1 + dist_from_center)
                            
                            noise = np.random.normal(0, local_strength, 2)
                            keypoint[t, v, :2] += noise
        
        return keypoint
    
    def _apply_region_adaptive_noise(self, keypoint: np.ndarray, 
                                   results: Dict) -> np.ndarray:
        """영역별 적응적 노이즈 추가"""
        
        # Enhanced metadata에서 영역별 점수 활용
        if 'enhanced_info' in results:
            selected_persons = results['enhanced_info'].get('selected_persons_info', [])
            if selected_persons:
                person_info = selected_persons[0]
                region_scores = person_info.get('region_breakdown', {})
                
                # 5영역 정의
                regions = {
                    'top_left': (0, 0, 0.5, 0.5),
                    'top_right': (0.5, 0, 1.0, 0.5),
                    'bottom_left': (0, 0.5, 0.5, 1.0),
                    'bottom_right': (0.5, 0.5, 1.0, 1.0),
                    'center': (0.3, 0.3, 0.7, 0.7)
                }
                
                for region_name, (x1, y1, x2, y2) in regions.items():
                    region_score = region_scores.get(region_name, 0.5)
                    
                    # 영역별 노이즈 강도 (점수가 높을수록 보존적)
                    noise_strength = 0.02 * (1.0 - region_score)
                    
                    # 해당 영역의 keypoint에 노이즈 추가
                    in_region_mask = ((keypoint[..., 0] >= x1) & (keypoint[..., 0] <= x2) &
                                     (keypoint[..., 1] >= y1) & (keypoint[..., 1] <= y2))
                    
                    if np.any(in_region_mask):
                        noise = np.random.normal(0, noise_strength, keypoint[in_region_mask].shape)
                        keypoint[in_region_mask] += noise
        
        return keypoint
    
    def _preserve_interaction_context(self, results: Dict) -> Dict:
        """상호작용 컨텍스트 보존"""
        
        # Enhanced metadata에서 interaction 정보 확인
        if 'enhanced_info' in results:
            selected_persons = results['enhanced_info'].get('selected_persons_info', [])
            
            for person_info in selected_persons:
                interaction_score = person_info.get('interaction_score', 0)
                
                # High interaction 영역의 keypoint는 더 보수적으로 처리
                if interaction_score > 0.6:
                    # 해당 person의 keypoint에 constraint 적용
                    # (구체적인 구현은 person tracking이 필요)
                    pass
        
        return results


@TRANSFORMS.register_module()
class EnhancedPoseFormat(BaseTransform):
    """Enhanced pose format을 MMAction2 standard format으로 변환
    
    Args:
        num_person (int): 사용할 person 수
        num_keypoints (int): keypoint 수  
        keypoint_layout (str): keypoint layout ('coco', 'coco_wholebody', etc.)
        preserve_enhanced_info (bool): Enhanced 정보 보존 여부
    """
    
    def __init__(self,
                 num_person: int = 1,
                 num_keypoints: int = 17,
                 keypoint_layout: str = 'coco',
                 preserve_enhanced_info: bool = True):
        
        self.num_person = num_person
        self.num_keypoints = num_keypoints
        self.keypoint_layout = keypoint_layout
        self.preserve_enhanced_info = preserve_enhanced_info
    
    def transform(self, results: Dict) -> Dict:
        """Enhanced format을 standard format으로 변환"""
        
        keypoint = results['keypoint']  # (T, V, C)
        keypoint_score = results['keypoint_score']  # (T, V)
        
        T, V, C = keypoint.shape
        
        # MMAction2 standard format: (M, T, V, C)
        # M = num_person
        formatted_keypoint = np.zeros((self.num_person, T, self.num_keypoints, C), dtype=np.float32)
        formatted_score = np.zeros((self.num_person, T, self.num_keypoints), dtype=np.float32)
        
        # 첫 번째 person에 데이터 할당
        actual_keypoints = min(V, self.num_keypoints)
        formatted_keypoint[0, :, :actual_keypoints] = keypoint[:, :actual_keypoints]
        formatted_score[0, :, :actual_keypoints] = keypoint_score[:, :actual_keypoints]
        
        # Enhanced 정보 보존
        if self.preserve_enhanced_info and 'enhanced_info' in results:
            results['enhanced_metadata'] = results['enhanced_info']
        
        # 결과 업데이트
        results['keypoint'] = formatted_keypoint
        results['keypoint_score'] = formatted_score
        results['num_person'] = self.num_person
        results['num_keypoints'] = self.num_keypoints
        
        return results


# Pipeline 조합을 위한 preset
def get_enhanced_train_pipeline():
    """Enhanced training pipeline preset"""
    return [
        dict(type='LoadEnhancedPoseAnnotation', 
             with_enhanced_info=True,
             use_composite_score=True),
        dict(type='EnhancedPoseNormalize',
             region_aware=True,
             preserve_center_region=True),
        dict(type='FightAwareAugmentation',
             fight_aug_prob=0.7,
             interaction_preserve_prob=0.8),
        dict(type='EnhancedPoseFormat',
             num_person=1,
             preserve_enhanced_info=True)
    ]

def get_enhanced_val_pipeline():
    """Enhanced validation pipeline preset"""
    return [
        dict(type='LoadEnhancedPoseAnnotation',
             with_enhanced_info=True,
             use_composite_score=False),  # validation에서는 score 조정 안함
        dict(type='EnhancedPoseNormalize',
             region_aware=True,
             preserve_center_region=True),
        dict(type='EnhancedPoseFormat',
             num_person=1,
             preserve_enhanced_info=True)
    ]