# Copyright (c) OpenMMLab. All rights reserved.
"""
Enhanced Pose Dataset for Custom RTMO Annotation Format
커스텀 RTMO 어노테이션 포맷을 위한 개선된 포즈 데이터셋

주요 기능:
1. Enhanced annotation format 지원
2. Fight-prioritized ranking 활용
3. 5-region 복합 점수 활용
4. 적응형 person 선택
5. 품질 기반 데이터 필터링
"""

import copy
import os
import pickle
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import mmengine
from mmengine.fileio import exists, load

from mmaction.registry import DATASETS
from mmaction.datasets.pose_dataset import PoseDataset


@DATASETS.register_module()
class EnhancedPoseDataset(PoseDataset):
    """Enhanced RTMO annotation format을 지원하는 포즈 데이터셋
    
    이 데이터셋은 enhanced_rtmo_bytetrack_pose_extraction.py에서 생성한
    복합 점수 기반 어노테이션을 처리합니다.
    
    Args:
        ann_file (str): Annotation file path (.pkl)
        pipeline (List[Dict]): Processing pipeline
        data_prefix (Dict): Data path prefix  
        test_mode (bool): Test mode flag
        multi_class (bool): Multi-class classification flag
        num_classes (int): Number of classes (default: 2 for fight/non-fight)
        start_index (int): Start index for temporal sampling
        modality (str): Input modality
        use_fight_ranking (bool): 싸움 우선순위 랭킹 사용 여부
        ranking_strategy (str): 랭킹 전략 ('top_score', 'adaptive', 'quality_weighted')
        min_quality_threshold (float): 최소 품질 임계값
        composite_score_weights (Dict): 복합 점수 가중치 오버라이드
    """
    
    def __init__(self, 
                 ann_file,
                 pipeline,
                 use_fight_ranking=True,
                 ranking_strategy='top_score', 
                 min_quality_threshold=0.3,
                 composite_score_weights=None,
                 **kwargs):
        
        # Enhanced 특화 파라미터 설정
        self.use_fight_ranking = use_fight_ranking
        self.ranking_strategy = ranking_strategy
        self.min_quality_threshold = min_quality_threshold
        self.composite_score_weights = composite_score_weights or {
            'movement_intensity': 0.30,
            'position_5region': 0.35,
            'interaction': 0.20,
            'temporal_consistency': 0.10,
            'persistence': 0.05
        }
        
        # Enhanced annotation에서 추출된 통계 정보
        self.enhanced_stats = {
            'total_videos': 0,
            'avg_persons_per_video': 0.0,
            'quality_distribution': {},
            'score_distribution': {}
        }
        
        # BaseDataset에서 지원하지 않는 인자들 제거
        unsupported_keys = ['times', 'dataset']
        for key in unsupported_keys:
            kwargs.pop(key, None)
        
        # PoseDataset 초기화 (modality는 PoseDataset에서 자동으로 'Pose'로 설정됨)
        super().__init__(ann_file=ann_file, pipeline=pipeline, **kwargs)
    
    def load_data_list(self) -> List[Dict]:
        """Enhanced annotation file을 로드하고 데이터 리스트로 변환"""
        
        if not exists(self.ann_file):
            raise FileNotFoundError(f'Annotation file {self.ann_file} not found')
        
        # Enhanced annotation 로드
        enhanced_data = mmengine.load(self.ann_file)
        
        print(f"Enhanced Annotation Statistics:")
        print(f"   - Total videos: {len(enhanced_data)}")
        
        data_list = []
        quality_scores = []
        person_counts = []
        
        for video_idx, (video_name, video_data) in enumerate(enhanced_data.items()):
            # _metadata 키 건너뛰기
            if video_name == '_metadata':
                continue
                
            # Enhanced annotation format 검증
            if not self._validate_enhanced_format(video_data):
                print(f"Skipping invalid annotation: {video_name}")
                continue
            
            # Fight-prioritized person 선택
            selected_persons = self._select_persons_by_ranking(video_data)
            
            if not selected_persons:
                print(f"No qualifying persons found in {video_name}")
                continue
            
            # MMAction2 format으로 변환
            sample = self._convert_to_mmaction_format(video_data, selected_persons)
            if sample:
                data_list.append(sample)
                quality_scores.append(video_data.get('quality_threshold', 0.3))
                person_counts.append(video_data.get('total_persons', 0))
        
        # 통계 정보 업데이트
        self._update_statistics(quality_scores, person_counts, len(data_list))
        
        print(f"Successfully loaded {len(data_list)} samples")
        if quality_scores:
            print(f"   - Avg quality: {np.mean(quality_scores):.3f}")
            print(f"   - Avg persons per video: {np.mean(person_counts):.1f}")
        
        return data_list
    
    def _validate_enhanced_format(self, video_data: Dict) -> bool:
        """Enhanced annotation format 유효성 검증"""
        required_keys = ['video_info', 'persons', 'score_weights']
        
        if not all(key in video_data for key in required_keys):
            return False
        
        # video_info 필수 필드 검증  
        video_info = video_data['video_info']
        required_video_keys = ['frame_dir', 'total_frames', 'img_shape']
        
        return all(key in video_info for key in required_video_keys)
    
    def _select_persons_by_ranking(self, video_data: Dict) -> List[Dict]:
        """Fight-prioritized ranking을 사용한 person 선택"""
        
        if not self.use_fight_ranking:
            # 기본 MMAction2 방식: 첫 번째 person만 사용
            persons_dict = video_data.get('persons', {})
            if persons_dict:
                first_key = list(persons_dict.keys())[0]
                first_person = persons_dict[first_key]
                if 'annotation' in first_person:
                    return [first_person['annotation']]
            return []
        
        # Enhanced format의 persons 정보 활용 (수정된 구조)
        persons_dict = video_data.get('persons', {})
        if not persons_dict:
            return []
        
        # persons는 dict이므로 list로 변환
        persons_info = list(persons_dict.values())
        
        # 품질 필터링 (수정된 키 이름)
        quality_filtered = [
            person for person in persons_info 
            if person.get('track_quality', 0) >= self.min_quality_threshold
        ]
        
        if not quality_filtered:
            quality_filtered = persons_info[:1]  # 최소 1명은 선택
        
        # 랭킹 전략별 선택
        if self.ranking_strategy == 'top_score':
            # 최고 점수 1명 선택
            selected = sorted(quality_filtered, 
                            key=lambda x: x.get('composite_score', 0), 
                            reverse=True)[:1]
            
        elif self.ranking_strategy == 'adaptive':
            # 적응형: 점수에 따라 1-2명 선택
            sorted_persons = sorted(quality_filtered,
                                  key=lambda x: x.get('composite_score', 0),
                                  reverse=True)
            
            if len(sorted_persons) >= 2:
                top_score = sorted_persons[0].get('composite_score', 0)
                second_score = sorted_persons[1].get('composite_score', 0)
                
                # 두 번째 인물의 점수가 첫 번째의 70% 이상이면 둘 다 선택
                if second_score >= top_score * 0.7:
                    selected = sorted_persons[:2]
                else:
                    selected = sorted_persons[:1]
            else:
                selected = sorted_persons
                
        elif self.ranking_strategy == 'quality_weighted':
            # 품질 가중: 복합 점수와 품질 점수를 결합
            for person in quality_filtered:
                composite = person.get('composite_score', 0)
                quality = person.get('quality_score', 0)
                person['weighted_score'] = composite * 0.7 + quality * 0.3
            
            selected = sorted(quality_filtered,
                            key=lambda x: x.get('weighted_score', 0),
                            reverse=True)[:1]
        else:
            selected = quality_filtered[:1]
        
        # annotation에서 실제 데이터 추출 (수정된 구조)
        result = []
        
        for person in selected:
            if 'annotation' in person:
                person_data = person['annotation'].copy()
                person_data['enhanced_info'] = person  # 추가 정보 보존
                result.append(person_data)
        
        return result
    
    def _convert_to_mmaction_format(self, video_data: Dict, selected_persons: List[Dict]) -> Optional[Dict]:
        """Enhanced format을 MMAction2 format으로 변환"""
        
        video_info = video_data['video_info']
        
        # 기본 정보 추출
        frame_dir = video_info['frame_dir']
        total_frames = video_info['total_frames']
        img_shape = video_info['img_shape']
        label = video_info['label']
        
        if not selected_persons:
            return None
        
        # 여러 person의 keypoint 데이터 결합
        all_keypoints = []
        all_scores = []
        
        for person_data in selected_persons:
            keypoints = person_data['keypoint']  # shape: (1, T, V, C)
            scores = person_data['keypoint_score']  # shape: (1, T, V)
            
            # Person dimension 제거 후 추가
            all_keypoints.append(keypoints[0])  # (T, V, C)
            all_scores.append(scores[0])  # (T, V)
        
        # 다중 person 데이터를 단일 형태로 결합
        if len(all_keypoints) == 1:
            final_keypoints = all_keypoints[0]  # (T, V, C)
            final_scores = all_scores[0]  # (T, V)
        else:
            # 여러 person이 있는 경우 concatenation 또는 averaging
            # 여기서는 가장 높은 점수의 keypoint를 frame별로 선택
            final_keypoints = np.zeros_like(all_keypoints[0])
            final_scores = np.zeros_like(all_scores[0])
            
            for t in range(len(all_keypoints[0])):
                # 각 frame에서 가장 높은 평균 점수를 가진 person 선택
                frame_scores = [np.mean(scores[t]) for scores in all_scores]
                best_person_idx = np.argmax(frame_scores)
                
                final_keypoints[t] = all_keypoints[best_person_idx][t]
                final_scores[t] = all_scores[best_person_idx][t]
        
        # MMAction2 STGCN++은 (M, T, V, C) shape을 기대하므로 person dimension 추가
        final_keypoints = final_keypoints[np.newaxis, ...]  # (1, T, V, C)
        final_scores = final_scores[np.newaxis, ...]  # (1, T, V)
        
        # MMAction2 sample format 생성
        sample = {
            'frame_dir': frame_dir,
            'total_frames': total_frames,
            'img_shape': img_shape,
            'original_shape': img_shape,
            'label': label,
            'keypoint': final_keypoints,  # (1, T, V, C)
            'keypoint_score': final_scores,  # (1, T, V)
            # Enhanced 정보 보존
            'enhanced_metadata': {
                'total_persons': video_data.get('total_persons', 0),
                'score_weights': video_data.get('score_weights', {}),
                'quality_threshold': video_data.get('quality_threshold', 0.3),
                'selected_strategy': self.ranking_strategy,
                'selected_persons_info': [p.get('enhanced_info', {}) for p in selected_persons]
            }
        }
        
        return sample
    
    def _update_statistics(self, quality_scores: List[float], 
                          person_counts: List[int], total_samples: int):
        """통계 정보 업데이트"""
        
        self.enhanced_stats.update({
            'total_videos': total_samples,
            'avg_persons_per_video': np.mean(person_counts) if person_counts else 0.0,
            'avg_quality': np.mean(quality_scores) if quality_scores else 0.0,
            'quality_distribution': {
                'min': np.min(quality_scores) if quality_scores else 0.0,
                'max': np.max(quality_scores) if quality_scores else 0.0,
                'std': np.std(quality_scores) if quality_scores else 0.0
            }
        })
    
    def get_enhanced_statistics(self) -> Dict:
        """Enhanced dataset 통계 정보 반환"""
        return self.enhanced_stats
    
    def evaluate(self, results: List, metrics: Union[str, List[str]] = 'top_k_accuracy', 
                 metric_options: Dict = dict(topk=(1, 5)), **kwargs) -> Dict:
        """평가 메트릭 계산 (Enhanced 정보 포함)"""
        
        # 기본 평가 수행
        eval_results = super().evaluate(results, metrics, metric_options, **kwargs)
        
        # Enhanced 정보 추가
        eval_results.update({
            'enhanced_dataset_info': self.enhanced_stats,
            'ranking_strategy': self.ranking_strategy,
            'min_quality_threshold': self.min_quality_threshold
        })
        
        return eval_results


@DATASETS.register_module() 
class EnhancedFightDataset(EnhancedPoseDataset):
    """Fight/Non-fight 이진 분류를 위한 특화된 Enhanced dataset"""
    
    def __init__(self, **kwargs):
        # Fight 특화 기본값 설정
        kwargs.setdefault('num_classes', 2)
        kwargs.setdefault('use_fight_ranking', True)
        kwargs.setdefault('ranking_strategy', 'adaptive')
        kwargs.setdefault('min_quality_threshold', 0.25)  # Fight 데이터는 조금 더 관대하게
        
        super().__init__(**kwargs)
    
    def _select_persons_by_ranking(self, video_data: Dict) -> List[Dict]:
        """Fight 특화 person 선택 로직"""
        
        persons_dict = video_data.get('persons', {})
        if not persons_dict:
            return []
        
        # persons는 dict이므로 list로 변환
        persons_info = list(persons_dict.values())
        
        # Fight 라벨인 경우 더 많은 person 허용
        is_fight = video_data['video_info'].get('label', 0) == 1
        
        # 품질 필터링 (수정된 키 이름)
        quality_filtered = [
            person for person in persons_info 
            if person.get('track_quality', 0) >= self.min_quality_threshold
        ]
        
        if not quality_filtered:
            quality_filtered = persons_info[:1]
        
        # Fight 영상의 경우 interaction 점수가 높은 상위 2명 선택
        if is_fight and len(quality_filtered) >= 2:
            # Interaction 점수로 정렬
            sorted_by_interaction = sorted(quality_filtered,
                                         key=lambda x: x.get('interaction_score', 0),
                                         reverse=True)
            
            # 상위 2명의 interaction 점수 확인
            if (len(sorted_by_interaction) >= 2 and 
                sorted_by_interaction[1].get('interaction_score', 0) > 0.3):
                selected = sorted_by_interaction[:2]
            else:
                selected = sorted_by_interaction[:1]
        else:
            # Non-fight이거나 사람이 적은 경우 복합 점수로 선택
            selected = sorted(quality_filtered,
                            key=lambda x: x.get('composite_score', 0),
                            reverse=True)[:1]
        
        # annotation에서 실제 데이터 추출 (수정된 구조)
        result = []
        
        for person in selected:
            if 'annotation' in person:
                person_data = person['annotation'].copy()
                person_data['enhanced_info'] = person  # 추가 정보 보존
                result.append(person_data)
        
        return result