#!/usr/bin/env python3
"""
Pose Analysis Logger - Step1 vs Step2 비교 분석 로그 시스템
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class PoseAnalysisLogger:
    """Step1 vs Step2 포즈 데이터 비교 분석 로그 클래스"""
    
    def __init__(self, log_dir: str, video_name: str):
        self.log_dir = log_dir
        self.video_name = video_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 파일 경로
        os.makedirs(log_dir, exist_ok=True)
        self.analysis_log = os.path.join(log_dir, f"{video_name}_analysis_{self.timestamp}.json")
        self.summary_log = os.path.join(log_dir, f"{video_name}_summary_{self.timestamp}.txt")
        
        # 분석 데이터 저장
        self.analysis_data = {
            'video_name': video_name,
            'timestamp': self.timestamp,
            'config': {},
            'step1_data': {},
            'step2_data': {},
            'comparison': {},
            'frame_analysis': []
        }
        
        # 통계 정보
        self.stats = {
            'step1_total_detections': 0,
            'step2_total_detections': 0,
            'step1_frames_with_poses': 0,
            'step2_frames_with_poses': 0,
            'filtered_out_detections': 0,
            'tracking_failures': 0,
            'quality_filtered': 0,
            'composite_score_filtered': 0
        }
    
    def log_config(self, config: Dict[str, Any]):
        """트래킹 및 필터링 설정 로그"""
        self.analysis_data['config'] = {
            'track_high_thresh': config.get('track_high_thresh', 0.2),
            'track_low_thresh': config.get('track_low_thresh', 0.1),
            'track_max_disappeared': config.get('track_max_disappeared', 30),
            'track_min_hits': config.get('track_min_hits', 2),
            'quality_threshold': config.get('quality_threshold', 0.2),
            'min_track_length': config.get('min_track_length', 5),
            'movement_weight': config.get('movement_weight', 0.40),
            'position_weight': config.get('position_weight', 0.15),
            'interaction_weight': config.get('interaction_weight', 0.30),
            'temporal_consistency_weight': config.get('temporal_consistency_weight', 0.08),
            'persistence_weight': config.get('persistence_weight', 0.02)
        }
    
    def log_step1_frame(self, frame_idx: int, pose_data: Dict[str, Any]):
        """Step1 프레임별 포즈 데이터 로그"""
        frame_info = {
            'frame_idx': frame_idx,
            'step': 'step1',
            'persons_detected': 0,
            'persons_data': []
        }
        
        if pose_data and 'persons' in pose_data:
            persons = pose_data['persons']
            frame_info['persons_detected'] = len(persons)
            self.stats['step1_total_detections'] += len(persons)
            
            for person_id, person_info in persons.items():
                if isinstance(person_info, dict) and 'keypoint' in person_info:
                    keypoints = person_info['keypoint']
                    
                    # 유효한 키포인트 수 계산
                    if isinstance(keypoints, np.ndarray):
                        if keypoints.ndim == 4:  # (1, 1, 17, 2)
                            kp_2d = keypoints[0, 0]
                        else:
                            kp_2d = keypoints
                        
                        valid_count = np.sum(np.any(kp_2d > 0, axis=1))
                        
                        person_data = {
                            'person_id': person_id,
                            'valid_keypoints': int(valid_count),
                            'total_keypoints': 17,
                            'keypoint_quality': valid_count / 17,
                            'bbox': self._estimate_bbox(kp_2d),
                            'center_position': self._get_center_position(kp_2d)
                        }
                        frame_info['persons_data'].append(person_data)
            
            if frame_info['persons_detected'] > 0:
                self.stats['step1_frames_with_poses'] += 1
        
        # step1 데이터에 프레임 정보 추가
        if 'frames' not in self.analysis_data['step1_data']:
            self.analysis_data['step1_data']['frames'] = {}
        self.analysis_data['step1_data']['frames'][str(frame_idx)] = frame_info
    
    def log_step2_frame(self, frame_idx: int, window_idx: int, window_data: Dict[str, Any], 
                       relative_frame_idx: int, filtered_reasons: List[str] = None):
        """Step2 프레임별 트래킹 데이터 로그"""
        frame_info = {
            'frame_idx': frame_idx,
            'step': 'step2',
            'window_idx': window_idx,
            'relative_frame_idx': relative_frame_idx,
            'persons_detected': 0,
            'persons_data': [],
            'filtered_reasons': filtered_reasons or []
        }
        
        if window_data and 'persons' in window_data:
            persons = window_data['persons']
            frame_info['persons_detected'] = len(persons)
            self.stats['step2_total_detections'] += len(persons)
            
            for person_id, person_info in persons.items():
                if isinstance(person_info, dict):
                    # 트래킹 정보
                    track_id = person_info.get('track_id', 'unknown')
                    composite_score = person_info.get('composite_score', 0.0)
                    
                    # 키포인트 정보
                    keypoints = person_info.get('keypoint')
                    valid_count = 0
                    keypoint_quality = 0.0
                    bbox = None
                    center_pos = None
                    
                    if keypoints is not None and isinstance(keypoints, np.ndarray):
                        # 키포인트 차원 처리
                        if keypoints.ndim == 4:  # (1, T, 17, 2)
                            if relative_frame_idx < keypoints.shape[1]:
                                kp_2d = keypoints[0, relative_frame_idx]
                            else:
                                kp_2d = np.zeros((17, 2))
                        elif keypoints.ndim == 3:  # (T, 17, 2)
                            if relative_frame_idx < keypoints.shape[0]:
                                kp_2d = keypoints[relative_frame_idx]
                            else:
                                kp_2d = np.zeros((17, 2))
                        else:
                            kp_2d = keypoints
                        
                        valid_count = np.sum(np.any(kp_2d > 0, axis=1))
                        keypoint_quality = valid_count / 17
                        bbox = self._estimate_bbox(kp_2d)
                        center_pos = self._get_center_position(kp_2d)
                    
                    # 필터링 분석
                    filtering_analysis = self._analyze_filtering(person_info, composite_score, keypoint_quality)
                    
                    person_data = {
                        'person_id': person_id,
                        'track_id': track_id,
                        'composite_score': composite_score,
                        'valid_keypoints': int(valid_count),
                        'total_keypoints': 17,
                        'keypoint_quality': keypoint_quality,
                        'bbox': bbox,
                        'center_position': center_pos,
                        'filtering_analysis': filtering_analysis
                    }
                    frame_info['persons_data'].append(person_data)
            
            if frame_info['persons_detected'] > 0:
                self.stats['step2_frames_with_poses'] += 1
        
        # step2 데이터에 프레임 정보 추가
        if 'frames' not in self.analysis_data['step2_data']:
            self.analysis_data['step2_data']['frames'] = {}
        self.analysis_data['step2_data']['frames'][str(frame_idx)] = frame_info
    
    def _analyze_filtering(self, person_info: Dict, composite_score: float, keypoint_quality: float) -> Dict:
        """필터링 원인 분석"""
        analysis = {
            'passed_quality_threshold': keypoint_quality >= self.analysis_data['config'].get('quality_threshold', 0.2),
            'passed_composite_score': composite_score > 0.0,
            'has_valid_keypoints': keypoint_quality > 0.0,
            'track_quality': person_info.get('track_quality', 'unknown'),
            'potential_filter_reasons': []
        }
        
        # 필터링 원인 추적
        if keypoint_quality < self.analysis_data['config'].get('quality_threshold', 0.2):
            analysis['potential_filter_reasons'].append(f'keypoint_quality_low: {keypoint_quality:.3f}')
            self.stats['quality_filtered'] += 1
        
        if composite_score <= 0.0:
            analysis['potential_filter_reasons'].append(f'composite_score_low: {composite_score:.3f}')
            self.stats['composite_score_filtered'] += 1
        
        return analysis
    
    def compare_frames(self, frame_idx: int):
        """특정 프레임의 step1 vs step2 비교"""
        step1_frame = self.analysis_data['step1_data']['frames'].get(str(frame_idx))
        step2_frame = self.analysis_data['step2_data']['frames'].get(str(frame_idx))
        
        if not step1_frame or not step2_frame:
            return
        
        comparison = {
            'frame_idx': frame_idx,
            'step1_persons': step1_frame['persons_detected'],
            'step2_persons': step2_frame['persons_detected'],
            'persons_lost': step1_frame['persons_detected'] - step2_frame['persons_detected'],
            'quality_comparison': [],
            'position_comparison': []
        }
        
        # 위치 기반 매칭으로 같은 사람 추적
        step1_persons = step1_frame['persons_data']
        step2_persons = step2_frame['persons_data']
        
        for s1_person in step1_persons:
            # step2에서 가장 가까운 사람 찾기
            closest_match = self._find_closest_person(s1_person, step2_persons)
            
            if closest_match:
                quality_comp = {
                    'step1_quality': s1_person['keypoint_quality'],
                    'step2_quality': closest_match['keypoint_quality'],
                    'step1_valid_kp': s1_person['valid_keypoints'],
                    'step2_valid_kp': closest_match['valid_keypoints'],
                    'composite_score': closest_match['composite_score'],
                    'track_id': closest_match['track_id'],
                    'filtering_reasons': closest_match['filtering_analysis']['potential_filter_reasons']
                }
                comparison['quality_comparison'].append(quality_comp)
            else:
                # step2에서 사라진 사람
                lost_person = {
                    'step1_quality': s1_person['keypoint_quality'],
                    'step1_valid_kp': s1_person['valid_keypoints'],
                    'reason': 'not_found_in_step2'
                }
                comparison['quality_comparison'].append(lost_person)
        
        self.analysis_data['comparison'][str(frame_idx)] = comparison
    
    def _find_closest_person(self, step1_person: Dict, step2_persons: List[Dict]) -> Optional[Dict]:
        """위치 기반으로 가장 가까운 사람 찾기"""
        if not step1_person.get('center_position') or not step2_persons:
            return None
        
        s1_center = step1_person['center_position']
        min_distance = float('inf')
        closest_person = None
        
        for s2_person in step2_persons:
            if s2_person.get('center_position'):
                s2_center = s2_person['center_position']
                distance = np.sqrt((s1_center[0] - s2_center[0])**2 + (s1_center[1] - s2_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_person = s2_person
        
        # 너무 멀리 떨어져 있으면 다른 사람으로 간주
        if min_distance > 100:  # 100 픽셀 이상 차이나면 다른 사람
            return None
        
        return closest_person
    
    def _estimate_bbox(self, keypoints: np.ndarray) -> Optional[List[float]]:
        """키포인트에서 바운딩 박스 추정"""
        try:
            valid_points = keypoints[np.any(keypoints > 0, axis=1)]
            if len(valid_points) < 2:
                return None
            
            x_min, y_min = np.min(valid_points, axis=0)
            x_max, y_max = np.max(valid_points, axis=0)
            return [float(x_min), float(y_min), float(x_max), float(y_max)]
        except:
            return None
    
    def _get_center_position(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """키포인트 중심 위치 계산"""
        try:
            valid_points = keypoints[np.any(keypoints > 0, axis=1)]
            if len(valid_points) < 2:
                return None
            
            center_x = np.mean(valid_points[:, 0])
            center_y = np.mean(valid_points[:, 1])
            return (float(center_x), float(center_y))
        except:
            return None
    
    def generate_summary(self):
        """분석 요약 생성"""
        # 프레임별 비교 분석
        total_frames = len(self.analysis_data['comparison'])
        frames_with_loss = 0
        total_persons_lost = 0
        
        quality_degradation_frames = 0
        avg_quality_drop = 0.0
        
        for frame_idx, comp in self.analysis_data['comparison'].items():
            if comp['persons_lost'] > 0:
                frames_with_loss += 1
                total_persons_lost += comp['persons_lost']
            
            # 품질 저하 분석
            for quality_comp in comp['quality_comparison']:
                if 'step2_quality' in quality_comp:
                    quality_drop = quality_comp['step1_quality'] - quality_comp['step2_quality']
                    if quality_drop > 0.1:  # 10% 이상 품질 저하
                        quality_degradation_frames += 1
                        avg_quality_drop += quality_drop
        
        if quality_degradation_frames > 0:
            avg_quality_drop /= quality_degradation_frames
        
        # 요약 텍스트 생성
        summary = f"""
=== Step1 vs Step2 포즈 분석 요약 ===
비디오: {self.video_name}
분석 시간: {self.timestamp}

=== 전체 통계 ===
Step1 총 감지: {self.stats['step1_total_detections']}
Step2 총 감지: {self.stats['step2_total_detections']}
감지 손실률: {((self.stats['step1_total_detections'] - self.stats['step2_total_detections']) / max(self.stats['step1_total_detections'], 1) * 100):.1f}%

Step1 포즈 있는 프레임: {self.stats['step1_frames_with_poses']}
Step2 포즈 있는 프레임: {self.stats['step2_frames_with_poses']}

=== 필터링 분석 ===
품질 필터링된 감지: {self.stats['quality_filtered']}
복합점수 필터링된 감지: {self.stats['composite_score_filtered']}
트래킹 실패: {self.stats['tracking_failures']}

=== 프레임별 손실 분석 ===
총 분석 프레임: {total_frames}
사람 손실 발생 프레임: {frames_with_loss}
총 손실된 사람 수: {total_persons_lost}
평균 프레임당 손실: {total_persons_lost / max(total_frames, 1):.2f}

=== 품질 분석 ===
품질 저하 프레임: {quality_degradation_frames}
평균 품질 저하율: {avg_quality_drop:.3f}

=== 설정 분석 ===
현재 품질 임계값: {self.analysis_data['config'].get('quality_threshold', 0.2)}
추천 품질 임계값: {max(0.1, min(0.3, avg_quality_drop * 0.5))} (품질 저하 고려)

현재 트래킹 임계값: {self.analysis_data['config'].get('track_high_thresh', 0.2)}
추천 트래킹 임계값: 0.15 (더 관대한 트래킹)

=== 주요 발견사항 ===
"""
        
        # 주요 문제점 식별
        loss_rate = (self.stats['step1_total_detections'] - self.stats['step2_total_detections']) / max(self.stats['step1_total_detections'], 1)
        
        if loss_rate > 0.3:
            summary += "- 심각한 감지 손실 (30% 이상): 트래킹 임계값이 너무 높음\n"
        
        if self.stats['quality_filtered'] > self.stats['step1_total_detections'] * 0.2:
            summary += "- 과도한 품질 필터링 (20% 이상): quality_threshold 완화 필요\n"
        
        if quality_degradation_frames > total_frames * 0.3:
            summary += "- 광범위한 품질 저하: 트래킹 알고리즘 문제 가능성\n"
        
        return summary
    
    def _convert_numpy_types(self, obj):
        """numpy 타입을 Python 네이티브 타입으로 변환"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def save_analysis(self):
        """분석 결과 저장"""
        # numpy 타입 변환
        converted_data = self._convert_numpy_types(self.analysis_data)
        
        # JSON 파일 저장
        with open(self.analysis_log, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        # 요약 텍스트 저장
        summary = self.generate_summary()
        with open(self.summary_log, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"분석 완료: {self.analysis_log}")
        print(f"요약 저장: {self.summary_log}")
        
        return self.analysis_log, self.summary_log