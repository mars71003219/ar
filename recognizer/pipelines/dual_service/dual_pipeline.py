"""
듀얼 서비스 파이프라인 래퍼
Fight와 Falldown 두 서비스를 동시에 실행하는 파이프라인
"""

import logging
import time
import traceback
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import pickle

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.factory import ModuleFactory
from utils.data_structure import PersonPose, FramePoses, WindowAnnotation, ClassificationResult
from pipelines.separated.data_structures import VisualizationData
from pipelines.base import BasePipeline
from visualization.inference_visualizer import InferenceVisualizer


class DualServicePipeline(BasePipeline):
    """듀얼 서비스 파이프라인 - Fight + Falldown 동시 처리"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.config = config
        self.dual_config = config.get('dual_service', {})
        
        # 듀얼 서비스 모드 확인 (enabled=false일 때도 단일 서비스로 동작)
        self.is_dual_mode = self.dual_config.get('enabled', False)
        
        # 서비스 목록
        self.services = self.dual_config.get('services', ['fight', 'falldown'])
        
        mode_name = "Dual" if self.is_dual_mode else "Single"
        logging.info(f"{mode_name} service pipeline initialized with services: {self.services}")
        
        # 공통 모듈들
        self.pose_estimator = None
        self.tracker = None
        self.window_processor = None
        
        # 서비스별 모듈들
        self.scorers = {}
        self.classifiers = {}
        
        # 시각화
        self.visualizer = InferenceVisualizer()
        
        # 프레임 인덱스 관리 (멀티 비디오 처리용)
        self.frame_idx = 0
        
        # 성능 추적
        self.performance_stats = {
            'frames_processed': 0,
            'windows_classified': 0,
            'service_fps': {}
        }
        
        self.service_timings = {service: [] for service in self.services}
        
        # 최근 분류 결과 저장 (지속적인 오버레이 표시용)
        self.latest_classification_result = None
    
    def initialize_pipeline(self) -> bool:
        """듀얼 서비스 파이프라인 초기화"""
        try:
            logging.info("Initializing dual service pipeline...")
            
            # 1. 공통 모듈 초기화
            if not self._initialize_common_modules():
                return False
            
            # 2. 서비스별 모듈 초기화
            if not self._initialize_service_modules():
                return False
            
            logging.info("Dual service pipeline initialization completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize dual service pipeline: {e}")
            return False
    
    def _initialize_common_modules(self) -> bool:
        """공통 모듈들 초기화"""
        try:
            # Pose Estimator
            pose_config = self.config.get('models', {}).get('pose_estimation', {}).get('onnx', {})
            self.pose_estimator = ModuleFactory.create_pose_estimator('rtmo_onnx', pose_config)
            
            if not self.pose_estimator or not self.pose_estimator.initialize_model():
                logging.error("Failed to initialize pose estimator")
                return False
            
            # Tracker
            tracker_config = self.config.get('models', {}).get('tracking', {})
            self.tracker = ModuleFactory.create_tracker('bytetrack', tracker_config)
            
            if not self.tracker or not self.tracker.initialize_tracker():
                logging.error("Failed to initialize tracker")
                return False
            
            # Window Processor
            window_config = self.config.get('performance', {})
            # realtime 모드로 명시적 설정
            window_config['processing_mode'] = 'realtime'
            self.window_processor = ModuleFactory.create_window_processor('sliding_window', window_config)
            
            if not self.window_processor:
                logging.error("Failed to initialize window processor")
                return False
            
            logging.info("Common modules initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize common modules: {e}")
            return False
    
    def _initialize_service_modules(self) -> bool:
        """서비스별 모듈들 초기화"""
        try:
            for service in self.services:
                logging.info(f"Initializing {service} service modules...")
                
                # Scorer 초기화
                scorer_config = self.config.get('models', {}).get('scoring', {}).get(service, {})
                if service == 'fight':
                    scorer = ModuleFactory.create_scorer('region_based', scorer_config)
                elif service == 'falldown':
                    scorer = ModuleFactory.create_scorer('falldown_scorer', scorer_config)
                else:
                    logging.warning(f"Unknown service: {service}")
                    continue
                
                if not scorer or not scorer.initialize_scorer():
                    logging.error(f"Failed to initialize {service} scorer")
                    return False
                
                self.scorers[service] = scorer
                
                # Classifier 초기화
                model_config = self.config.get('models', {}).get(f'{service}_classification', {})
                logging.info(f"Creating {service} classifier with config keys: {list(model_config.keys())}")
                logging.info(f"Model name from config: {model_config.get('model_name', 'not_found')}")
                if service == 'fight':
                    logging.info("Creating fight classifier with name: stgcn")
                    classifier = ModuleFactory.create_classifier('stgcn', model_config)
                elif service == 'falldown':
                    logging.info("Creating falldown classifier with name: stgcn")
                    classifier = ModuleFactory.create_classifier('stgcn', model_config)
                else:
                    continue
                
                if not classifier or not classifier.initialize_model():
                    logging.error(f"Failed to initialize {service} classifier")
                    return False
                
                self.classifiers[service] = classifier
                
                logging.info(f"{service} service modules initialized successfully")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize service modules: {e}")
            return False
    
    def process_frame(self, frame) -> Tuple[FramePoses, Dict[str, Any]]:
        """단일 프레임 처리 - 공통 포즈 추정 및 트래킹"""
        try:
            # 1. 포즈 추정
            pose_start = time.time()
            pose_results = self.pose_estimator.process_frame(frame)
            pose_time = time.time() - pose_start
            
            # 2. 트래킹
            track_start = time.time()
            tracked_results = self.tracker.track_frame_poses(pose_results)
            track_time = time.time() - track_start
            
            # 성능 통계 업데이트
            self.performance_stats['frames_processed'] += 1
            
            # FPS 정보 포함
            fps_info = {
                'pose_fps': 1.0 / pose_time if pose_time > 0 else 0,
                'track_fps': 1.0 / track_time if track_time > 0 else 0
            }
            
            return tracked_results, fps_info
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None, {}
    
    def process_window_dual_service(self, window_data: WindowAnnotation) -> Dict[str, ClassificationResult]:
        """듀얼 서비스 윈도우 처리"""
        results = {}
        
        for service in self.services:
            try:
                service_start = time.time()
                
                # 1. 서비스별 스코어링
                scorer = self.scorers.get(service)
                if scorer:
                    # Window 데이터에서 포즈 히스토리 추출 및 스코어링 적용
                    try:
                        # WindowAnnotation에서 프레임별 포즈 데이터를 FramePoses 리스트로 변환
                        tracked_poses = self._extract_tracked_poses_from_window(window_data)
                        
                        logging.info(f"*** {service.upper()} POSE EXTRACTION DEBUG ***")
                        logging.info(f"Extracted tracked_poses: {len(tracked_poses)} frames")
                        if tracked_poses:
                            logging.info(f"Sample frame persons: {len(tracked_poses[0].persons)}")
                            if tracked_poses[0].persons:
                                logging.info(f"Sample person track_id: {tracked_poses[0].persons[0].track_id}")
                                logging.info(f"Sample person bbox: {tracked_poses[0].persons[0].bbox}")
                                logging.info(f"Sample person score: {tracked_poses[0].persons[0].score:.3f}")
                            
                            # 모든 프레임의 track_id 분포 확인
                            track_counts = {}
                            for frame in tracked_poses:
                                for person in frame.persons:
                                    track_id = person.track_id
                                    track_counts[track_id] = track_counts.get(track_id, 0) + 1
                            logging.info(f"Track distribution: {track_counts}")
                        
                        if tracked_poses:
                            # 스코어링 실행
                            logging.info(f"*** {service.upper()} SCORER EXECUTION START ***")
                            logging.info(f"Input: {len(tracked_poses)} frames for {service} scoring")
                            
                            track_scores = scorer.calculate_scores(tracked_poses)
                            
                            logging.info(f"*** {service.upper()} SCORER RESULTS ***")
                            logging.info(f"Generated {len(track_scores)} track scores:")
                            for track_id, scores in track_scores.items():
                                logging.info(f"  Track {track_id}: composite={scores.composite_score:.3f}")
                                if hasattr(scores, 'height_change_score'):
                                    logging.info(f"    height_change={scores.height_change_score:.3f}")
                                if hasattr(scores, 'posture_angle_score'):
                                    logging.info(f"    posture_angle={scores.posture_angle_score:.3f}")
                                if hasattr(scores, 'movement_intensity_score'):
                                    logging.info(f"    movement_intensity={scores.movement_intensity_score:.3f}")
                                if hasattr(scores, 'persistence_score'):
                                    logging.info(f"    persistence={scores.persistence_score:.3f}")
                            
                            # 스코어링 결과를 윈도우 데이터에 반영
                            scored_data = self._apply_scores_to_window(window_data, track_scores)
                        else:
                            logging.warning(f"No tracked poses extracted for {service} scoring")
                            scored_data = window_data
                    except Exception as e:
                        logging.error(f"Error in {service} scoring: {e}")
                        scored_data = window_data
                else:
                    scored_data = window_data
                
                # 2. 서비스별 분류
                classifier = self.classifiers.get(service)
                if classifier:
                    logging.info(f"*** {service.upper()} STGCN CLASSIFICATION START ***")
                    logging.info(f"Classifier type: {type(classifier).__name__}")
                    logging.info(f"Model name: {getattr(classifier, 'model_name', 'unknown')}")
                    logging.info(f"Input scored_data type: {type(scored_data)}")
                    if hasattr(scored_data, 'person_rankings') and scored_data.person_rankings:
                        logging.info(f"Person rankings in scored_data: {scored_data.person_rankings}")
                    else:
                        logging.info("No person rankings found in scored_data")
                    
                    classification_result = classifier.classify_window(scored_data)
                    
                    logging.info(f"*** {service.upper()} CLASSIFICATION RESULT ***")
                    if classification_result:
                        logging.info(f"Prediction: {classification_result.prediction}")
                        logging.info(f"Confidence: {classification_result.confidence:.6f}")
                        logging.info(f"Probabilities: {classification_result.probabilities}")
                        logging.info(f"Model: {classification_result.model_name}")
                    
                    results[service] = classification_result
                else:
                    logging.error(f"No classifier found for {service} service")
                
                service_time = time.time() - service_start
                self.service_timings[service].append(service_time)
                
                # FPS 계산 (최근 10개 평균)
                recent_times = self.service_timings[service][-10:]
                avg_time = sum(recent_times) / len(recent_times)
                self.performance_stats['service_fps'][f'{service}_cls_fps'] = 1.0 / avg_time if avg_time > 0 else 0
                
            except Exception as e:
                logging.error(f"Error processing {service} service: {e}")
                # 에러 시 기본 결과 생성
                results[service] = ClassificationResult(
                    prediction=0,
                    confidence=0.0,
                    probabilities=[1.0, 0.0],
                    model_name=f'{service}_error'
                )
        
        self.performance_stats['windows_classified'] += 1
        return results
    
    def _extract_tracked_poses_from_window(self, window_data: WindowAnnotation) -> List:
        """WindowAnnotation에서 FramePoses 리스트 추출"""
        try:
            from utils.data_structure import FramePoses, PersonPose
            
            # WindowAnnotation의 keypoint 데이터는 [M, T, V, C] 형태
            keypoint = window_data.keypoint  # [M, T, V, C]
            keypoint_score = window_data.keypoint_score  # [M, T, V]
            
            M, T, V, _ = keypoint.shape
            tracked_poses = []
            
            # 각 프레임별로 FramePoses 생성
            for t in range(T):
                persons = []
                
                # 각 person별로 PersonPose 생성
                for m in range(M):
                    person_keypoints = keypoint[m, t, :, :].tolist()  # [V, C]
                    person_scores = keypoint_score[m, t, :].tolist()   # [V]
                    
                    # 유효한 키포인트가 있는지 확인
                    valid_keypoints = sum(1 for score in person_scores if score > 0.3)
                    if valid_keypoints < 5:  # 최소 5개 키포인트 필요
                        continue
                    
                    # 키포인트에 신뢰도 추가 [x, y, confidence]
                    keypoints_with_confidence = []
                    for v in range(V):
                        x, y = person_keypoints[v][0], person_keypoints[v][1]
                        conf = person_scores[v]
                        keypoints_with_confidence.append([x, y, conf])
                    
                    # 바운딩 박스 계산 (유효한 키포인트 기준)
                    valid_points = [(kpt[0], kpt[1]) for kpt in keypoints_with_confidence if kpt[2] > 0.3 and kpt[0] > 0 and kpt[1] > 0]
                    if len(valid_points) < 4:
                        continue
                    
                    xs, ys = zip(*valid_points)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    # bbox가 유효한지 확인 (양수이고 합리적인 크기)
                    if min_x < 0 or min_y < 0 or max_x <= min_x or max_y <= min_y:
                        continue
                    
                    bbox = [min_x, min_y, max_x, max_y]
                    
                    # PersonPose 생성
                    person = PersonPose(
                        person_id=m,
                        bbox=bbox,
                        keypoints=keypoints_with_confidence,
                        score=sum(person_scores) / len(person_scores)
                    )
                    # WindowAnnotation에서는 이미 같은 person이 M 차원에서 추적되고 있음
                    # 따라서 M 인덱스를 track_id로 사용하면 모든 프레임에서 일관성 유지
                    person.track_id = m
                    persons.append(person)
                
                # FramePoses 생성
                if persons:
                    frame_poses = FramePoses(
                        frame_idx=window_data.start_frame + t,
                        persons=persons,
                        timestamp=None,
                        image_shape=window_data.img_shape
                    )
                    tracked_poses.append(frame_poses)
            
            logging.info(f"*** POSE EXTRACTION RESULT ***")
            logging.info(f"Window keypoint shape: {keypoint.shape}")
            logging.info(f"Extracted {len(tracked_poses)} frames with poses from window")
            
            # 추출된 데이터 검증
            if tracked_poses:
                sample_frame = tracked_poses[0]
                logging.info(f"Sample frame {sample_frame.frame_idx}: {len(sample_frame.persons)} persons")
                if sample_frame.persons:
                    sample_person = sample_frame.persons[0]
                    logging.info(f"Sample person track_id: {sample_person.track_id}, bbox: {sample_person.bbox}")
                    logging.info(f"Sample person score: {sample_person.score:.3f}")
            
            return tracked_poses
            
        except Exception as e:
            logging.error(f"Error extracting tracked poses from window: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def _apply_scores_to_window(self, window_data: WindowAnnotation, track_scores: Dict) -> WindowAnnotation:
        """스코어링 결과를 WindowAnnotation에 반영"""
        try:
            # 복합점수 정보를 윈도우 메타데이터에 추가
            if not hasattr(window_data, 'composite_scores') or window_data.composite_scores is None:
                window_data.composite_scores = {}
            
            # 각 트랙의 복합점수 저장
            for track_id, person_scores in track_scores.items():
                window_data.composite_scores[track_id] = person_scores.composite_score
                logging.debug(f"Track {track_id} composite score: {person_scores.composite_score:.3f}")
            
            # 개인 순위도 업데이트
            if track_scores:
                rankings = [(track_id, scores.composite_score) for track_id, scores in track_scores.items()]
                rankings.sort(key=lambda x: x[1], reverse=True)
                window_data.person_rankings = rankings
                logging.debug(f"Person rankings: {rankings}")
            
            return window_data
            
        except Exception as e:
            logging.error(f"Error applying scores to window: {e}")
            return window_data
    
    def create_dual_classification_result(self, service_results: Dict[str, ClassificationResult], 
                                        window_idx: int, fps_info: Dict[str, Any]) -> Dict[str, Any]:
        """듀얼 서비스 결과를 통합 분류 결과로 변환"""
        
        # Fight 결과
        fight_result = service_results.get('fight')
        if fight_result:
            fight_class = 'Fight' if fight_result.prediction == 1 else 'NonFight'
            # Fight 스코어는 항상 probabilities[1] (Fight 클래스 확률) 사용
            fight_confidence = fight_result.probabilities[1] if len(fight_result.probabilities) > 1 else fight_result.confidence
            logging.info(f"Fight result - class: {fight_class}, confidence: {fight_confidence:.3f}, probabilities: {fight_result.probabilities}")
        else:
            fight_class = 'NonFight'
            fight_confidence = 0.0
        
        # Falldown 결과
        falldown_result = service_results.get('falldown')
        if falldown_result:
            falldown_class = 'Falldown' if falldown_result.prediction == 1 else 'Normal'
            # Falldown 스코어는 항상 probabilities[1] (Falldown 클래스 확률) 사용
            falldown_confidence = falldown_result.probabilities[1] if len(falldown_result.probabilities) > 1 else falldown_result.confidence
        else:
            falldown_class = 'Normal'
            falldown_confidence = 0.0
        
        # 통합 결과 생성 (듀얼 서비스 시각화용)
        combined_result = {
            'window_number': window_idx,
            
            # Fight 정보
            'fight_predicted_class': fight_class,
            'fight_confidence': fight_confidence,
            
            # Falldown 정보  
            'falldown_predicted_class': falldown_class,
            'falldown_confidence': falldown_confidence,
            
            # FPS 정보
            'pose_fps': fps_info.get('pose_fps', 30.0),
            'fight_cls_fps': self.performance_stats['service_fps'].get('fight_cls_fps', 10.0),
            'falldown_cls_fps': self.performance_stats['service_fps'].get('falldown_cls_fps', 10.0),
            
            # 시각화 호환성을 위한 필수 키들
            'show_classification': True,
            'show_keypoints': True,
            'show_tracking_ids': True,
            
            # 기존 호환성 (우선순위 높은 서비스 기준)
            'predicted_class': falldown_class if falldown_class == 'Falldown' else fight_class,
            'confidence': falldown_confidence if falldown_class == 'Falldown' else fight_confidence
        }
        
        return combined_result
    
    def reset_pipeline_state(self):
        """파이프라인 상태 초기화 (새 비디오 처리 시)"""
        try:
            logging.info("Resetting dual service pipeline state for new video")
            
            # 프레임 인덱스 초기화
            self.frame_idx = 0
            logging.info("Frame index reset to 0")
            
            # 포즈 추정기 통계 초기화
            if self.pose_estimator and hasattr(self.pose_estimator, 'reset_stats'):
                self.pose_estimator.reset_stats()
                logging.info("Pose estimator statistics reset")
            
            # 트래커 초기화
            if self.tracker:
                self.tracker.reset()
                logging.info("Tracker state reset")
            
            # 윈도우 프로세서 초기화
            if self.window_processor:
                self.window_processor.reset()
                logging.info("WindowProcessor reset completed")
            
            # 분류기들 초기화
            if hasattr(self, 'classifiers') and self.classifiers:
                for service, classifier in self.classifiers.items():
                    if hasattr(classifier, 'reset'):
                        classifier.reset()
                        logging.info(f"{service} classifier window counter reset")
                    else:
                        logging.warning(f"{service} classifier does not have reset method")
            
            # 성능 통계 초기화
            self.performance_stats = {
                'frames_processed': 0,
                'windows_classified': 0,
                'service_fps': {}
            }
            self.service_timings = {service: [] for service in self.services}
            
            # 분류 결과 초기화
            self.latest_classification_result = None
            
            logging.info("Dual service pipeline state reset completed")
            
        except Exception as e:
            logging.error(f"Error resetting pipeline state: {e}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """파이프라인 정보 반환"""
        return {
            'pipeline_type': 'dual_service',
            'services': self.services,
            'performance_stats': self.performance_stats,
            'modules_initialized': {
                'pose_estimator': self.pose_estimator is not None,
                'tracker': self.tracker is not None,
                'scorers': list(self.scorers.keys()),
                'classifiers': list(self.classifiers.keys())
            }
        }
    
    def start_realtime_display(self, 
                             input_source, 
                             display_width: int = 1280,
                             display_height: int = 720,
                             save_output: bool = False,
                             output_path = None) -> bool:
        """
        실시간 디스플레이 모드 시작 (듀얼 서비스용)
        InferencePipeline과 호환되는 인터페이스 제공
        """
        try:
            from utils.realtime_input import RealtimeInputManager
            from visualization.inference_visualizer import InferenceVisualizer
            import cv2
            
            # # 입력 매니저 초기화
            # input_manager = RealtimeInputManager(input_source)
            
            # 실시간 입력 관리자 초기화
            input_manager = RealtimeInputManager(
                input_source=input_source,
                buffer_size=5,
                target_fps=getattr(self.config, 'target_fps', 30)
            )
            
            # 시각화 초기화
            visualizer = InferenceVisualizer()
            
            if not input_manager.start():
                logging.error("Failed to start input capture")
                return False
            
            # 비디오 작성기 설정
            video_writer = None
            if save_output and output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (display_width, display_height))
                logging.info(f"Video writer setup successfully: {output_path}")
            
            # 시각화 초기화 (단순화)
            logging.info(f"Visualizer initialized: mode=realtime, {display_width}x{display_height}, FPS=30")
            
            # 윈도우 초기화
            service_mode = "DUAL SERVICE" if self.is_dual_mode else f"SINGLE SERVICE ({', '.join(self.services).upper()})"
            window_name = f"Violence Detection - {service_mode}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, display_width, display_height)
            logging.info(f"Display window started: {window_name} ({display_width}x{display_height})")
            
            # 클래스 변수 사용 (reset_pipeline_state에서 초기화됨)
            none_frame_count = 0  # None 프레임 연속 카운터
            max_none_frames = 100  # 최대 허용 None 프레임 수 (비디오 종료 감지용)
            
            try:
                while True:
                    frame_data = input_manager.get_latest_frame()
                    if frame_data is None:
                        none_frame_count += 1
                        # 비디오 종료 감지: None 프레임이 많거나 캡처 스레드가 종료됨
                        if none_frame_count >= max_none_frames or not input_manager.is_alive():
                            logging.info(f"Video ended - none_frames: {none_frame_count}, input_running: {input_manager.is_running}")
                            break
                        time.sleep(0.001)
                        continue
                    
                    # 프레임을 받았으므로 카운터 리셋
                    none_frame_count = 0
                    
                    frame, _, _ = frame_data
                    self.frame_idx += 1
                    
                    # 프레임 처리 (듀얼 서비스)
                    try:
                        # 1. 포즈 추정
                        pose_start = time.time()
                        frame_poses = self.pose_estimator.process_frame(frame, self.frame_idx)
                        
                        # frame_idx 강제 재설정 (멀티 비디오 처리 시 연속성 보장)
                        if frame_poses:
                            frame_poses.frame_idx = self.frame_idx
                        
                        pose_time = time.time() - pose_start
                        
                        if self.frame_idx < 5:  # 처음 5프레임만 로깅
                            logging.info(f"After pose estimation - frame {self.frame_idx}: {len(frame_poses.persons)} persons")
                        
                        # 2. 트래킹 (존재 확인 후 처리)
                        track_time = 0
                        if self.tracker:
                            track_start = time.time()
                            tracked_results = self.tracker.track_frame_poses(frame_poses)
                            track_time = time.time() - track_start
                            frame_poses = tracked_results
                            
                            if self.frame_idx < 5:  # 처음 5프레임만 로깅
                                logging.info(f"After tracking - frame {self.frame_idx}: {len(frame_poses.persons)} persons")
                        
                        # FPS 정보 업데이트
                        fps_info = {
                            'pose_fps': 1.0 / pose_time if pose_time > 0 else 0,
                            'track_fps': 1.0 / track_time if track_time > 0 and self.tracker else 0
                        }
                        
                    except Exception as e:
                        logging.error(f"Frame {self.frame_idx} processing error: {e}")
                        logging.error(f"Detailed traceback: {traceback.format_exc()}")
                        # 에러 시 빈 포즈 데이터로 처리 계속
                        frame_poses = None
                        fps_info = {'pose_fps': 0, 'track_fps': 0}
                    
                    # 윈도우 처리
                    if frame_poses and self.window_processor:
                        logging.debug(f"Frame {self.frame_idx}: Processing {len(frame_poses.persons)} persons")
                        # 윈도우 데이터 추가
                        ready_windows = self.window_processor.add_frame(frame_poses)
                        logging.debug(f"Frame {self.frame_idx}: Got {len(ready_windows)} ready windows")
                        
                        # 준비된 윈도우 처리 - 최신 결과 저장
                        for window_data in ready_windows:
                            logging.info(f"Processing window {window_data.window_idx} with {window_data.total_frames} frames")
                            service_results = self.process_window_dual_service(window_data)
                            new_classification_result = self.create_dual_classification_result(
                                service_results, 
                                window_data.window_idx, 
                                fps_info
                            )
                            # 최신 분류 결과 저장 (지속적인 표시용)
                            self.latest_classification_result = new_classification_result
                            logging.info(f"Window {window_data.window_idx} classification: {new_classification_result.get('show_classification', False)}")
                    else:
                        if not frame_poses:
                            logging.debug(f"Frame {self.frame_idx}: No frame_poses")
                        if not self.window_processor:
                            logging.debug(f"Frame {self.frame_idx}: No window_processor")
                    
                    # 시각화용 분류 결과 준비
                    current_classification_result = self.latest_classification_result
                    if current_classification_result is None:
                        # 아직 분류 결과가 없는 경우 기본값 설정
                        current_classification_result = {
                            'window_number': 0,
                            'fight_predicted_class': 'Normal',
                            'fight_confidence': 0.0,
                            'falldown_predicted_class': 'Normal', 
                            'falldown_confidence': 0.0,
                            'pose_fps': fps_info.get('pose_fps', 30.0),
                            'fight_cls_fps': 0.0,
                            'falldown_cls_fps': 0.0,
                            'show_classification': False,  # 윈도우가 없으면 분류 안함
                            'show_keypoints': True,
                            'show_tracking_ids': True,
                            'predicted_class': 'Normal',
                            'confidence': 0.0
                        }
                    else:
                        # 최신 분류 결과가 있으면 FPS 정보만 업데이트
                        current_classification_result = current_classification_result.copy()
                        current_classification_result['pose_fps'] = fps_info.get('pose_fps', 30.0)
                    
                    # 시각화
                    try:
                        display_frame = visualizer.visualize_frame(
                            frame, frame_poses, current_classification_result
                        )
                    except Exception as e:
                        logging.error(f"Visualization error: {e}")
                        logging.error(f"Classification result type: {type(current_classification_result)}")
                        if current_classification_result:
                            logging.error(f"Classification keys: {list(current_classification_result.keys())}")
                        traceback.print_exc()
                        display_frame = frame.copy()
                    
                    # 화면 표시
                    cv2.imshow(window_name, display_frame)
                    
                    # 비디오 저장
                    if video_writer is not None:
                        video_writer.write(display_frame)
                    
                    # ESC 키로 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        logging.info("ESC key pressed, stopping...")
                        break
                        
            except KeyboardInterrupt:
                logging.info("Interrupted by user")
            except Exception as e:
                logging.error(f"Error in realtime processing: {e}")
                logging.error(f"Full traceback: {traceback.format_exc()}")
                return False
            finally:
                # 정리
                input_manager.stop()
                cv2.destroyAllWindows()
                if video_writer:
                    video_writer.release()
                
            logging.info("Dual service realtime display completed")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start dual service realtime display: {e}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 공통 모듈 정리
            if self.pose_estimator:
                self.pose_estimator.cleanup()
            if self.tracker:
                self.tracker.cleanup()
            
            # 서비스별 모듈 정리
            for scorer in self.scorers.values():
                if scorer:
                    scorer.cleanup()
            
            for classifier in self.classifiers.values():
                if classifier:
                    classifier.cleanup()
            
            logging.info("Dual service pipeline cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during pipeline cleanup: {e}")

    def process_video_file_for_analysis(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        """분석 모드용 비디오 파일 처리 (JSON/PKL 출력)"""
        try:
            import json
            import cv2

            logging.info(f"Processing video for analysis: {video_path}")

            # 출력 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 직접 OpenCV로 비디오 처리 (기존 extract_video_poses 방식 사용)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return {'success': False, 'error': 'Failed to open video'}

            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

            # 프레임 인덱스 초기화
            self.frame_idx = 0

            # 윈도우 프로세서 상태 초기화
            if self.window_processor:
                self.window_processor.reset()

            # 결과 저장을 위한 데이터 구조 (FramePoses 객체 리스트로 변경)
            all_frames_poses = []
            classification_results = []

            try:
                # 프레임 처리 루프
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.info("End of video file reached")
                        break

                    self.frame_idx += 1

                    # 포즈 추정
                    frame_poses = self.pose_estimator.process_frame(frame, self.frame_idx)
                    if frame_poses:
                        frame_poses.frame_idx = self.frame_idx

                    # 트래킹
                    if self.tracker:
                        frame_poses = self.tracker.track_frame_poses(frame_poses)

                    # FramePoses 객체 직접 저장
                    if frame_poses:
                        all_frames_poses.append(frame_poses)

                    # 윈도우 처리 (분류용)
                    if self.window_processor:
                        windows = self.window_processor.add_frame(frame_poses)

                        for window in windows:
                            if window:
                                # 서비스별 분류 실행
                                service_results = {}

                                for service in self.services:
                                    if service in self.classifiers:
                                        try:
                                            result = self.classifiers[service].classify_window(window)
                                            # ClassificationResult의 실제 속성명 사용
                                            predicted_class = result.prediction if hasattr(result, 'prediction') else result.predicted_class
                                            confidence = result.confidence
                                            all_predictions = result.probabilities if hasattr(result, 'probabilities') else getattr(result, 'all_predictions', [])

                                            service_results[service] = {
                                                'predicted_class': predicted_class,
                                                'confidence': confidence,
                                                'all_predictions': all_predictions
                                            }
                                            logging.info(f"Window {window.window_idx} classified by {service}: {predicted_class} ({confidence:.2f})")
                                        except Exception as e:
                                            logging.error(f"Classification error for {service}: {e}")
                                            service_results[service] = {
                                                'predicted_class': 'error',
                                                'confidence': 0.0,
                                                'all_predictions': []
                                            }

                                # 분류 결과 저장
                                classification_data = {
                                    'window_start_frame': window.start_frame,
                                    'window_end_frame': window.end_frame,
                                    'services': service_results
                                }
                                classification_results.append(classification_data)

                    # 진행 상황 로깅
                    if self.frame_idx % 500 == 0:
                        logging.info(f"Processed {self.frame_idx}/{total_frames} frames ({self.frame_idx/total_frames*100:.1f}%)")

            finally:
                cap.release()

            # 비디오 끝에서 남은 프레임들로 윈도우 생성
            if self.window_processor:
                final_windows = self.window_processor.finalize_processing()
                logging.info(f"Generated {len(final_windows)} final windows from remaining frames")
                for window in final_windows:
                    if window:
                        # 서비스별 분류 실행
                        service_results = {}

                        for service in self.services:
                            if service in self.classifiers:
                                try:
                                    result = self.classifiers[service].classify_window(window)
                                    # ClassificationResult의 실제 속성명 사용
                                    predicted_class = result.prediction if hasattr(result, 'prediction') else result.predicted_class
                                    confidence = result.confidence
                                    all_predictions = result.probabilities if hasattr(result, 'probabilities') else getattr(result, 'all_predictions', [])

                                    service_results[service] = {
                                        'predicted_class': predicted_class,
                                        'confidence': confidence,
                                        'all_predictions': all_predictions
                                    }
                                    logging.info(f"Final window classified by {service}: {predicted_class} ({confidence:.2f})")
                                except Exception as e:
                                    logging.error(f"Final classification error for {service}: {e}")
                                    service_results[service] = {
                                        'predicted_class': 'error',
                                        'confidence': 0.0,
                                        'all_predictions': []
                                    }

                        # 분류 결과 저장
                        classification_data = {
                            'window_start_frame': window.start_frame,
                            'window_end_frame': window.end_frame,
                            'services': service_results
                        }
                        classification_results.append(classification_data)

            # 결과 저장
            video_name = Path(video_path).stem

            # JSON용 프레임 데이터 변환 (FramePoses → dict)
            frames_data = []
            for frame_poses in all_frames_poses:
                frame_data = {
                    'frame_idx': frame_poses.frame_idx,
                    'persons': []
                }
                if frame_poses.persons:
                    for person in frame_poses.persons:
                        person_data = {
                            'person_id': person.person_id,
                            'track_id': getattr(person, 'track_id', None),
                            'bbox': person.bbox,
                            'keypoints': person.keypoints,
                            'score': person.score
                        }
                        frame_data['persons'].append(person_data)
                frames_data.append(frame_data)

            # JSON 결과 저장
            json_file = output_path / f"{video_name}_results.json"
            results_data = {
                'video_info': {
                    'path': video_path,
                    'total_frames': self.frame_idx,
                    'services': self.services
                },
                'frames': frames_data,
                'classification_results': classification_results  # PKLVisualizer가 기대하는 키명
            }

            # JSON serialization을 위한 numpy 배열 처리
            def convert_numpy_to_list(obj):
                """numpy 배열을 리스트로 변환하는 재귀 함수"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_list(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_list(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_to_list(item) for item in obj)
                elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                    return obj.item()
                else:
                    return obj

            # numpy 배열을 리스트로 변환
            serializable_results = convert_numpy_to_list(results_data)

            with open(json_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            # PKL 결과 저장 (VisualizationData 객체로 래핑)
            pkl_file = output_path / f"{video_name}_frame_poses.pkl"
            visualization_data = VisualizationData(
                video_name=video_name,
                frame_data=all_frames_poses,
                stage_info={'stage': 'inference_analysis', 'services': self.services},
                poses_with_tracking=all_frames_poses
            )
            with open(pkl_file, 'wb') as f:
                pickle.dump(visualization_data, f)

            # 분류 결과만 따로 저장
            if classification_results:
                classification_pkl = output_path / f"{video_name}_classifications.pkl"
                with open(classification_pkl, 'wb') as f:
                    pickle.dump(classification_results, f)

            logging.info(f"Analysis completed for {video_name}: {self.frame_idx} frames processed")
            logging.info(f"Results saved to {output_dir}")

            return {
                'success': True,
                'frames_processed': self.frame_idx,
                'classifications_count': len(classification_results),
                'output_files': {
                    'json': str(json_file),
                    'poses_pkl': str(pkl_file),
                    'classifications_pkl': str(classification_pkl) if classification_results else None
                }
            }

        except Exception as e:
            logging.error(f"Error in analysis processing: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def process_video_file_for_annotation(self, video_path: str, output_dir: str, stage: str = "stage1") -> Dict[str, Any]:
        """Annotation 모드용 비디오 파일 처리 (PKL 출력)"""
        try:
            import cv2

            logging.info(f"Processing video for annotation ({stage}): {video_path}")

            # 출력 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 직접 OpenCV로 비디오 처리 (기존 extract_video_poses 방식 사용)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return {'success': False, 'error': 'Failed to open video'}

            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

            # 프레임 인덱스 초기화
            self.frame_idx = 0

            # 결과 저장을 위한 데이터 구조
            all_frames_data = []

            try:
                # 프레임 처리 루프
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.info("End of video file reached")
                        break

                    self.frame_idx += 1

                    # Stage에 따른 처리
                    if stage == "stage1":
                        # Stage 1: 포즈 추정만
                        frame_poses = self.pose_estimator.process_frame(frame, self.frame_idx)
                        if frame_poses:
                            frame_poses.frame_idx = self.frame_idx

                        # 프레임 데이터 저장
                        frame_data_dict = {
                            'frame_idx': self.frame_idx,
                            'persons': []
                        }

                        if frame_poses and frame_poses.persons:
                            for person in frame_poses.persons:
                                person_data = {
                                    'person_id': person.person_id,
                                    'bbox': person.bbox,
                                    'keypoints': person.keypoints,
                                    'score': person.score
                                }
                                frame_data_dict['persons'].append(person_data)

                        all_frames_data.append(frame_data_dict)

                    elif stage == "stage2":
                        # Stage 2: 포즈 추정 + 트래킹
                        frame_poses = self.pose_estimator.process_frame(frame, self.frame_idx)
                        if frame_poses:
                            frame_poses.frame_idx = self.frame_idx

                        # 트래킹
                        if self.tracker:
                            frame_poses = self.tracker.track_frame_poses(frame_poses)

                        # 프레임 데이터 저장 (트래킹 ID 포함)
                        frame_data_dict = {
                            'frame_idx': self.frame_idx,
                            'persons': []
                        }

                        if frame_poses and frame_poses.persons:
                            for person in frame_poses.persons:
                                person_data = {
                                    'person_id': person.person_id,
                                    'track_id': getattr(person, 'track_id', None),
                                    'bbox': person.bbox,
                                    'keypoints': person.keypoints,
                                    'score': person.score
                                }
                                frame_data_dict['persons'].append(person_data)

                        all_frames_data.append(frame_data_dict)

                    # 진행 상황 로깅
                    if self.frame_idx % 500 == 0:
                        logging.info(f"Processed {self.frame_idx}/{total_frames} frames ({self.frame_idx/total_frames*100:.1f}%)")

            finally:
                cap.release()

            # 결과 저장
            video_name = Path(video_path).stem
            pkl_file = output_path / f"{video_name}_{stage}_poses.pkl"

            with open(pkl_file, 'wb') as f:
                pickle.dump(all_frames_data, f)

            logging.info(f"Annotation {stage} completed for {video_name}: {self.frame_idx} frames processed")
            logging.info(f"Results saved to {pkl_file}")

            return {
                'success': True,
                'frames_processed': self.frame_idx,
                'output_file': str(pkl_file),
                'output_path': str(output_path)
            }

        except Exception as e:
            logging.error(f"Error in annotation processing: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def process_pkl_file_for_annotation(self, pkl_path: str, output_dir: str, stage: str) -> Dict[str, Any]:
        """PKL 파일을 사용한 Annotation 처리 (Stage2, Stage3용)"""

        try:
            logging.info(f"Processing PKL for annotation ({stage}): {pkl_path}")

            # PKL 처리용 최소 초기화 (scorers만 초기화)
            if not self.scorers and stage == "stage2":
                self._initialize_scorers_only()

            # PKL 파일 로드
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)

            if stage == "stage2":
                return self._process_stage2_tracking(pkl_data, pkl_path, output_dir)
            elif stage == "stage3":
                return self._process_stage3_dataset(pkl_data, pkl_path, output_dir)
            else:
                return {"success": False, "error": f"Unknown stage: {stage}"}

        except Exception as e:
            logging.error(f"Error in PKL annotation processing: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _initialize_scorers_only(self) -> bool:
        """PKL 처리를 위한 scorer만 초기화"""
        try:
            for service in self.services:
                logging.info(f"Initializing {service} scorer for PKL processing...")

                # Scorer 설정
                scoring_config = self.config.get('scoring', {}).get(service, self.config.get('scoring', {}))

                # 서비스별 scorer 초기화
                if service == 'fight':
                    from scoring.motion_based import MotionBasedScorer
                    from utils.data_structure import ScoringConfig

                    # dict를 ScoringConfig 객체로 변환
                    weights = scoring_config.get('weights', {
                        'movement': 0.4,
                        'interaction': 0.4,
                        'position': 0.1,
                        'temporal': 0.1
                    })

                    scoring_cfg = ScoringConfig(
                        scorer_name=scoring_config.get('scorer_name', 'motion_based'),
                        min_track_length=scoring_config.get('min_track_length', 10),
                        quality_threshold=scoring_config.get('quality_threshold', 0.3),
                        movement_weight=weights.get('movement', 0.4),
                        interaction_weight=weights.get('interaction', 0.4),
                        position_weight=weights.get('position', 0.1),
                        temporal_weight=weights.get('temporal', 0.1)
                    )

                    scorer = MotionBasedScorer(
                        scoring_cfg,
                        img_width=640,
                        img_height=480
                    )
                elif service == 'falldown':
                    from scoring.motion_based import FalldownScorer
                    from utils.data_structure import ScoringConfig

                    # dict를 ScoringConfig 객체로 변환
                    weights = scoring_config.get('weights', {
                        'height_change': 0.35,
                        'posture_angle': 0.25,
                        'movement_intensity': 0.20,
                        'persistence': 0.15,
                        'position': 0.05
                    })

                    scoring_cfg = ScoringConfig(
                        scorer_name=scoring_config.get('scorer_name', 'falldown_scorer'),
                        min_track_length=scoring_config.get('min_track_length', 10),
                        quality_threshold=scoring_config.get('quality_threshold', 0.3),
                        movement_weight=weights.get('movement_intensity', 0.20),
                        interaction_weight=weights.get('posture_angle', 0.25),
                        position_weight=weights.get('position', 0.05),
                        temporal_weight=weights.get('persistence', 0.15)
                    )

                    scorer = FalldownScorer(
                        scoring_cfg,
                        img_width=640,
                        img_height=480
                    )
                else:
                    logging.warning(f"Unknown service: {service}")
                    continue

                if scorer:
                    self.scorers[service] = scorer
                    logging.info(f"✓ {service} scorer initialized")
                else:
                    logging.warning(f"Failed to initialize {service} scorer")

            return len(self.scorers) > 0

        except Exception as e:
            logging.error(f"Failed to initialize scorers: {e}")
            traceback.print_exc()
            return False

    def _process_stage2_tracking(self, pkl_data: dict, pkl_path: str, output_dir: str) -> Dict[str, Any]:
        """Stage2 tracking 처리"""
        logging.info(f"Processing Stage2 tracking for: {pkl_path}")

        try:

            # 파일명에서 비디오 이름 추출
            pkl_file = Path(pkl_path)
            video_name = pkl_file.name.replace('_stage1_poses.pkl', '')

            # 출력 파일 경로
            output_file = Path(output_dir) / f"{video_name}_tracking.pkl"

            # 간단한 tracking 처리 (실제로는 ByteTrack 등을 사용)
            tracking_results = []

            if isinstance(pkl_data, list):
                frame_data_list = pkl_data
            else:
                frame_data_list = [pkl_data]

            # 먼저 기본 tracking 데이터 구성
            for i, frame_data in enumerate(frame_data_list):
                frame_idx = frame_data.get('frame_idx', i)
                persons = frame_data.get('persons', [])

                # 간단한 tracking ID 할당
                tracked_persons = []
                for j, person in enumerate(persons):
                    tracked_person = person.copy()
                    tracked_person['track_id'] = j  # 간단한 ID 할당

                    # 기본 스코어 초기화
                    for service_name in self.services:
                        tracked_person[f'{service_name}_score'] = 0.0

                    tracked_persons.append(tracked_person)

                tracking_results.append({
                    'frame_idx': frame_idx,
                    'persons': tracked_persons
                })

            # 활성화된 서비스에 대한 스코어 계산
            if self.scorers:
                try:
                    # FramePoses 형태로 변환하여 스코어 계산
                    from utils.data_structure import FramePoses, PersonPose

                    frame_poses_list = []
                    for frame_data in tracking_results:
                        frame_idx = frame_data['frame_idx']
                        persons_data = []

                        for person in frame_data['persons']:
                            if 'keypoints' in person and len(person['keypoints']) > 0:
                                keypoints = np.array(person['keypoints'])
                                if keypoints.ndim == 2:  # (17, 3) 형태
                                    person_pose = PersonPose(
                                        person_id=person.get('person_id', person['track_id']),
                                        bbox=person.get('bbox', [0, 0, 100, 100]),
                                        keypoints=keypoints,
                                        score=person.get('score', 1.0),
                                        track_id=person['track_id']
                                    )
                                    persons_data.append(person_pose)

                        if persons_data:
                            frame_poses = FramePoses(
                                frame_idx=frame_idx,
                                persons=persons_data,
                                timestamp=float(frame_idx),  # 간단히 frame_idx를 timestamp로 사용
                                image_shape=(480, 640),
                                metadata={}
                            )
                            frame_poses_list.append(frame_poses)

                    # 각 서비스별 스코어 계산
                    for service_name, scorer in self.scorers.items():
                        if frame_poses_list:
                            try:
                                scores_dict = scorer.calculate_scores(frame_poses_list)

                                # 계산된 스코어를 tracking_results에 적용
                                for frame_data in tracking_results:
                                    for person in frame_data['persons']:
                                        track_id = person['track_id']
                                        if track_id in scores_dict:
                                            person_scores = scores_dict[track_id]
                                            person[f'{service_name}_score'] = person_scores.composite_score

                                logging.info(f"✓ Calculated {service_name} scores for {len(scores_dict)} tracks")

                            except Exception as e:
                                logging.warning(f"Failed to calculate {service_name} scores: {e}")

                except Exception as e:
                    logging.warning(f"Error in score calculation: {e}")

            # 결과를 기존 PKL 구조로 저장 (separated 파이프라인과 동일)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # StageResult 형태로 저장하기 위해 필요한 imports
            try:
                from pipelines.separated.data_structures import StageResult, VisualizationData
            except ImportError:
                # StageResult가 없으면 간단하게 저장
                logging.warning("StageResult not available, saving as simple format")
                with open(output_file, 'wb') as f:
                    pickle.dump(tracking_results, f)
                logging.info(f"Stage2 tracking completed (simple format): {len(tracking_results)} frames → {output_file}")
                return {
                    "success": True,
                    "output_path": str(output_file),
                    "frames_processed": len(tracking_results)
                }

            # frame_poses_list를 사용할 수 있으면 그것을 사용, 없으면 tracking_results에서 생성
            if self.scorers and 'frame_poses_list' in locals() and frame_poses_list:
                poses_data = frame_poses_list
            else:
                # dict 형태를 FramePoses로 변환
                poses_data = []
                for frame_data in tracking_results:
                    frame_idx = frame_data['frame_idx']
                    persons_data = []

                    for person in frame_data['persons']:
                        if 'keypoints' in person and len(person['keypoints']) > 0:
                            keypoints = np.array(person['keypoints'])
                            if keypoints.ndim == 2:  # (17, 3) 형태
                                person_pose = PersonPose(
                                    person_id=person.get('person_id', person.get('track_id', 0)),
                                    bbox=person.get('bbox', [0, 0, 100, 100]),
                                    keypoints=keypoints,
                                    score=person.get('score', 1.0),
                                    track_id=person.get('track_id', 0)
                                )
                                persons_data.append(person_pose)

                    if persons_data:
                        frame_poses = FramePoses(
                            frame_idx=frame_idx,
                            persons=persons_data,
                            timestamp=float(frame_idx),
                            image_shape=(480, 640),
                            metadata={}
                        )
                        poses_data.append(frame_poses)

            # 기존 PKL 구조와 동일하게 VisualizationData로 저장
            video_name = Path(pkl_path).name.replace('_stage1_poses.pkl', '')
            stage_result = VisualizationData(
                video_name=video_name,
                frame_data=poses_data,  # 기본 프레임 데이터
                stage_info={
                    'stage': 'stage2',
                    'processing_time': 0,
                    'services': list(self.services) if self.services else []
                },
                poses_with_tracking=poses_data,  # Stage2 특화 데이터
                tracking_info={
                    'total_tracks': len(set(p.track_id for frame in poses_data for p in frame.persons if p.track_id is not None)),
                    'frames_processed': len(poses_data)
                }
            )

            with open(output_file, 'wb') as f:
                pickle.dump(stage_result, f)

            logging.info(f"Stage2 tracking completed: {len(tracking_results)} frames → {output_file}")
            return {
                "success": True,
                "output_path": str(output_file),
                "frames_processed": len(tracking_results)
            }

        except Exception as e:
            logging.error(f"Error in stage2 tracking: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def process_stage3_full_dataset(self, stage2_output_dir: str, output_dir: str) -> Dict[str, Any]:
        """Stage3 전체 데이터셋 생성 - 모든 PKL 파일을 수집하여 train/val/test 분할"""
        logging.info("Starting Stage3 full dataset creation...")

        try:
            import os
            from sklearn.model_selection import train_test_split

            # 모든 PKL 파일 수집
            stage2_dir = Path(stage2_output_dir)
            pkl_files = list(stage2_dir.glob("*.pkl"))

            if not pkl_files:
                return {"success": False, "error": f"No PKL files found in {stage2_output_dir}"}

            logging.info(f"Found {len(pkl_files)} PKL files to process")

            # 전체 데이터셋 엔트리 수집
            all_dataset_entries = []
            processed_count = 0
            skipped_count = 0

            for pkl_path in pkl_files:
                try:
                    with open(pkl_path, 'rb') as f:
                        pkl_data = pickle.load(f)

                    # VisualizationData에서 프레임 데이터 추출
                    frame_poses_list = self._extract_frame_poses_from_pkl(pkl_data, str(pkl_path))

                    if not frame_poses_list:
                        skipped_count += 1
                        continue

                    # 라벨 추출
                    label = self._extract_label_from_path(str(pkl_path))

                    # 데이터셋 엔트리 생성
                    entries = self._create_dataset_entries(frame_poses_list, pkl_path.stem, label)
                    all_dataset_entries.extend(entries)
                    processed_count += 1

                except Exception as e:
                    logging.warning(f"Error processing {pkl_path}: {e}")
                    skipped_count += 1
                    continue

            logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
            logging.info(f"Total dataset entries: {len(all_dataset_entries)}")

            if not all_dataset_entries:
                return {"success": False, "error": "No valid dataset entries found"}

            # Split ratios from config
            split_ratios = self.config.get('annotation', {}).get('stage3', {}).get('split_ratios', {
                'train': 0.7, 'val': 0.2, 'test': 0.1
            })

            # train/val/test 분할
            train_data, temp_data = train_test_split(
                all_dataset_entries,
                test_size=(split_ratios['val'] + split_ratios['test']),
                random_state=42,
                shuffle=True
            )

            # val/test 분할
            val_size = split_ratios['val'] / (split_ratios['val'] + split_ratios['test'])
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_size),
                random_state=42,
                shuffle=True
            )

            # 출력 디렉토리 생성
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 각 분할 저장
            datasets = {
                'train.pkl': train_data,
                'val.pkl': val_data,
                'test.pkl': test_data
            }

            for filename, dataset in datasets.items():
                output_file = output_path / filename
                with open(output_file, 'wb') as f:
                    pickle.dump(dataset, f)
                logging.info(f"Saved {filename}: {len(dataset)} entries")

            return {
                "success": True,
                "output_dir": str(output_path),
                "train_count": len(train_data),
                "val_count": len(val_data),
                "test_count": len(test_data),
                "total_entries": len(all_dataset_entries)
            }

        except Exception as e:
            logging.error(f"Error in Stage3 full dataset creation: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _extract_frame_poses_from_pkl(self, pkl_data, pkl_path: str):
        """PKL 데이터에서 프레임 포즈 데이터 추출"""
        from pipelines.separated.data_structures import VisualizationData

        if isinstance(pkl_data, VisualizationData):
            for attr_name in ['poses_with_tracking', 'frame_data', 'poses_with_scores', 'poses_only']:
                attr_value = getattr(pkl_data, attr_name, None)
                if attr_value and isinstance(attr_value, list) and len(attr_value) > 0:
                    return attr_value
            return None
        elif isinstance(pkl_data, list):
            return pkl_data
        else:
            return None

    def _extract_label_from_path(self, pkl_path: str) -> int:
        """경로에서 라벨 추출"""
        input_path = self.config.get('annotation', {}).get('input', '')
        if 'falldown' in pkl_path.lower() or 'falldown' in input_path.lower():
            return 1  # falldown 라벨
        elif 'fight' in pkl_path.lower() or 'fight' in input_path.lower():
            return 1  # fight 라벨
        return 0  # normal 라벨

    def _create_dataset_entries(self, frame_poses_list, video_name: str, label: int):
        """FramePoses 리스트에서 데이터셋 엔트리 생성"""

        dataset_entries = []

        for frame_poses in frame_poses_list:
            if not hasattr(frame_poses, 'persons') or not frame_poses.persons:
                continue

            for person in frame_poses.persons:
                if hasattr(person, 'keypoints') and person.keypoints is not None and len(person.keypoints) > 0:
                    keypoints = person.keypoints
                    if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2:  # (17, 3) 형태
                        entry = {
                            'frame_dir': video_name,
                            'label': label,
                            'img_shape': (480, 640),
                            'original_shape': (480, 640),
                            'total_frames': 1,
                            'keypoint': keypoints.reshape(1, 17, 3),  # (1, 17, 3)
                            'keypoint_score': keypoints[:, 2:3].reshape(1, 17, 1)
                        }
                        dataset_entries.append(entry)

        return dataset_entries

    def _process_stage3_dataset(self, pkl_data: dict, pkl_path: str, output_dir: str) -> Dict[str, Any]:
        """Stage3 dataset 생성 - 개별 비디오를 임시 파일로 저장"""
        logging.info(f"Processing Stage3 dataset for: {pkl_path}")

        try:
            import os

            # 파일명에서 비디오 이름 추출
            pkl_file = Path(pkl_path)
            video_name = pkl_file.name.replace('_tracking.pkl', '').replace('_stage2_poses.pkl', '')

            # 임시 디렉토리 생성 (개별 프로세스별)
            temp_dir = Path(output_dir) / "temp_stage3"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # 프로세스별 임시 파일 (PID 사용으로 중복 방지)
            process_id = os.getpid()
            temp_file = temp_dir / f"{video_name}_{process_id}_temp.pkl"

            # 간단한 데이터셋 변환 처리
            dataset_entries = []

            # VisualizationData 객체에서 프레임 데이터 추출
            from pipelines.separated.data_structures import VisualizationData

            if isinstance(pkl_data, VisualizationData):
                # VisualizationData 객체인 경우 - 모든 가능한 속성 확인
                frame_poses_list = None
                for attr_name in ['poses_with_tracking', 'frame_data', 'poses_with_scores', 'poses_only']:
                    attr_value = getattr(pkl_data, attr_name, None)
                    if attr_value and isinstance(attr_value, list) and len(attr_value) > 0:
                        frame_poses_list = attr_value
                        logging.info(f"Using {attr_name} from VisualizationData: {len(frame_poses_list)} frames")
                        break

                if frame_poses_list is None or len(frame_poses_list) == 0:
                    logging.warning(f"Empty VisualizationData, skipping: {pkl_path}")
                    return {"success": True, "skipped": True, "reason": "Empty VisualizationData"}
            elif isinstance(pkl_data, list):
                frame_poses_list = pkl_data
            else:
                logging.error(f"Unsupported PKL format in {pkl_path}: {type(pkl_data)}")
                return {"success": False, "error": "Unsupported PKL format"}

            # 폴더명에서 라벨 추출
            label = 0  # 기본값
            input_path = self.config.get('annotation', {}).get('input', '')
            if 'falldown' in pkl_path.lower() or 'falldown' in input_path.lower():
                label = 1  # falldown 라벨
            elif 'fight' in pkl_path.lower() or 'fight' in input_path.lower():
                label = 1  # fight 라벨
            # normal은 0으로 유지

            # FramePoses 객체에서 데이터 추출
            for frame_poses in frame_poses_list:
                frame_idx = frame_poses.frame_idx if hasattr(frame_poses, 'frame_idx') else 0
                persons = frame_poses.persons if hasattr(frame_poses, 'persons') else []

                for person in persons:
                    # PersonPose 객체에서 MMAction2 형식으로 변환
                    if hasattr(person, 'keypoints') and person.keypoints is not None and len(person.keypoints) > 0:
                        keypoints = person.keypoints
                        if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2:  # (17, 3) 형태
                            entry = {
                                'frame_dir': video_name,
                                'label': label,  # 폴더명 기반 라벨
                                'img_shape': (480, 640),
                                'original_shape': (480, 640),
                                'total_frames': 1,
                                'keypoint': keypoints.reshape(1, 17, 3),  # (1, 17, 3)
                                'keypoint_score': keypoints[:, 2:3].reshape(1, 17, 1)
                            }
                            dataset_entries.append(entry)

            if not dataset_entries:
                logging.warning(f"No valid dataset entries generated from {pkl_path}")
                return {"success": False, "error": "No valid dataset entries"}

            # 임시 파일로 저장 (나중에 통합 처리)
            temp_data = {
                'video_name': video_name,
                'dataset_entries': dataset_entries,
                'label': label,
                'pkl_path': pkl_path
            }

            with open(temp_file, 'wb') as f:
                pickle.dump(temp_data, f)

            logging.info(f"Stage3 temp file saved: {len(dataset_entries)} entries → {temp_file}")
            return {
                "success": True,
                "temp_file": str(temp_file),
                "temp_dir": str(temp_dir),
                "total_entries": len(dataset_entries),
                "label": label,
                "video_name": video_name
            }

        except Exception as e:
            logging.error(f"Error in stage3 dataset: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    @staticmethod
    def merge_stage3_temp_files(temp_dir: str, output_dir: str, split_ratios: dict = None) -> Dict[str, Any]:
        """Stage3 임시 파일들을 통합하여 최종 데이터셋 생성"""
        try:
            import random

            temp_dir_path = Path(temp_dir)
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            if not temp_dir_path.exists():
                return {"success": False, "error": f"Temp directory not found: {temp_dir}"}

            # 모든 임시 파일 수집
            temp_files = list(temp_dir_path.glob("*_temp.pkl"))
            if not temp_files:
                return {"success": False, "error": "No temp files found"}

            logging.info(f"Found {len(temp_files)} temp files to merge")

            # 모든 데이터 수집
            all_entries = []
            video_count = 0
            label_counts = {}

            for temp_file in temp_files:
                try:
                    with open(temp_file, 'rb') as f:
                        temp_data = pickle.load(f)

                    entries = temp_data.get('dataset_entries', [])
                    label = temp_data.get('label', 0)
                    video_name = temp_data.get('video_name', 'unknown')

                    all_entries.extend(entries)
                    video_count += 1

                    label_counts[label] = label_counts.get(label, 0) + len(entries)

                    logging.info(f"Merged {video_name}: {len(entries)} entries (label: {label})")

                except Exception as e:
                    logging.error(f"Error reading temp file {temp_file}: {e}")
                    continue

            if not all_entries:
                return {"success": False, "error": "No valid entries found in temp files"}

            # 데이터 셔플
            random.shuffle(all_entries)

            # Train/Val/Test 분할
            if split_ratios is None:
                split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

            total = len(all_entries)
            train_end = int(total * split_ratios['train'])
            val_end = int(total * (split_ratios['train'] + split_ratios['val']))

            train_data = all_entries[:train_end]
            val_data = all_entries[train_end:val_end]
            test_data = all_entries[val_end:]

            # 최종 파일 저장
            train_file = output_dir_path / "train.pkl"
            val_file = output_dir_path / "val.pkl"
            test_file = output_dir_path / "test.pkl"

            with open(train_file, 'wb') as f:
                pickle.dump(train_data, f)

            with open(val_file, 'wb') as f:
                pickle.dump(val_data, f)

            with open(test_file, 'wb') as f:
                pickle.dump(test_data, f)

            # 임시 파일 정리
            for temp_file in temp_files:
                temp_file.unlink()

            if temp_dir_path.exists() and not list(temp_dir_path.iterdir()):
                temp_dir_path.rmdir()

            logging.info(f"Stage3 dataset merge completed:")
            logging.info(f"  Videos processed: {video_count}")
            logging.info(f"  Total entries: {total}")
            logging.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            logging.info(f"  Label distribution: {label_counts}")
            logging.info(f"  Output: {output_dir_path}")

            return {
                "success": True,
                "output_dir": str(output_dir_path),
                "total_entries": total,
                "train_entries": len(train_data),
                "val_entries": len(val_data),
                "test_entries": len(test_data),
                "video_count": video_count,
                "label_counts": label_counts,
                "files": {
                    "train": str(train_file),
                    "val": str(val_file),
                    "test": str(test_file)
                }
            }

        except Exception as e:
            logging.error(f"Error merging stage3 temp files: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}


def create_dual_service_pipeline(config: Dict[str, Any]) -> Optional[DualServicePipeline]:
    """듀얼 서비스 파이프라인 생성 함수"""
    try:
        # 듀얼 서비스 설정 확인 (enabled=false일 때도 단일 서비스로 생성)
        dual_config = config.get('dual_service', {})
        services = dual_config.get('services', ['fight', 'falldown'])
        mode_name = "Dual" if dual_config.get('enabled', False) else "Single"
        logging.info(f"Creating {mode_name} service pipeline with services: {services}")
        
        # 파이프라인 생성 및 초기화
        pipeline = DualServicePipeline(config)
        if pipeline.initialize_pipeline():
            logging.info("Dual service pipeline created successfully")
            return pipeline
        else:
            logging.error("Failed to initialize dual service pipeline")
            return None
            
    except Exception as e:
        logging.error(f"Failed to create dual service pipeline: {e}")
        return None