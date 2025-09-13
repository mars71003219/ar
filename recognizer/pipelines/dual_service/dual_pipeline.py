"""
듀얼 서비스 파이프라인 래퍼
Fight와 Falldown 두 서비스를 동시에 실행하는 파이프라인
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys

# recognizer 모듈 경로 추가
recognizer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(recognizer_root))

from utils.factory import ModuleFactory
from utils.data_structure import PersonPose, FramePoses, WindowAnnotation, ClassificationResult
from pipelines.base import BasePipeline
from visualization.inference_visualizer import InferenceVisualizer


class DualServicePipeline(BasePipeline):
    """듀얼 서비스 파이프라인 - Fight + Falldown 동시 처리"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.config = config
        self.dual_config = config.get('dual_service', {})
        
        # 듀얼 서비스 활성화 확인
        if not self.dual_config.get('enabled', False):
            raise ValueError("Dual service is not enabled in config")
        
        # 서비스 목록
        self.services = self.dual_config.get('services', ['fight', 'falldown'])
        self.priority = self.dual_config.get('priority', {'fight': 1, 'falldown': 2})
        
        logging.info(f"Dual service pipeline initialized with services: {self.services}")
        
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
            import traceback
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
            'priority': self.priority,
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
            window_name = "Violence Detection - DUAL SERVICE"
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
                        if none_frame_count >= max_none_frames or not input_manager.is_running:
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
                        import traceback
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
                        import traceback
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
                import traceback
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
            import traceback
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


def create_dual_service_pipeline(config: Dict[str, Any]) -> Optional[DualServicePipeline]:
    """듀얼 서비스 파이프라인 생성 함수"""
    try:
        # 듀얼 서비스 설정 확인
        dual_config = config.get('dual_service', {})
        if not dual_config.get('enabled', False):
            logging.warning("Dual service is not enabled, returning None")
            return None
        
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