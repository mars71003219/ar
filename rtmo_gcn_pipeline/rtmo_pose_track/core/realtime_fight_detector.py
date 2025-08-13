#!/usr/bin/env python3
"""
Realtime Fight Detector - 실시간 CCTV 폭력 탐지 시스템

기존 모듈화된 코드를 최대한 활용하여 RTSP/카메라 입력으로부터
실시간 폭력 탐지를 수행하는 통합 시스템입니다.
"""

import time
import json
import uuid
import threading
import numpy as np
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from .pose_extractor import EnhancedRTMOPoseExtractor
from .tracker import ByteTracker
from .scoring_system import EnhancedFightInvolvementScorer
from .realtime_window_manager import RealtimeWindowManager, WindowData, FrameData
from .realtime_decision_engine import RealtimeFightAnalyzer, DecisionResult, AlertLevel
from processing.rtsp_stream_processor import RTSPStreamProcessor


@dataclass
class DetectionEvent:
    """실시간 탐지 이벤트 데이터 구조"""
    event_id: str
    timestamp: str
    source_info: Dict[str, Any]
    event_summary: Dict[str, Any]
    observed_objects: List[Dict[str, Any]]
    confidence: float
    is_fight: bool


class RealtimeFightDetector:
    """
    실시간 CCTV 폭력 탐지 메인 클래스
    
    기존 모듈 구성:
    - EnhancedRTMOPoseExtractor: 포즈 추정
    - ByteTracker: 객체 추적 
    - EnhancedFightInvolvementScorer: 복합 점수 계산
    - STGCN 모델: 행동 분류 (MMAction2)
    - RTSPStreamProcessor: 실시간 스트림 입력
    - RealtimeWindowManager: 슬라이딩 윈도우 관리
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 시스템 설정
                {
                    'source': 'rtsp://...' or 0,
                    'detector_config': 'path/to/rtmo_config.py',
                    'detector_checkpoint': 'path/to/rtmo.pth',
                    'action_config': 'path/to/stgcn_config.py', 
                    'action_checkpoint': 'path/to/stgcn.pth',
                    'device': 'cuda:0',
                    'clip_len': 100,
                    'inference_stride': 50,
                    'classification_threshold': 0.5,
                    'stream_config': {...}
                }
        """
        self.config = config
        self.is_running = False
        
        # 이벤트 콜백
        self.event_callbacks: List[Callable[[DetectionEvent], None]] = []
        
        # 성능 모니터링
        self.stats = {
            'frames_processed': 0,
            'windows_processed': 0,
            'fight_detections': 0,
            'total_processing_time': 0.0,
            'avg_fps': 0.0
        }
        
        # 스레드 관리
        self.processing_thread = None
        self.result_queue = Queue(maxsize=100)
        
        # 컴포넌트 초기화
        self._init_components()
    
    def _init_components(self):
        """기존 모듈 구성 요소들 초기화"""
        print("Initializing realtime fight detection components...")
        
        # 1. RTSP 스트림 프로세서
        stream_config = self.config.get('stream_config', {})
        self.stream_processor = RTSPStreamProcessor(
            source=self.config['source'],
            config=stream_config
        )
        
        # 2. 포즈 추출기
        self.pose_extractor = EnhancedRTMOPoseExtractor(
            config_file=self.config['detector_config'],
            checkpoint=self.config['detector_checkpoint'],
            device=self.config['device']
        )
        
        # 3. 객체 추적기
        track_config = self.config.get('track_config', {})
        self.tracker = ByteTracker(
            high_thresh=track_config.get('track_high_thresh', 0.6),
            low_thresh=track_config.get('track_low_thresh', 0.1),
            max_disappeared=track_config.get('track_max_disappeared', 30),
            min_hits=track_config.get('track_min_hits', 3)
        )
        
        # 4. 점수 계산기
        scorer_config = self.config.get('scorer_config', {})
        self.scorer = EnhancedFightInvolvementScorer(config=scorer_config)
        
        # 5. 실시간 윈도우 관리자
        self.window_manager = RealtimeWindowManager(
            clip_len=self.config.get('clip_len', 100),
            inference_stride=self.config.get('inference_stride', 50),
            max_persons=self.config.get('max_persons', 4)
        )
        
        # 6. 실시간 의사결정 엔진
        decision_config = self.config.get('decision_config', {})
        decision_config.update({
            'consecutive_threshold': self.config.get('consecutive_threshold', 3),
            'fight_ratio_threshold': self.config.get('fight_ratio_threshold', 0.4),
            'classification_threshold': self.config.get('classification_threshold', 0.5)
        })
        self.decision_analyzer = RealtimeFightAnalyzer(decision_config)
        
        # 7. STGCN 모델 (MMAction2)
        self._init_action_model()
        
        print("All components initialized successfully")
    
    def _init_action_model(self):
        """STGCN++ 모델 초기화 (기존 코드 활용)"""
        try:
            from mmaction.apis import init_recognizer
            
            self.action_recognizer = init_recognizer(
                config=self.config['action_config'],
                checkpoint=self.config['action_checkpoint'],
                device=self.config['device']
            )
            print("STGCN++ model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load STGCN++ model: {e}")
            self.action_recognizer = None
    
    def add_event_callback(self, callback: Callable[[DetectionEvent], None]):
        """이벤트 콜백 추가"""
        self.event_callbacks.append(callback)
    
    def start_detection(self):
        """실시간 탐지 시작"""
        if self.is_running:
            print("Detection already running")
            return
        
        self.is_running = True
        
        # 스트림 시작
        self.stream_processor.start_capture()
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        print(f"Started realtime fight detection from {self.config['source']}")
    
    def stop_detection(self):
        """실시간 탐지 중지"""
        self.is_running = False
        
        if self.stream_processor:
            self.stream_processor.stop_capture()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        print("Stopped realtime fight detection")
    
    def _processing_loop(self):
        """메인 처리 루프"""
        print("Started processing loop")
        
        while self.is_running:
            try:
                # 프레임 읽기
                frame_data = self.stream_processor.read_frame(timeout=1.0)
                if frame_data is None:
                    continue
                
                frame_idx, frame = frame_data
                start_time = time.time()
                
                # 1. 포즈 추정
                pose_results = self._extract_poses(frame)
                if not pose_results:
                    continue
                
                # 2. 객체 추적
                track_results = self._apply_tracking(pose_results)
                
                # 3. 점수 계산
                scores = self._calculate_scores(pose_results, track_results)
                
                # 4. 윈도우 매니저에 프레임 데이터 추가
                window_ready = self.window_manager.add_frame_data(
                    frame_idx=frame_idx,
                    pose_results=pose_results,
                    track_results=track_results,
                    scores=scores
                )
                
                # 5. 윈도우가 준비되면 추론 실행
                if window_ready:
                    self._process_window()
                
                # 성능 통계 업데이트
                processing_time = time.time() - start_time
                self._update_stats(processing_time)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)
        
        print("Processing loop ended")
    
    def _extract_poses(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """프레임에서 포즈 추출"""
        try:
            # EnhancedRTMOPoseExtractor 활용
            results = self.pose_extractor.extract_poses_from_frame(frame)
            return results if results else []
            
        except Exception as e:
            print(f"Error extracting poses: {e}")
            return []
    
    def _apply_tracking(self, pose_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """객체 추적 적용"""
        try:
            # ByteTracker 활용
            detections = []
            for pose_data in pose_results:
                if 'bbox' in pose_data and 'score' in pose_data:
                    detections.append([
                        *pose_data['bbox'],  # x1, y1, x2, y2
                        pose_data['score']   # confidence
                    ])
            
            if detections:
                tracks = self.tracker.update(np.array(detections))
                
                # 추적 결과를 기존 형식으로 변환
                track_ids = []
                for track in tracks:
                    track_ids.append(int(track[4]))  # track_id
                
                return {
                    'track_ids': track_ids,
                    'tracks': tracks
                }
            
            return {'track_ids': [], 'tracks': []}
            
        except Exception as e:
            print(f"Error in tracking: {e}")
            return {'track_ids': [], 'tracks': []}
    
    def _calculate_scores(self, pose_results: List[Dict[str, Any]], 
                         track_results: Dict[str, Any]) -> Dict[int, float]:
        """복합 점수 계산"""
        try:
            # EnhancedFightInvolvementScorer 활용
            track_ids = track_results.get('track_ids', [])
            scores = {}
            
            for i, track_id in enumerate(track_ids):
                if i < len(pose_results):
                    # 개별 점수 계산 (기존 로직 활용)
                    pose_data = pose_results[i]
                    score = self.scorer.calculate_person_score(pose_data)
                    scores[track_id] = score
            
            return scores
            
        except Exception as e:
            print(f"Error calculating scores: {e}")
            return {}
    
    def _process_window(self):
        """윈도우 처리 및 실시간 의사결정"""
        try:
            # 윈도우 생성
            window_data = self.window_manager.create_window()
            if window_data is None:
                return
            
            # STGCN++ 모델로 행동 분류
            prediction_result = self._classify_action(window_data)
            
            if prediction_result:
                # 실시간 의사결정 엔진으로 최종 판단
                decision_result, analysis_data = self.decision_analyzer.analyze_window(
                    window_data, prediction_result
                )
                
                # 이벤트 생성 (의사결정 결과 반영)
                event = self._create_detection_event(
                    window_data, prediction_result, decision_result
                )
                
                # 콜백 호출 (알림 레벨에 따라 차별화)
                self._notify_event(event, decision_result.alert_level)
                
                # 통계 업데이트
                self.stats['windows_processed'] += 1
                if decision_result.is_fight:  # 최종 판단 기준
                    self.stats['fight_detections'] += 1
                
                # 의사결정 상세 로깅 (디버그 모드에서)
                if self.config.get('debug_mode', False):
                    print(f"Window {window_data.window_idx}: "
                          f"STGCN={prediction_result['is_fight']}({prediction_result['confidence']:.3f}), "
                          f"Decision={decision_result.is_fight}({decision_result.confidence:.3f}), "
                          f"Alert={decision_result.alert_level.value}, "
                          f"Consec={decision_result.consecutive_count}, "
                          f"Ratio={decision_result.recent_fight_ratio:.3f}, "
                          f"Reason='{decision_result.reason}'")
            
        except Exception as e:
            print(f"Error processing window: {e}")
    
    def _classify_action(self, window_data: WindowData) -> Optional[Dict[str, Any]]:
        """행동 분류 (STGCN++ 활용)"""
        if not self.action_recognizer:
            return None
        
        try:
            # 기존 추론 로직 활용
            annotation = window_data.annotation
            
            # STGCN++ 입력 형식으로 변환
            stgcn_input = self._convert_to_stgcn_format(annotation)
            if stgcn_input is None:
                return None
            
            # 추론 실행
            from mmaction.apis import inference_recognizer
            
            result = inference_recognizer(
                self.action_recognizer,
                stgcn_input
            )
            
            # 결과 파싱
            if hasattr(result, 'pred_score'):
                fight_score = float(result.pred_score)
                is_fight = fight_score > self.config.get('classification_threshold', 0.5)
                
                return {
                    'fight_score': fight_score,
                    'is_fight': is_fight,
                    'confidence': fight_score if is_fight else (1.0 - fight_score)
                }
            
            return None
            
        except Exception as e:
            print(f"Error in action classification: {e}")
            return None
    
    def _convert_to_stgcn_format(self, annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """STGCN++ 입력 형식으로 변환 (기존 로직 활용)"""
        try:
            persons = annotation.get('persons', {})
            if not persons:
                return None
            
            # 기존 변환 로직 적용
            total_frames = annotation.get('total_frames', 100)
            
            # 최대 4명까지만 처리
            selected_persons = list(persons.items())[:4]
            
            keypoint_data = []
            for person_key, person_data in selected_persons:
                keypoints = person_data.get('keypoint')
                if keypoints is not None and hasattr(keypoints, 'shape'):
                    keypoint_data.append(keypoints)
            
            if not keypoint_data:
                return None
            
            # 패딩 또는 자르기
            while len(keypoint_data) < 4:
                # 제로 패딩
                keypoint_data.append(np.zeros((1, total_frames, 17, 2), dtype=np.float32))
            
            keypoint_data = keypoint_data[:4]  # 최대 4명
            
            # numpy 배열로 변환
            final_data = np.concatenate(keypoint_data, axis=0)  # (4, T, 17, 2)
            
            return {
                'keypoint': final_data,
                'total_frames': total_frames,
                'img_shape': (1080, 1920),  # 기본값
                'original_shape': (1080, 1920),
                'label': 0  # placeholder
            }
            
        except Exception as e:
            print(f"Error converting to STGCN format: {e}")
            return None
    
    def _create_detection_event(self, window_data: WindowData, 
                              prediction: Dict[str, Any],
                              decision: Optional[DecisionResult] = None) -> DetectionEvent:
        """탐지 이벤트 생성 (MODEL_OUTPUT_SCHEMA.md 준수)"""
        event_id = f"evt-{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # 소스 정보
        source_info = {
            'type': 'stream',
            'sourceId': str(self.config['source']),
            'streamUrl': self.config['source'] if isinstance(self.config['source'], str) else None,
            'clipStartFrame': window_data.start_frame,
            'clipEndFrame': window_data.end_frame
        }
        
        # 이벤트 요약 (의사결정 결과 우선 사용)
        final_is_fight = decision.is_fight if decision else prediction['is_fight']
        final_confidence = decision.confidence if decision else prediction['confidence']
        
        event_summary = {
            'isFight': final_is_fight,
            'confidence': final_confidence,
            'label': 'Fight' if final_is_fight else 'NonFight',
            'totalPersons': len(window_data.annotation.get('persons', {})),
            'rankedPersons': min(4, len(window_data.annotation.get('persons', {})))
        }
        
        # 의사결정 정보 추가 (확장 필드)
        if decision:
            event_summary.update({
                'alertLevel': decision.alert_level.value,
                'consecutiveCount': decision.consecutive_count,
                'recentFightRatio': decision.recent_fight_ratio,
                'stgcnPrediction': prediction['is_fight'],
                'stgcnConfidence': prediction['confidence'],
                'decisionReason': decision.reason
            })
        
        # 관찰된 객체들
        observed_objects = self._create_observed_objects(window_data)
        
        return DetectionEvent(
            event_id=event_id,
            timestamp=timestamp,
            source_info=source_info,
            event_summary=event_summary,
            observed_objects=observed_objects,
            confidence=final_confidence,
            is_fight=final_is_fight
        )
    
    def _create_observed_objects(self, window_data: WindowData) -> List[Dict[str, Any]]:
        """관찰된 객체 목록 생성"""
        objects = []
        persons = window_data.annotation.get('persons', {})
        
        rank = 1
        for person_key, person_data in persons.items():
            track_id = person_data.get('track_id', -1)
            
            # 바운딩 박스 정보 (프레임별)
            bounding_boxes = []
            for frame in window_data.frames:
                if (frame.track_results and 'track_ids' in frame.track_results 
                    and track_id in frame.track_results['track_ids']):
                    # 바운딩 박스 추출 로직
                    bbox = self._extract_bbox_for_track(frame, track_id)
                    if bbox:
                        bounding_boxes.append({
                            'frameIndex': frame.frame_idx - window_data.start_frame,
                            'box2d': bbox
                        })
            
            # 키포인트 정보 (프레임별)
            keypoints_list = []
            for frame in window_data.frames:
                pose_data = self._get_pose_for_track(frame, track_id)
                if pose_data:
                    keypoints_list.append({
                        'frameIndex': frame.frame_idx - window_data.start_frame,
                        'pose2d': pose_data.get('keypoints', []),
                        'avgConfidence': pose_data.get('score', 0.0)
                    })
            
            obj = {
                'objectId': f'person-track-{track_id}',
                'label': 'Person',
                'rank': rank,
                'compositeScore': person_data.get('score', 0.0),
                'boundingBoxes': bounding_boxes,
                'keypoints': keypoints_list
            }
            
            objects.append(obj)
            rank += 1
        
        return objects
    
    def _extract_bbox_for_track(self, frame: FrameData, track_id: int) -> Optional[List[float]]:
        """특정 트랙 ID의 바운딩 박스 추출"""
        try:
            if not frame.track_results or 'track_ids' not in frame.track_results:
                return None
            
            track_ids = frame.track_results['track_ids']
            if track_id not in track_ids:
                return None
            
            track_idx = track_ids.index(track_id)
            if track_idx < len(frame.pose_results):
                pose_data = frame.pose_results[track_idx]
                return pose_data.get('bbox', [])
            
            return None
            
        except Exception as e:
            print(f"Error extracting bbox: {e}")
            return None
    
    def _get_pose_for_track(self, frame: FrameData, track_id: int) -> Optional[Dict[str, Any]]:
        """특정 트랙 ID의 포즈 데이터 반환"""
        try:
            if not frame.track_results or 'track_ids' not in frame.track_results:
                return None
            
            track_ids = frame.track_results['track_ids']
            if track_id not in track_ids:
                return None
            
            track_idx = track_ids.index(track_id)
            if track_idx < len(frame.pose_results):
                return frame.pose_results[track_idx]
            
            return None
            
        except Exception as e:
            print(f"Error getting pose data: {e}")
            return None
    
    def _notify_event(self, event: DetectionEvent, alert_level: Optional[AlertLevel] = None):
        """이벤트 콜백 호출 (알림 레벨에 따른 차별화)"""
        for callback in self.event_callbacks:
            try:
                # 콜백이 alert_level을 지원하는지 확인
                if hasattr(callback, '__call__') and len(getattr(callback, '__code__', getattr(callback.__call__, '__code__', type('', (), {'co_argcount': 1})())).co_varnames) > 1:
                    callback(event, alert_level)
                else:
                    callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")
    
    def _update_stats(self, processing_time: float):
        """통계 업데이트"""
        self.stats['frames_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # FPS 계산 (최근 100프레임 기준)
        if self.stats['frames_processed'] > 0:
            self.stats['avg_fps'] = 1.0 / (self.stats['total_processing_time'] / self.stats['frames_processed'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        stream_stats = self.stream_processor.get_stream_info()
        window_stats = self.window_manager.get_statistics()
        
        return {
            'detector_stats': self.stats.copy(),
            'stream_stats': stream_stats,
            'window_stats': window_stats,
            'is_running': self.is_running
        }
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.start_detection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.stop_detection()


class EventHandler:
    """이벤트 핸들러 기본 클래스"""
    
    def handle_fight_detected(self, event: DetectionEvent):
        """폭력 탐지 시 처리"""
        print(f"FIGHT DETECTED! Event ID: {event.event_id}")
        print(f"Confidence: {event.confidence:.3f}")
        print(f"Persons involved: {event.event_summary['totalPersons']}")
    
    def handle_normal_activity(self, event: DetectionEvent):
        """정상 활동 시 처리"""
        if event.event_summary['confidence'] > 0.8:  # 높은 신뢰도만 로깅
            print(f"Normal activity detected (confidence: {event.confidence:.3f})")


class RealtimeEventLogger(EventHandler):
    """실시간 이벤트 로거 (의사결정 정보 포함)"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
    
    def __call__(self, event: DetectionEvent, alert_level: Optional[AlertLevel] = None):
        """이벤트 로깅 (알림 레벨 정보 포함)"""
        try:
            # JSON 형식으로 로그 저장
            log_entry = asdict(event)
            
            # 알림 레벨 정보 추가
            if alert_level:
                log_entry['alert_level'] = alert_level.value
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # 콘솔 출력 (알림 레벨별 차별화)
            if event.is_fight:
                self.handle_fight_detected(event, alert_level)
            else:
                self.handle_normal_activity(event, alert_level)
                
        except Exception as e:
            print(f"Error logging event: {e}")
    
    def handle_fight_detected(self, event: DetectionEvent, alert_level: Optional[AlertLevel] = None):
        """폭력 탐지 시 처리 (확장)"""
        alert_symbols = {
            AlertLevel.SUSPICIOUS: "⚠️ ",
            AlertLevel.WARNING: "🔥",
            AlertLevel.CRITICAL: "🚨🚨",
            AlertLevel.COOLING_DOWN: "🔄"
        }
        
        symbol = alert_symbols.get(alert_level, "🚨") if alert_level else "🚨"
        level_name = alert_level.value.upper() if alert_level else "ALERT"
        
        print(f"\n{symbol} {level_name} - FIGHT DETECTED!")
        print(f"   Event ID: {event.event_id[:8]}...")
        print(f"   Confidence: {event.confidence:.3f}")
        print(f"   Persons involved: {event.event_summary['totalPersons']}")
        
        # 의사결정 세부 정보 출력
        if 'consecutiveCount' in event.event_summary:
            print(f"   Consecutive count: {event.event_summary['consecutiveCount']}")
            print(f"   Fight ratio: {event.event_summary['recentFightRatio']:.3f}")
            print(f"   Reason: {event.event_summary.get('decisionReason', 'N/A')}")
        
        # 해제 진행상황 표시 (COOLING_DOWN 레벨일 때)
        if alert_level == AlertLevel.COOLING_DOWN and 'recovery_progress' in event.event_summary:
            recovery_progress = event.event_summary.get('recovery_progress', 0.0)
            stability_score = event.event_summary.get('stability_score', 0.0)
            print(f"   Recovery progress: {recovery_progress:.1%}")
            print(f"   Stability score: {stability_score:.1%}")
        
        print(f"   Time: {event.timestamp}")
    
    def handle_normal_activity(self, event: DetectionEvent, alert_level: Optional[AlertLevel] = None):
        """정상 활동 시 처리 (확장)"""
        # SUSPICIOUS 레벨도 로깅
        if alert_level == AlertLevel.SUSPICIOUS:
            print(f"⚠️  Suspicious activity detected (confidence: {event.confidence:.3f})")
        elif alert_level == AlertLevel.COOLING_DOWN:
            print(f"🔄 Alert cooling down - situation improving")
        elif event.confidence > 0.8:  # 높은 신뢰도 정상 활동
            print(f"✅ Normal activity confirmed (confidence: {event.confidence:.3f})")
        # 기타 정상 활동은 조용히 처리


# 사용 예시 및 테스트 코드
if __name__ == "__main__":
    # 테스트 설정
    test_config = {
        'source': 0,  # 웹캠 또는 'rtsp://...'
        'detector_config': 'configs/rtmo_config.py',
        'detector_checkpoint': 'checkpoints/rtmo.pth',
        'action_config': 'configs/stgcn_config.py',
        'action_checkpoint': 'checkpoints/stgcn.pth',
        'device': 'cuda:0',
        'clip_len': 100,
        'inference_stride': 50,
        'classification_threshold': 0.5,
        'stream_config': {
            'buffer_size': 30,
            'target_fps': 15,
            'frame_skip': 2
        }
    }
    
    # 이벤트 로거
    logger = RealtimeEventLogger('realtime_events.log')
    
    # 탐지기 시작
    with RealtimeFightDetector(test_config) as detector:
        detector.add_event_callback(logger)
        
        print("Realtime fight detection started. Press Ctrl+C to stop.")
        
        try:
            while True:
                stats = detector.get_statistics()
                print(f"\rProcessed: {stats['detector_stats']['frames_processed']} frames, "
                      f"FPS: {stats['detector_stats']['avg_fps']:.1f}, "
                      f"Fights: {stats['detector_stats']['fight_detections']}", end='')
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping detection...")