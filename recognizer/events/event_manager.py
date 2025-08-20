"""
이벤트 관리 시스템
폭력 탐지 이벤트의 발생, 지속, 해제를 관리
"""
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from .event_types import EventType, EventData
from .event_logger import EventLogger

@dataclass
class EventConfig:
    """이벤트 관리 설정"""
    # 이벤트 발생 조건
    alert_threshold: float = 0.7  # 폭력 탐지 신뢰도 임계값
    min_consecutive_detections: int = 3  # 연속 탐지 최소 횟수
    
    # 이벤트 해제 조건
    normal_threshold: float = 0.5  # 정상 상태 신뢰도 임계값
    min_consecutive_normal: int = 5  # 연속 정상 최소 횟수
    
    # 시간 기반 조건
    min_event_duration: float = 2.0  # 최소 이벤트 지속 시간 (초)
    max_event_duration: float = 300.0  # 최대 이벤트 지속 시간 (초)
    cooldown_duration: float = 10.0  # 이벤트 쿨다운 시간 (초)
    
    # 알림 설정
    enable_ongoing_alerts: bool = True  # 진행 중 알림 활성화
    ongoing_alert_interval: float = 30.0  # 진행 중 알림 간격 (초)
    
    # 로그 설정
    save_event_log: bool = True  # 이벤트 로그 저장 여부
    event_log_format: str = "json"  # 로그 형식 (json/csv)
    event_log_path: str = "output/event_logs"  # 이벤트 로그 저장 경로

class EventManager:
    """이벤트 관리 시스템"""
    
    def __init__(self, config: EventConfig):
        self.config = config
        
        # 이벤트 상태
        self.current_event_active = False
        self.current_event_start_time = None
        self.current_event_start_window = None
        self.last_event_end_time = None
        self.last_ongoing_alert_time = None
        
        # 연속 탐지 카운터
        self.consecutive_violence = 0
        self.consecutive_normal = 0
        
        # 이벤트 히스토리
        self.event_history: List[EventData] = []
        
        # 콜백 함수들
        self.event_callbacks: Dict[EventType, List[Callable]] = {
            EventType.VIOLENCE_START: [],
            EventType.VIOLENCE_END: [],
            EventType.VIOLENCE_ONGOING: [],
            EventType.NORMAL: []
        }
        
        # 이벤트 로거 초기화
        self.logger = None
        if self.config.save_event_log:
            self.logger = EventLogger(
                log_path=self.config.event_log_path,
                log_format=self.config.event_log_format,
                enable_logging=True
            )
            logging.info(f"Event logger initialized: {self.config.event_log_format} format")
        
        logging.info(f"EventManager initialized with config: {self.config}")
    
    def add_event_callback(self, event_type: EventType, callback: Callable):
        """이벤트 콜백 추가"""
        self.event_callbacks[event_type].append(callback)
        logging.info(f"Added callback for event type: {event_type.value}")
    
    def remove_event_callback(self, event_type: EventType, callback: Callable):
        """이벤트 콜백 제거"""
        if callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
            logging.info(f"Removed callback for event type: {event_type.value}")
    
    def _fire_event_callbacks(self, event_data: EventData):
        """이벤트 콜백 실행"""
        callbacks = self.event_callbacks.get(event_data.event_type, [])
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logging.error(f"Error in event callback for {event_data.event_type.value}: {e}")
    
    def process_classification_result(self, result: Dict[str, Any]) -> Optional[EventData]:
        """
        분류 결과를 처리하여 이벤트 발생 여부 판단
        
        Args:
            result: 분류 결과 딕셔너리
                - window_id: 윈도우 ID
                - prediction: 예측 결과 (violence/normal)
                - confidence: 신뢰도
                - timestamp: 타임스탬프
                - frame_number: 프레임 번호 (선택적)
        
        Returns:
            발생한 이벤트 데이터 (없으면 None)
        """
        window_id = result.get('window_id', 0)
        prediction = result.get('prediction', 'normal')
        confidence = result.get('confidence', 0.0)
        timestamp = result.get('timestamp', time.time())
        frame_number = result.get('frame_number')
        
        event_data = None
        
        # 폭력 탐지 처리
        if prediction == 'violence' and confidence >= self.config.alert_threshold:
            logging.debug(f"Violence detected: window={window_id}, confidence={confidence:.3f}, threshold={self.config.alert_threshold}")
            event_data = self._handle_violence_detection(
                window_id, confidence, prediction, timestamp, frame_number
            )
        else:
            # 정상 상태 처리 (violence가 아니거나 임계값 미달)
            if prediction == 'violence':
                logging.debug(f"Violence below threshold: window={window_id}, confidence={confidence:.3f} < {self.config.alert_threshold}")
            event_data = self._handle_normal_detection(
                window_id, confidence, prediction, timestamp, frame_number
            )
        
        # 진행 중 알림 체크
        if self.current_event_active and self.config.enable_ongoing_alerts:
            ongoing_event = self._check_ongoing_alert(window_id, timestamp)
            if ongoing_event and not event_data:  # 다른 이벤트가 없을 때만
                event_data = ongoing_event
        
        # 최대 지속 시간 체크
        if self.current_event_active:
            max_duration_event = self._check_max_duration(window_id, timestamp)
            if max_duration_event:
                event_data = max_duration_event
        
        return event_data
    
    def _handle_violence_detection(self, window_id: int, confidence: float, 
                                 prediction: str, timestamp: float, 
                                 frame_number: Optional[int] = None) -> Optional[EventData]:
        """폭력 탐지 처리"""
        self.consecutive_violence += 1
        self.consecutive_normal = 0
        
        # 쿨다운 시간 체크
        if (self.last_event_end_time and 
            timestamp - self.last_event_end_time < self.config.cooldown_duration):
            logging.debug(f"Violence detection during cooldown period (remaining: {self.config.cooldown_duration - (timestamp - self.last_event_end_time):.1f}s)")
            return None
        
        # 이벤트 시작 조건 체크
        if (not self.current_event_active and 
            self.consecutive_violence >= self.config.min_consecutive_detections):
            
            self.current_event_active = True
            self.current_event_start_time = timestamp
            self.current_event_start_window = window_id
            
            event_data = EventData(
                event_type=EventType.VIOLENCE_START,
                timestamp=timestamp,
                window_id=window_id,
                confidence=confidence,
                prediction=prediction,
                frame_number=frame_number,
                metadata={
                    'consecutive_detections': self.consecutive_violence,
                    'start_window': window_id
                }
            )
            
            self.event_history.append(event_data)
            self._fire_event_callbacks(event_data)
            
            # 이벤트 로그
            if self.logger:
                self.logger.log_event(event_data)
            
            logging.warning(f"[EVENT] Violence detected! Window {window_id}, confidence: {confidence:.3f}, consecutive: {self.consecutive_violence}")
            return event_data
        
        return None
    
    def _handle_normal_detection(self, window_id: int, confidence: float,
                               prediction: str, timestamp: float,
                               frame_number: Optional[int] = None) -> Optional[EventData]:
        """정상 상태 처리"""
        if prediction == 'normal' or confidence < self.config.normal_threshold:
            self.consecutive_normal += 1
            self.consecutive_violence = 0
        
        # 이벤트 해제 조건 체크
        if (self.current_event_active and 
            self.consecutive_normal >= self.config.min_consecutive_normal):
            
            # 최소 지속 시간 체크
            event_duration = timestamp - self.current_event_start_time
            if event_duration >= self.config.min_event_duration:
                
                event_data = EventData(
                    event_type=EventType.VIOLENCE_END,
                    timestamp=timestamp,
                    window_id=window_id,
                    confidence=confidence,
                    prediction=prediction,
                    frame_number=frame_number,
                    duration=event_duration,
                    metadata={
                        'consecutive_normal': self.consecutive_normal,
                        'start_window': self.current_event_start_window,
                        'end_window': window_id
                    }
                )
                
                self.event_history.append(event_data)
                self._fire_event_callbacks(event_data)
                
                # 상태 리셋
                self.current_event_active = False
                self.last_event_end_time = timestamp
                self.current_event_start_time = None
                self.current_event_start_window = None
                self.last_ongoing_alert_time = None
                
                logging.info(f"[EVENT] Violence ended. Window {window_id}, duration: {event_duration:.1f}s, consecutive normal: {self.consecutive_normal}")
                return event_data
            else:
                logging.debug(f"Event too short ({event_duration:.1f}s < {self.config.min_event_duration}s), waiting for minimum duration")
        
        return None
    
    def _check_ongoing_alert(self, window_id: int, timestamp: float) -> Optional[EventData]:
        """진행 중 알림 체크"""
        if not self.current_event_active:
            return None
        
        # 첫 진행 중 알림이거나 알림 간격이 지났을 때
        if (not self.last_ongoing_alert_time or 
            timestamp - self.last_ongoing_alert_time >= self.config.ongoing_alert_interval):
            
            event_duration = timestamp - self.current_event_start_time
            
            event_data = EventData(
                event_type=EventType.VIOLENCE_ONGOING,
                timestamp=timestamp,
                window_id=window_id,
                confidence=0.0,  # 진행 중 알림은 신뢰도 없음
                prediction='violence',
                duration=event_duration,
                metadata={
                    'start_window': self.current_event_start_window,
                    'current_window': window_id,
                    'ongoing_duration': event_duration
                }
            )
            
            self.last_ongoing_alert_time = timestamp
            self.event_history.append(event_data)
            self._fire_event_callbacks(event_data)
            
            # 이벤트 로그
            if self.logger:
                self.logger.log_event(event_data)
            
            logging.info(f"[EVENT] Violence ongoing. Window {window_id}, duration: {event_duration:.1f}s")
            return event_data
        
        return None
    
    def _check_max_duration(self, window_id: int, timestamp: float) -> Optional[EventData]:
        """최대 지속 시간 체크"""
        if not self.current_event_active:
            return None
        
        event_duration = timestamp - self.current_event_start_time
        if event_duration >= self.config.max_event_duration:
            
            event_data = EventData(
                event_type=EventType.VIOLENCE_END,
                timestamp=timestamp,
                window_id=window_id,
                confidence=0.0,
                prediction='violence',
                duration=event_duration,
                metadata={
                    'reason': 'max_duration_exceeded',
                    'start_window': self.current_event_start_window,
                    'end_window': window_id
                }
            )
            
            self.event_history.append(event_data)
            self._fire_event_callbacks(event_data)
            
            # 이벤트 로그
            if self.logger:
                self.logger.log_event(event_data)
            
            # 상태 리셋
            self.current_event_active = False
            self.last_event_end_time = timestamp
            self.current_event_start_time = None
            self.current_event_start_window = None
            self.last_ongoing_alert_time = None
            self.consecutive_violence = 0
            self.consecutive_normal = 0
            
            logging.warning(f"[EVENT] Violence ended due to max duration ({event_duration:.1f}s). Window {window_id}")
            return event_data
        
        return None
    
    def get_event_status(self) -> Dict[str, Any]:
        """현재 이벤트 상태 반환"""
        current_duration = None
        if self.current_event_active and self.current_event_start_time:
            current_duration = time.time() - self.current_event_start_time
        
        return {
            'event_active': self.current_event_active,
            'event_start_time': self.current_event_start_time,
            'event_duration': current_duration,
            'consecutive_violence': self.consecutive_violence,
            'consecutive_normal': self.consecutive_normal,
            'total_events': len(self.event_history),
            'last_event_end_time': self.last_event_end_time
        }
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """이벤트 히스토리 반환"""
        history = self.event_history
        if limit:
            history = history[-limit:]
        return [event.to_dict() for event in history]
    
    def reset(self):
        """이벤트 관리자 리셋"""
        # 현재 세션 종료 (있다면)
        if self.logger:
            final_stats = {
                'total_events': len(self.event_history),
                'event_active_at_reset': self.current_event_active,
                'consecutive_violence': self.consecutive_violence,
                'consecutive_normal': self.consecutive_normal
            }
            self.logger.finalize_session(final_stats)
        
        self.current_event_active = False
        self.current_event_start_time = None
        self.current_event_start_window = None
        self.last_event_end_time = None
        self.last_ongoing_alert_time = None
        self.consecutive_violence = 0
        self.consecutive_normal = 0
        self.event_history.clear()
        
        # 새 세션용 로거 재초기화
        if self.config.save_event_log:
            self.logger = EventLogger(
                log_path=self.config.event_log_path,
                log_format=self.config.event_log_format,
                enable_logging=True
            )
        
        logging.info("EventManager reset completed")
    
    def finalize_session(self, final_stats: Optional[Dict[str, Any]] = None):
        """세션 종료 처리"""
        if self.logger:
            stats = {
                'total_events': len(self.event_history),
                'event_active_at_end': self.current_event_active,
                'consecutive_violence': self.consecutive_violence,
                'consecutive_normal': self.consecutive_normal
            }
            if final_stats:
                stats.update(final_stats)
            
            self.logger.finalize_session(stats)
            logging.info(f"Event session finalized with {len(self.event_history)} events")
    
    def get_logger_stats(self) -> Dict[str, Any]:
        """로거 통계 반환"""
        if self.logger:
            return self.logger.get_session_stats()
        return {}