"""
이벤트 로그 저장 기능
JSON 및 CSV 형식으로 이벤트 히스토리를 저장
"""
import json
import csv
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from .event_types import EventData

class EventLogger:
    """이벤트 로거 클래스"""
    
    def __init__(self, log_path: str = "output/event_logs", 
                 log_format: str = "json", 
                 enable_logging: bool = True):
        self.log_path = Path(log_path)
        self.log_format = log_format.lower()
        self.enable_logging = enable_logging
        self.session_start = datetime.now()
        
        if self.enable_logging:
            self._setup_log_directory()
            self._init_log_file()
    
    def _setup_log_directory(self):
        """로그 디렉터리 생성"""
        try:
            self.log_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Event log directory created: {self.log_path}")
        except Exception as e:
            logging.error(f"Failed to create event log directory: {e}")
            self.enable_logging = False
    
    def _init_log_file(self):
        """로그 파일 초기화"""
        if not self.enable_logging:
            return
        
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        if self.log_format == "json":
            self.log_file = self.log_path / f"events_{timestamp}.json"
            self._init_json_log()
        elif self.log_format == "csv":
            self.log_file = self.log_path / f"events_{timestamp}.csv"
            self._init_csv_log()
        else:
            logging.error(f"Unsupported log format: {self.log_format}")
            self.enable_logging = False
    
    def _init_json_log(self):
        """JSON 로그 파일 초기화"""
        try:
            initial_data = {
                "session_info": {
                    "start_time": self.session_start.isoformat(),
                    "format_version": "1.0"
                },
                "events": []
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"JSON event log initialized: {self.log_file}")
            
        except Exception as e:
            logging.error(f"Failed to initialize JSON log: {e}")
            self.enable_logging = False
    
    def _init_csv_log(self):
        """CSV 로그 파일 초기화"""
        try:
            fieldnames = [
                'timestamp', 'event_type', 'window_id', 'confidence', 
                'prediction', 'duration', 'frame_number', 'metadata'
            ]
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # 세션 정보를 첫 번째 행에 기록
                writer.writerow({
                    'timestamp': self.session_start.isoformat(),
                    'event_type': 'session_start',
                    'window_id': 0,
                    'confidence': 0.0,
                    'prediction': 'session_info',
                    'duration': None,
                    'frame_number': None,
                    'metadata': f"Session started at {self.session_start.isoformat()}"
                })
            
            logging.info(f"CSV event log initialized: {self.log_file}")
            
        except Exception as e:
            logging.error(f"Failed to initialize CSV log: {e}")
            self.enable_logging = False
    
    def log_event(self, event_data: EventData):
        """단일 이벤트 로그"""
        if not self.enable_logging:
            return
        
        try:
            if self.log_format == "json":
                self._log_event_json(event_data)
            elif self.log_format == "csv":
                self._log_event_csv(event_data)
                
        except Exception as e:
            logging.error(f"Failed to log event: {e}")
    
    def _log_event_json(self, event_data: EventData):
        """JSON 형식으로 이벤트 로그"""
        try:
            # 기존 데이터 읽기
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이벤트 추가
            event_dict = event_data.to_dict()
            event_dict['logged_at'] = datetime.now().isoformat()
            data['events'].append(event_dict)
            
            # 파일 업데이트
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to log JSON event: {e}")
    
    def _log_event_csv(self, event_data: EventData):
        """CSV 형식으로 이벤트 로그"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'event_type', 'window_id', 'confidence', 
                    'prediction', 'duration', 'frame_number', 'metadata'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 메타데이터를 문자열로 변환
                metadata_str = json.dumps(event_data.metadata) if event_data.metadata else None
                
                writer.writerow({
                    'timestamp': datetime.fromtimestamp(event_data.timestamp).isoformat(),
                    'event_type': event_data.event_type.value,
                    'window_id': event_data.window_id,
                    'confidence': event_data.confidence,
                    'prediction': event_data.prediction,
                    'duration': event_data.duration,
                    'frame_number': event_data.frame_number,
                    'metadata': metadata_str
                })
                
        except Exception as e:
            logging.error(f"Failed to log CSV event: {e}")
    
    def log_events_batch(self, events: List[EventData]):
        """배치 이벤트 로그"""
        if not self.enable_logging or not events:
            return
        
        try:
            if self.log_format == "json":
                self._log_events_batch_json(events)
            elif self.log_format == "csv":
                self._log_events_batch_csv(events)
                
            logging.info(f"Logged {len(events)} events to {self.log_file}")
            
        except Exception as e:
            logging.error(f"Failed to log event batch: {e}")
    
    def _log_events_batch_json(self, events: List[EventData]):
        """JSON 형식으로 배치 이벤트 로그"""
        try:
            # 기존 데이터 읽기
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 이벤트들 추가
            current_time = datetime.now().isoformat()
            for event_data in events:
                event_dict = event_data.to_dict()
                event_dict['logged_at'] = current_time
                data['events'].append(event_dict)
            
            # 파일 업데이트
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to log JSON event batch: {e}")
    
    def _log_events_batch_csv(self, events: List[EventData]):
        """CSV 형식으로 배치 이벤트 로그"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'event_type', 'window_id', 'confidence', 
                    'prediction', 'duration', 'frame_number', 'metadata'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                for event_data in events:
                    # 메타데이터를 문자열로 변환
                    metadata_str = json.dumps(event_data.metadata) if event_data.metadata else None
                    
                    writer.writerow({
                        'timestamp': datetime.fromtimestamp(event_data.timestamp).isoformat(),
                        'event_type': event_data.event_type.value,
                        'window_id': event_data.window_id,
                        'confidence': event_data.confidence,
                        'prediction': event_data.prediction,
                        'duration': event_data.duration,
                        'frame_number': event_data.frame_number,
                        'metadata': metadata_str
                    })
                    
        except Exception as e:
            logging.error(f"Failed to log CSV event batch: {e}")
    
    def finalize_session(self, final_stats: Optional[Dict[str, Any]] = None):
        """세션 종료 로그"""
        if not self.enable_logging:
            return
        
        try:
            session_end = datetime.now()
            duration = (session_end - self.session_start).total_seconds()
            
            if self.log_format == "json":
                self._finalize_session_json(session_end, duration, final_stats)
            elif self.log_format == "csv":
                self._finalize_session_csv(session_end, duration, final_stats)
                
            logging.info(f"Session finalized. Duration: {duration:.1f}s")
            
        except Exception as e:
            logging.error(f"Failed to finalize session: {e}")
    
    def _finalize_session_json(self, session_end: datetime, duration: float, 
                              final_stats: Optional[Dict[str, Any]]):
        """JSON 형식으로 세션 종료"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 세션 정보 업데이트
            data['session_info'].update({
                'end_time': session_end.isoformat(),
                'duration_seconds': duration,
                'total_events': len(data['events']),
                'final_stats': final_stats or {}
            })
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to finalize JSON session: {e}")
    
    def _finalize_session_csv(self, session_end: datetime, duration: float, 
                             final_stats: Optional[Dict[str, Any]]):
        """CSV 형식으로 세션 종료"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'event_type', 'window_id', 'confidence', 
                    'prediction', 'duration', 'frame_number', 'metadata'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                stats_str = json.dumps(final_stats) if final_stats else None
                
                writer.writerow({
                    'timestamp': session_end.isoformat(),
                    'event_type': 'session_end',
                    'window_id': 0,
                    'confidence': 0.0,
                    'prediction': 'session_info',
                    'duration': duration,
                    'frame_number': None,
                    'metadata': stats_str
                })
                
        except Exception as e:
            logging.error(f"Failed to finalize CSV session: {e}")
    
    def get_log_file_path(self) -> Optional[Path]:
        """로그 파일 경로 반환"""
        return self.log_file if self.enable_logging else None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """현재 세션 통계 반환"""
        if not self.enable_logging:
            return {}
        
        try:
            if self.log_format == "json" and self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        'total_events': len(data['events']),
                        'session_duration': (datetime.now() - self.session_start).total_seconds(),
                        'log_file': str(self.log_file),
                        'format': self.log_format
                    }
            else:
                return {
                    'session_duration': (datetime.now() - self.session_start).total_seconds(),
                    'log_file': str(self.log_file),
                    'format': self.log_format
                }
        except Exception as e:
            logging.error(f"Failed to get session stats: {e}")
            return {}