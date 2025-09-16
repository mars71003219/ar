"""
이벤트 타입 정의
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class EventType(Enum):
    """이벤트 타입"""
    VIOLENCE_START = "violence_start"
    VIOLENCE_END = "violence_end"
    VIOLENCE_ONGOING = "violence_ongoing"
    NORMAL = "normal"

@dataclass
class EventData:
    """이벤트 데이터"""
    event_type: EventType
    timestamp: float
    window_id: int
    confidence: float
    prediction: str
    frame_number: Optional[int] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'window_id': self.window_id,
            'confidence': self.confidence,
            'prediction': self.prediction,
            'frame_number': self.frame_number,
            'duration': self.duration,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventData':
        """딕셔너리에서 생성"""
        return cls(
            event_type=EventType(data['event_type']),
            timestamp=data['timestamp'],
            window_id=data['window_id'],
            confidence=data['confidence'],
            prediction=data['prediction'],
            frame_number=data.get('frame_number'),
            duration=data.get('duration'),
            metadata=data.get('metadata', {})
        )