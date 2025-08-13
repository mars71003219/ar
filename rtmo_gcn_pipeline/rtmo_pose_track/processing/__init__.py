"""
Processing pipeline modules
"""

from .base_processor import BaseProcessor
from .rtsp_stream_processor import RTSPStreamProcessor

__all__ = [
    'BaseProcessor',
    'RTSPStreamProcessor'
]