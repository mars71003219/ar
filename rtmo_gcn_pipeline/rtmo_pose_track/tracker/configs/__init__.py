"""
Configuration files for tracker
"""

from .default_config import DefaultTrackerConfig
from .rtmo_tracker_config import RTMOTrackerConfig

__all__ = [
    'DefaultTrackerConfig',
    'RTMOTrackerConfig'
]