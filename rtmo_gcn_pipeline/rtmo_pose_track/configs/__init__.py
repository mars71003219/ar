#!/usr/bin/env python3
"""
Configuration package for pose extraction pipeline
"""

from .config_loader import ConfigLoader, load_config
from .default_config import DefaultConfig

__all__ = [
    'ConfigLoader',
    'load_config', 
    'DefaultConfig',
]