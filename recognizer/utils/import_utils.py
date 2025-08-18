"""
Import 유틸리티 모듈
중복되는 조건부 import 패턴을 통합 관리
"""

import logging
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)


# 기존 import 헬퍼들은 단순한 import 패턴을 복잡하게 만들고 있음
# 직접 import를 사용하는 것이 더 명확하고 이해하기 쉬움


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    표준화된 로거 설정
    중복되는 로거 설정 코드를 통합
    
    Args:
        name: 로거 이름
        level: 로그 레벨
    
    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger