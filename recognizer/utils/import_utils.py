"""
Import 유틸리티 모듈
중복되는 조건부 import 패턴을 통합 관리
"""

import logging
import traceback
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)


# 기존 import 헬퍼들은 단순한 import 패턴을 복잡하게 만들고 있음
# 직접 import를 사용하는 것이 더 명확하고 이해하기 쉬움


def log_error_with_traceback(message: str, exception: Exception = None, logger_name: str = None) -> None:
    """
    에러 메시지와 함께 traceback을 로깅하는 공통 함수

    Args:
        message: 에러 메시지
        exception: 예외 객체 (선택사항)
        logger_name: 로거 이름 (선택사항, 기본값은 호출 모듈)
    """
    if logger_name:
        error_logger = logging.getLogger(logger_name)
    else:
        error_logger = logging.getLogger(__name__)

    if exception:
        error_logger.error(f"{message}: {exception}")
    else:
        error_logger.error(message)

    error_logger.error(f"Traceback: {traceback.format_exc()}")


def safe_operation(operation_func, error_message: str = "Operation failed",
                  default_return: Any = None, logger_name: str = None):
    """
    안전한 작업 실행을 위한 래퍼 함수

    Args:
        operation_func: 실행할 함수
        error_message: 에러 시 표시할 메시지
        default_return: 에러 시 반환할 기본값
        logger_name: 로거 이름

    Returns:
        함수 실행 결과 또는 기본값
    """
    try:
        return operation_func()
    except Exception as e:
        log_error_with_traceback(error_message, e, logger_name)
        return default_return


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