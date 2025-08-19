#!/usr/bin/env python3
"""
Recognizer 메인 실행기 - 완전 모듈화 버전
모든 모드를 통합 관리하는 일반화된 구조
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# recognizer 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(log_level: str):
    """로깅 설정"""
    level = getattr(logging, log_level.upper())
    logging.getLogger().setLevel(level)
    
    # 모든 하위 모듈의 로깅 레벨도 설정
    for name in logging.Logger.manager.loggerDict:
        if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
            logging.Logger.manager.loggerDict[name].setLevel(level)


def register_modules():
    """모듈 등록"""
    from utils.factory import ModuleFactory
    
    try:
        # RTMO 포즈 추정기
        from pose_estimation.rtmo.rtmo_estimator import RTMOPoseEstimator
        ModuleFactory.register_pose_estimator(
            name='rtmo',
            estimator_class=RTMOPoseEstimator,
            default_config={'score_threshold': 0.3, 'device': 'cuda:0'}
        )
        
        # ByteTracker
        from tracking.bytetrack.byte_tracker import ByteTrackerWrapper
        ModuleFactory.register_tracker(
            name='bytetrack',
            tracker_class=ByteTrackerWrapper,
            default_config={'track_thresh': 0.5}
        )
        
        # Region-based Scorer
        from scoring.region_based.region_scorer import RegionBasedScorer
        ModuleFactory.register_scorer(
            name='region_based',
            scorer_class=RegionBasedScorer,
            default_config={'distance_threshold': 100.0}
        )
        
        # STGCN Action Classifier
        from action_classification.stgcn.stgcn_classifier import STGCNActionClassifier
        ModuleFactory.register_classifier(
            name='stgcn',
            classifier_class=STGCNActionClassifier,
            default_config={'num_classes': 2, 'device': 'cuda:0'}
        )
        
        # Window Processor
        from utils.window_processor import SlidingWindowProcessor
        ModuleFactory.register_window_processor(
            name='sliding_window',
            processor_class=SlidingWindowProcessor,
            default_config={'window_size': 100, 'window_stride': 50}
        )
        
        logger.info("All modules registered successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register modules: {e}")
        return False


def load_config(config_file: str = None):
    """설정 로드"""
    try:
        from utils.config_loader import load_config
        
        if not config_file:
            config_file = "config.yaml"
        
        config = load_config(
            config_file=config_file,
            mode=None,
            args_dict={}
        )
        
        logger.info(f"Configuration loaded from: {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def main():
    """메인 실행 함수 - 완전 일반화"""
    parser = argparse.ArgumentParser(description="Recognizer - Unified Mode Manager")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str,
                       help='Override mode from config (e.g., inference.analysis, annotation.stage1)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--list-modes', action='store_true',
                       help='List all available modes')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    
    try:
        # 설정 로드
        config = load_config(args.config)
        if not config:
            return False
        
        # 모드 매니저 초기화
        from core import ModeManager
        mode_manager = ModeManager(config)
        
        # 모드 목록 표시
        if args.list_modes:
            modes = mode_manager.list_modes()
            logger.info("Available modes:")
            for mode_name, description in modes.items():
                logger.info(f"  {mode_name}: {description}")
            return True
        
        # 모듈 등록
        if not register_modules():
            return False
        
        # 모드 결정 (인자 우선, 그 다음 설정 파일)
        mode = args.mode or config.get('mode', 'inference.analysis')
        
        # 파이프라인 모드 자동 감지 (시각화 모드는 제외)
        if mode.startswith('annotation.') and mode not in ['annotation.pipeline', 'annotation.visualize']:
            from core.annotation_pipeline_mode import should_use_pipeline_mode
            if should_use_pipeline_mode(config):
                mode = 'annotation.pipeline'
                logger.info("Pipeline mode detected - switching to annotation.pipeline")
        
        logger.info(f"=== EXECUTING MODE: {mode} ===")
        
        # 모드 실행
        success = mode_manager.execute(mode)
        
        if success:
            logger.info(f"Mode {mode} completed successfully")
        else:
            logger.error(f"Mode {mode} failed")
        
        return success
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)