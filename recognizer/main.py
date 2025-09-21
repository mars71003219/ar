#!/usr/bin/env python3
"""
Recognizer 메인 실행기 - 완전 모듈화 버전
모든 모드를 통합 관리하는 일반화된 구조
"""

import argparse
import sys
import logging
from pathlib import Path

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
        # RTMO 포즈 추정기 (PyTorch)
        from pose_estimation.rtmo.rtmo_estimator import RTMOPoseEstimator
        ModuleFactory.register_pose_estimator(
            name='rtmo',
            estimator_class=RTMOPoseEstimator,
            default_config={'score_threshold': 0.3, 'device': 'cuda:0'}
        )
        
        # RTMO ONNX 추정기
        try:
            from pose_estimation.rtmo.rtmo_onnx_estimator import RTMOONNXEstimator
            ModuleFactory.register_pose_estimator(
                name='rtmo_onnx',
                estimator_class=RTMOONNXEstimator,
                default_config={'score_threshold': 0.3, 'device': 'cuda:0'}
            )
        except ImportError as e:
            logger.warning(f"ONNX estimator not available: {e}")
        
        # RTMO TensorRT 추정기
        try:
            from pose_estimation.rtmo.rtmo_tensorrt_estimator import RTMOTensorRTEstimator
            ModuleFactory.register_pose_estimator(
                name='rtmo_tensorrt', 
                estimator_class=RTMOTensorRTEstimator,
                default_config={'score_threshold': 0.3, 'device': 'cuda:0'}
            )
        except ImportError as e:
            logger.warning(f"TensorRT estimator not available: {e}")
        
        # ByteTracker
        from tracking.bytetrack.byte_tracker import ByteTrackerWrapper
        ModuleFactory.register_tracker(
            name='bytetrack',
            tracker_class=ByteTrackerWrapper,
            default_config={'track_thresh': 0.5}
        )
        
        # Motion-based Scorer (이전 Region-based Scorer)
        from scoring.motion_based.fight_scorer import MotionBasedScorer
        ModuleFactory.register_scorer(
            name='region_based',
            scorer_class=MotionBasedScorer,
            default_config={'distance_threshold': 100.0}
        )

        # Movement-based Scorer (same as motion_based but optimized for movement)
        ModuleFactory.register_scorer(
            name='movement_based',
            scorer_class=MotionBasedScorer,
            default_config={'distance_threshold': 100.0}
        )
        
        # Falldown Scorer (쓰러짐 전용 점수 계산기)
        from scoring.motion_based.falldown_scorer import FalldownScorer
        ModuleFactory.register_scorer(
            name='falldown_scorer',
            scorer_class=FalldownScorer,
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
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str,
                       help='Override mode from config (e.g., inference.analysis, annotation.stage1)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--list-modes', action='store_true',
                       help='List all available modes')
    
    # 멀티 프로세스 어노테이션 옵션
    parser.add_argument('--multi-process', action='store_true',
                       help='Run multi-process annotation')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes for multi-process annotation (default: 4)')
    parser.add_argument('--gpus', type=str, default='0,1',
                       help='GPU assignments for multi-process (comma-separated, e.g. 0,1)')
    
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
        
        # 모드 결정 (인자 우선, 그 다음 설정 파일)
        mode = args.mode or config.get('mode', 'inference.analysis')
        
        # 멀티 프로세스 처리 설정 (annotation, inference.analysis 또는 evaluation 모드)
        annotation_config = config.get('annotation', {})
        stage1_config = annotation_config.get('stage1', {})
        stage1_multi_process = stage1_config.get('multi_process', {})
        should_run_multi_process = args.multi_process or stage1_multi_process.get('enabled', False)

        # 멀티프로세스는 stage1만 지원하도록 제한
        if mode.startswith('annotation.') and should_run_multi_process:
            stage1_enabled = stage1_config.get('enabled', True)

            # stage1이 아닌 다른 annotation 모드들은 단일 프로세스로 실행
            if mode in ['annotation.stage2', 'annotation.stage3', 'annotation.visualize']:
                logger.info(f"{mode} - switching to single process mode (only stage1 supports multi-process)")
                should_run_multi_process = False
            elif not stage1_enabled:
                logger.info("Stage1 disabled - switching to single process mode to use existing stage1 data")
                should_run_multi_process = False
            else:
                # stage1이 활성화되고 멀티프로세스 모드인 경우에만 멀티프로세스 실행
                return run_multi_process_annotation(config, args)
        elif mode == 'inference.analysis':
            # inference.analysis 멀티프로세스 설정 확인
            inference_config = config.get('inference', {})
            analysis_config = inference_config.get('analysis', {})
            analysis_multi_process = analysis_config.get('multi_process', {})
            analysis_should_run_multi_process = args.multi_process or analysis_multi_process.get('enabled', False)

            if analysis_should_run_multi_process:
                return run_multi_process_inference_analysis(config, args)
        elif mode == 'evaluation':
            # evaluation 모드의 멀티프로세스 설정
            if args.multi_process:
                config['multi_process'] = True
                config['num_processes'] = args.num_processes
        
        # 모듈 등록
        if not register_modules():
            return False
        
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


def run_multi_process_annotation(config, args):
    """멀티 프로세스 어노테이션 실행"""
    try:
        from utils.multi_process_splitter import run_multi_process_annotation as run_mp
        
        # config에서 stage1 multi-process 설정 가져오기
        annotation_config = config.get('annotation', {})
        stage1_config = annotation_config.get('stage1', {})
        multi_process_config = stage1_config.get('multi_process', {})
        
        # config 우선, command line args는 fallback
        if hasattr(args, 'num_processes') and args.num_processes != 4:
            # command line에서 기본값이 아닌 값이 설정된 경우
            num_processes = args.num_processes
        else:
            num_processes = multi_process_config.get('num_processes', 4)
        
        # GPU 할당 설정 - 라운드 로빈 방식
        if hasattr(args, 'gpus') and args.gpus != '0,1':
            # command line에서 기본값이 아닌 값이 설정된 경우
            available_gpus = [int(x.strip()) for x in args.gpus.split(',')]
        else:
            # config에서 GPU 목록 가져오기
            available_gpus = multi_process_config.get('gpus', [0, 1])
        
        # 라운드 로빈으로 GPU 할당
        gpu_assignments = [available_gpus[i % len(available_gpus)] for i in range(num_processes)]
        
        # 설정에서 입력/출력 경로 가져오기
        input_dir = config.get('annotation', {}).get('input', '/aivanas/raw/surveillance/action/violence/action_recognition/data/RWF-2000')
        output_dir = config.get('annotation', {}).get('output_dir', 'output')
        
        # 절대 경로로 변환
        if not Path(output_dir).is_absolute():
            output_dir = str(Path.cwd() / output_dir)
        
        # 설정 파일 경로를 절대 경로로 변환
        config_path = args.config
        if not Path(config_path).is_absolute():
            config_path = str(Path.cwd() / config_path)
        
        logger.info("=== Multi-Process Annotation Configuration ===")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Config file: {config_path}")
        logger.info(f"Number of processes: {num_processes}")
        logger.info(f"GPU assignments: {gpu_assignments}")
        logger.info(f"Config source: {'Config file' if multi_process_config.get('enabled', False) else 'Command line'}")
        
        # 멀티 프로세스 어노테이션 실행
        success = run_mp(
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
            num_processes=num_processes,
            gpu_assignments=gpu_assignments
        )
        
        if success:
            logger.info("Multi-process annotation completed successfully!")
        else:
            logger.error("Multi-process annotation failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Multi-process annotation error: {e}")
        return False


def run_multi_process_inference_analysis(config, args):
    """멀티 프로세스 inference.analysis 실행"""
    try:
        from utils.multi_process_splitter import run_multi_process_inference_analysis as run_mp
        
        # config에서 inference.analysis multi-process 설정 가져오기
        inference_config = config.get('inference', {})
        analysis_config = inference_config.get('analysis', {})
        multi_process_config = analysis_config.get('multi_process', {})
        
        # config 우선, command line args는 fallback
        if hasattr(args, 'num_processes') and args.num_processes != 4:
            num_processes = args.num_processes
        else:
            num_processes = multi_process_config.get('num_processes', 4)
        
        # GPU 할당 설정
        if hasattr(args, 'gpus') and args.gpus != '0,1':
            available_gpus = [int(x.strip()) for x in args.gpus.split(',')]
        else:
            available_gpus = multi_process_config.get('gpus', [0, 1])
        
        # 라운드 로빈으로 GPU 할당
        gpu_assignments = [available_gpus[i % len(available_gpus)] for i in range(num_processes)]
        
        # 설정에서 입력/출력 경로 가져오기
        input_dir = config.get('inference', {}).get('analysis', {}).get('input')
        if not input_dir:
            # inference.realtime의 input을 fallback으로 사용
            input_dir = config.get('inference', {}).get('realtime', {}).get('input', '/aivanas/raw/surveillance/action/violence/action_recognition/data/UBI_demo')
        
        output_dir = config.get('inference', {}).get('analysis', {}).get('output_dir', 'output')
        
        # 절대 경로로 변환
        if not Path(output_dir).is_absolute():
            output_dir = str(Path.cwd() / output_dir)
        
        # 설정 파일 경로를 절대 경로로 변환
        config_path = args.config
        if not Path(config_path).is_absolute():
            config_path = str(Path.cwd() / config_path)
        
        logger.info("=== Multi-Process Inference Analysis Configuration ===")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Config file: {config_path}")
        logger.info(f"Number of processes: {num_processes}")
        logger.info(f"GPU assignments: {gpu_assignments}")
        logger.info(f"Config source: {'Config file' if multi_process_config.get('enabled', False) else 'Command line'}")
        
        # 멀티 프로세스 inference.analysis 실행
        success = run_mp(
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
            num_processes=num_processes,
            gpu_assignments=gpu_assignments
        )
        
        if success:
            logger.info("Multi-process inference analysis completed successfully!")
        else:
            logger.error("Multi-process inference analysis failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Multi-process inference analysis error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)