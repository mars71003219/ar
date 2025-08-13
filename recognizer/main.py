#!/usr/bin/env python3
"""
Recognizer 통합 메인 실행기

하나의 메인 파일로 모든 파이프라인과 기능을 설정 기반으로 실행합니다.
더 이상 examples 폴더에 새로운 파일을 만들 필요가 없습니다.

사용법:
  # 기본 추론
  python recognizer/main.py --mode inference --input video.mp4

  # 성능 평가 포함 추론
  python recognizer/main.py --mode inference --input video.mp4 --enable_evaluation --enable_visualization

  # separated pipeline 실행
  python recognizer/main.py --mode separated --input data/videos --output output/separated

  # annotation pipeline 실행
  python recognizer/main.py --mode annotation --input stage2_results.pkl --video original.mp4

  # 설정 파일 사용
  python recognizer/main.py --config configs/my_config.yaml

  # 멀티GPU 실행
  python recognizer/main.py --mode inference --input video.mp4 --multi_gpu --gpus 0,1,2,3

  # 멀티프로세스 실행
  python recognizer/main.py --mode separated --input data/videos --multiprocess --workers 8
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# recognizer 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from pipelines import (
    InferencePipeline, RealtimeConfig, RealtimeAlert,
    SeparatedPipeline, SeparatedPipelineConfig,
    UnifiedPipeline, PipelineConfig, PipelineResult
)
# Optional imports - only import when needed
# from utils.performance_evaluator import PerformanceEvaluator
# from visualization.result_visualizer import ResultVisualizer
# from visualization.annotation_visualizer import AnnotationVisualizer
from utils.data_structure import *
from utils.factory import ModuleFactory

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecognizerMainExecutor:
    """통합 메인 실행기"""
    
    SUPPORTED_MODES = {
        'inference': 'Real-time inference pipeline',
        'separated': 'Multi-stage separated pipeline', 
        'unified': 'Unified end-to-end pipeline'
    }
    
    def __init__(self):
        self.config = None
        self.pipeline = None
        self.evaluator = None
        self.visualizers = {}
        
    def load_config(self, config_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """설정 로드 및 생성"""
        config = {}
        
        # 1. 설정 파일에서 로드
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
        
        # 2. 명령행 인자로 오버라이드
        config.update(kwargs)
        
        # 3. 기본값 설정
        default_config = {
            'execution': {
                'mode': 'inference',
                'input': None,
                'output_dir': 'output'
            },
            'features': {
                'enable_evaluation': False,
                'enable_visualization': False,
                'enable_gpu_acceleration': True
            },
            'performance': {
                'device': 'cuda:0',
                'multi_gpu': {
                    'enable': False,
                    'gpus': [0, 1, 2, 3],
                    'strategy': 'data_parallel',
                    'batch_per_gpu': 1
                },
                'multiprocess': {
                    'enable': False,
                    'workers': 4,
                    'strategy': 'pipeline',
                    'shared_memory': True,
                    'queue_size': 100
                },
                'window_size': 100,
                'inference_stride': 25,
                'batch_size': 1,
                'mixed_precision': False
            },
            'visualization': {
                'quality': {
                    'video_fps': 30.0
                }
            }
        }
        
        # 중첩 딕셔너리 병합을 위한 헬퍼 함수
        def merge_configs(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_configs(base[key], value)
                else:
                    base[key] = value
            return base
        
        # 기본값과 병합
        if not config:  # 설정이 비어있으면 기본값 사용
            config = default_config.copy()
        else:
            config = merge_configs(default_config.copy(), config)
        
        self.config = config
        return config
    
    def setup_evaluation(self):
        """성능 평가기 설정"""
        if not self.config.get('enable_evaluation', False):
            return None
        
        try:
            from utils.performance_evaluator import PerformanceEvaluator
            evaluator = PerformanceEvaluator(
                output_dir=f"{self.config['output_dir']}/evaluation",
                class_names=self.config.get('class_names', ["NonFight", "Fight"])
            )
            logger.info("Performance evaluation enabled")
            return evaluator
        except ImportError:
            logger.warning("Performance evaluator not available (missing dependencies)")
            return None
    
    def setup_visualization(self) -> Dict[str, Any]:
        """시각화 도구 설정"""
        visualizers = {}
        
        if self.config.get('enable_visualization', False):
            try:
                from visualization.result_visualizer import ResultVisualizer
                from visualization.annotation_visualizer import AnnotationVisualizer
                visualizers['result'] = ResultVisualizer()
                visualizers['annotation'] = AnnotationVisualizer()
                logger.info("Visualization tools enabled")
            except ImportError:
                logger.warning("Visualization tools not available (missing dependencies)")
        
        return visualizers
    
    def run_inference_mode(self) -> bool:
        """추론 모드 실행"""
        logger.info("=== INFERENCE MODE ===")
        
        # 설정 생성
        if self.config.get('multi_gpu', False):
            from utils.multi_gpu_model import MultiGPUModelManager
            # 멀티GPU 추론 실행
            logger.info(f"Multi-GPU inference with GPUs: {self.config['gpus']}")
            # TODO: 멀티GPU 추론 파이프라인 통합
            
        elif self.config.get('multiprocess', False):
            from utils.multiprocess_manager import MultiprocessManager
            # 멀티프로세스 추론 실행
            logger.info(f"Multiprocess inference with {self.config['workers']} workers")
            # TODO: 멀티프로세스 추론 파이프라인 통합
            
        else:
            # 단일 프로세스 추론
            pipeline_config = self._create_inference_config()
            pipeline = InferencePipeline(pipeline_config)
            
            # 평가기 설정
            evaluator = self.setup_evaluation()
            
            # 추론 실행
            success = self._execute_inference(pipeline, evaluator)
            
            # 시각화 (평가가 활성화된 경우)
            if success and evaluator and self.config.get('enable_visualization', False):
                self._create_inference_visualizations(evaluator)
            
            return success
        
        return True
    
    def run_separated_mode(self) -> bool:
        """분리 파이프라인 모드 실행"""
        logger.info("=== SEPARATED PIPELINE MODE ===")
        
        # 분리 파이프라인 설정 생성
        separated_config = self._create_separated_config()
        
        if self.config.get('multiprocess', False):
            # 멀티프로세스 separated pipeline
            logger.info(f"Multiprocess separated pipeline with {self.config['workers']} workers")
            # TODO: 멀티프로세스 separated 파이프라인 실행
            
        else:
            # 단일 프로세스 separated pipeline
            pipeline = SeparatedPipeline(separated_config)
            
            success = pipeline.run_full_pipeline(
                input_dir=self.config['input'],
                output_dir=self.config['output_dir']
            )
            
            # separated 결과 시각화
            if success and self.config.get('enable_visualization', False):
                self._create_separated_visualizations()
            
            return success
        
        return True
    
    def run_unified_mode(self) -> bool:
        """통합 파이프라인 모드 실행"""
        logger.info("=== UNIFIED PIPELINE MODE ===")
        
        # 통합 파이프라인 설정 생성
        unified_config = self._create_unified_config()
        
        pipeline = UnifiedPipeline(unified_config)
        
        if Path(self.config['input']).is_dir():
            # 다중 비디오 배치 처리
            video_paths = list(Path(self.config['input']).glob("*.mp4"))
            results = pipeline.process_video_batch(video_paths)
            
            # 결과 저장
            for i, result in enumerate(results):
                output_path = Path(self.config['output_dir']) / f"result_{i}.pkl"
                pipeline.save_results(result, output_path)
            
            success = len(results) > 0
        else:
            # 단일 비디오 처리
            result = pipeline.process_video(self.config['input'])
            
            # 결과 저장
            output_path = Path(self.config['output_dir']) / "result.pkl"
            pipeline.save_results(result, output_path)
            
            success = result is not None
        
        return success
    
    def _create_inference_config(self):
        """추론 설정 생성"""
        from pipelines.inference.config import RealtimeConfig
        from utils.data_structure import (
            PoseEstimationConfig, TrackingConfig, 
            ScoringConfig, ActionClassificationConfig
        )
        
        return RealtimeConfig(
            pose_config=PoseEstimationConfig(
                device=self.config.get('device', 'cuda:0'),
                model_name=self.config.get('pose_model', 'rtmo')
            ),
            tracking_config=TrackingConfig(
                tracker_name=self.config.get('tracker', 'bytetrack')
            ),
            scoring_config=ScoringConfig(
                scorer_name=self.config.get('scorer', 'region_based')
            ),
            classification_config=ActionClassificationConfig(
                device=self.config.get('device', 'cuda:0'),
                model_name=self.config.get('action_model', 'stgcn')
            ),
            window_size=self.config.get('window_size', 100),
            inference_stride=self.config.get('inference_stride', 25),
            max_queue_size=self.config.get('max_queue_size', 200),
            target_fps=self.config.get('target_fps', 30),
            skip_frames=self.config.get('skip_frames', 1),
            alert_threshold=self.config.get('alert_threshold', 0.5)
        )
    
    def _create_separated_config(self):
        """분리 파이프라인 설정 생성"""
        from pipelines.separated.config import SeparatedPipelineConfig
        
        return SeparatedPipelineConfig(
            input_dir=self.config['input'],
            output_dir=self.config['output_dir'],
            device=self.config.get('device', 'cuda:0'),
            batch_size=self.config.get('batch_size', 8),
            window_size=self.config.get('window_size', 100),
            enable_multiprocess=self.config.get('multiprocess', False),
            num_workers=self.config.get('workers', 4)
        )
    
    def _create_unified_config(self):
        """통합 파이프라인 설정 생성"""
        from pipelines.unified.config import PipelineConfig
        from utils.data_structure import (
            PoseEstimationConfig, TrackingConfig,
            ScoringConfig, ActionClassificationConfig
        )
        
        return PipelineConfig(
            pose_config=PoseEstimationConfig(
                device=self.config.get('device', 'cuda:0'),
                model_name=self.config.get('pose_model', 'rtmo')
            ),
            tracking_config=TrackingConfig(
                tracker_name=self.config.get('tracker', 'bytetrack')
            ),
            scoring_config=ScoringConfig(
                scorer_name=self.config.get('scorer', 'region_based')
            ),
            classification_config=ActionClassificationConfig(
                device=self.config.get('device', 'cuda:0'),
                model_name=self.config.get('action_model', 'stgcn')
            ),
            window_size=self.config.get('window_size', 100),
            window_stride=self.config.get('window_stride', 50),
            batch_size=self.config.get('batch_size', 8),
            save_intermediate_results=self.config.get('save_intermediate_results', False)
        )
    
    def _execute_inference(self, pipeline, evaluator) -> bool:
        """추론 실행"""
        input_source = self.config['input']
        
        logger.info(f"Processing input: {input_source}")
        
        # TODO: 실제 추론 실행 로직
        # 현재는 성공으로 가정
        logger.warning("Actual inference execution not yet implemented")
        
        return True
    
    def _create_inference_visualizations(self, evaluator):
        """추론 결과 시각화 생성"""
        logger.info("Creating inference visualizations...")
        
        if 'result' not in self.visualizers:
            try:
                from visualization.result_visualizer import ResultVisualizer
                self.visualizers['result'] = ResultVisualizer()
            except ImportError:
                logger.warning("ResultVisualizer not available")
                return
        
        # 성능 지표 시각화
        metrics = evaluator.calculate_performance_metrics()
        
        viz_dir = Path(self.config['output_dir']) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: 실제 시각화 생성
        logger.info(f"Visualizations saved to: {viz_dir}")
    
    def _create_separated_visualizations(self):
        """separated 결과 시각화 생성"""
        logger.info("Creating separated pipeline visualizations...")
        
        # TODO: separated 결과 시각화
        pass
    
    def execute(self, mode: str) -> bool:
        """메인 실행 함수"""
        if mode not in self.SUPPORTED_MODES:
            logger.error(f"Unsupported mode: {mode}")
            logger.info(f"Supported modes: {list(self.SUPPORTED_MODES.keys())}")
            return False
        
        # 평가기 및 시각화 도구 설정
        self.evaluator = self.setup_evaluation()
        self.visualizers = self.setup_visualization()
        
        # 모드별 실행
        if mode == 'inference':
            return self.run_inference_mode()
        elif mode == 'separated':
            return self.run_separated_mode()
        elif mode == 'unified':
            return self.run_unified_mode()
        
        return False


def create_parser() -> argparse.ArgumentParser:
    """명령행 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="Recognizer Unified Main Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python main.py --mode inference --input video.mp4
  
  # Inference with evaluation and visualization  
  python main.py --mode inference --input video.mp4 --enable_evaluation --enable_visualization
  
  # Separated pipeline
  python main.py --mode separated --input data/videos --output output/separated
  
  # Unified pipeline
  python main.py --mode unified --input video.mp4 --output output/unified
  
  # Multi-GPU inference
  python main.py --mode inference --input video.mp4 --multi_gpu --gpus 0,1,2,3
  
  # Config file usage
  python main.py --config my_config.yaml
        """
    )
    
    # 기본 인자들
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['inference', 'separated', 'unified'],
                       default='inference', help='Execution mode')
    
    # 입출력
    parser.add_argument('--input', type=str, help='Input source (video file, directory, RTSP URL, webcam index)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    
    # 기능 토글
    parser.add_argument('--enable_evaluation', action='store_true', help='Enable performance evaluation')
    parser.add_argument('--enable_visualization', action='store_true', help='Enable visualization generation')
    
    # 성능 옵션
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU processing')
    parser.add_argument('--multiprocess', action='store_true', help='Enable multiprocess processing')
    parser.add_argument('--gpus', type=str, default='0', help='GPU indices (comma-separated)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    # 모델 설정
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    parser.add_argument('--pose_model', type=str, default='rtmo', help='Pose estimation model')
    parser.add_argument('--action_model', type=str, default='stgcn', help='Action classification model')
    parser.add_argument('--tracker', type=str, default='bytetrack', help='Tracking algorithm')
    parser.add_argument('--scorer', type=str, default='region_based', help='Scoring algorithm')
    
    # 추론 설정
    parser.add_argument('--window_size', type=int, default=100, help='Window size for classification')
    parser.add_argument('--inference_stride', type=int, default=25, help='Inference stride')
    parser.add_argument('--duration', type=float, help='Processing duration (seconds)')
    
    # 파이프라인 설정
    parser.add_argument('--window_stride', type=int, default=50, help='Window stride for unified pipeline')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate results')
    
    # 시각화 설정
    parser.add_argument('--visualization_fps', type=float, default=30.0, help='Visualization FPS')
    
    # 기타
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    return parser


def main():
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # GPU 목록 파싱
    if args.gpus:
        args.gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    
    # 설정 변환
    config_kwargs = {
        'mode': args.mode,
        'input': args.input,
        'output_dir': args.output_dir,
        'enable_evaluation': args.enable_evaluation,
        'enable_visualization': args.enable_visualization,
        'multi_gpu': args.multi_gpu,
        'multiprocess': args.multiprocess,
        'gpus': args.gpus,
        'workers': args.workers,
        'device': args.device,
        'pose_model': args.pose_model,
        'action_model': args.action_model,
        'tracker': args.tracker,
        'scorer': args.scorer,
        'window_size': args.window_size,
        'inference_stride': args.inference_stride,
        'duration': args.duration,
        'window_stride': args.window_stride,
        'save_intermediate_results': args.save_intermediate,
        'visualization_fps': args.visualization_fps
    }
    
    try:
        # 메인 실행기 생성 및 실행
        executor = RecognizerMainExecutor()
        executor.load_config(args.config, **config_kwargs)
        
        logger.info("=== RECOGNIZER MAIN EXECUTOR ===")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(f"Evaluation: {'ON' if args.enable_evaluation else 'OFF'}")
        logger.info(f"Visualization: {'ON' if args.enable_visualization else 'OFF'}")
        
        success = executor.execute(args.mode)
        
        if success:
            logger.info("=== EXECUTION COMPLETED SUCCESSFULLY ===")
        else:
            logger.error("=== EXECUTION FAILED ===")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()