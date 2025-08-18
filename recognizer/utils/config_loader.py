"""
통합 설정 로더

YAML 파일과 argparse 인수를 통합하여 일관된 설정을 제공합니다.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

# 설정 파일 기본 경로
CONFIG_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class UnifiedConfig:
    """통합 설정 클래스"""
    
    # 실행 설정
    mode: str = "inference.analysis"
    processing_mode: str = "realtime"  # 처리 모드: "realtime" 또는 "analysis"
    input_source: str = "test_video.mp4"
    input_dir: str = ""  # 폴더 단위 처리용 입력 디렉토리
    output_dir: str = "output"
    
    # 비디오 처리 설정
    video_extensions: list = None
    processing_duration: float = 10.0
    
    # 동적 버퍼 설정 (자동 계산됨)
    buffer_size: int = 150  # window_size + stride로 자동 계산
    classification_delay: int = 100  # window_size와 동일하게 자동 계산
    show_keypoints: bool = True
    show_tracking_ids: bool = True
    show_composite_score: bool = False
    
    # 실시간 모드 설정
    enable_realtime_display: bool = False
    display_width: int = 1280
    display_height: int = 720
    save_realtime_video: bool = False
    realtime_output_path: str = "output/realtime_output.mp4"
    frame_buffer_size: int = 5
    display_fps_limit: int = 30
    
    # 모델 설정
    pose_model: str = "rtmo"
    pose_config: str = ""
    pose_checkpoint: str = ""
    pose_device: str = "cuda:0"
    pose_batch_size: int = 8
    pose_input_size: tuple = (640, 640)
    pose_score_threshold: float = 0.3
    
    tracker_name: str = "bytetrack"
    track_thresh: float = 0.4
    track_buffer: int = 50
    match_thresh: float = 0.4
    
    scorer_name: str = "region_based"
    scoring_quality_threshold: float = 0.3
    min_track_length: int = 10
    
    classifier_name: str = "stgcn"
    classifier_config: str = ""
    classifier_checkpoint: str = ""
    classifier_device: str = "cuda:0"
    classifier_window_size: int = 100
    classifier_confidence_threshold: float = 0.4
    class_names: list = None
    
    # 파이프라인 설정
    window_size: int = 100
    window_stride: int = 50
    inference_stride: int = 50
    batch_size: int = 8
    
    # 성능 설정
    device: str = "cuda:0"
    enable_gpu: bool = True
    multi_gpu_enable: bool = False
    multi_gpu_devices: list = None
    multiprocess_enable: bool = False
    multiprocess_workers: int = 4
    mixed_precision: bool = False
    
    # 실시간 처리 설정 (inference 모드)
    target_fps: float = 30.0
    max_queue_size: int = 200
    skip_frames: int = 1
    resize_input: Optional[tuple] = None
    min_confidence: float = 0.5
    alert_threshold: float = 0.7
    
    # 분리형 파이프라인 설정 (separated 모드)
    stages_to_run: list = None
    stage1_output_dir: str = "output/separated/stage1_poses"
    stage2_output_dir: str = "output/separated/stage2_tracking"
    stage3_output_dir: str = "output/separated/stage3_scoring"
    stage4_output_dir: str = "output/separated/stage4_unified"
    enable_resume: bool = True
    save_intermediate_results: bool = True
    save_visualizations: bool = True
    
    # 품질 관리
    enable_quality_filter: bool = True
    min_keypoint_score: float = 0.3
    max_track_gap: int = 10
    
    # 데이터셋 분할 (annotation 모드)
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 평가 설정
    enable_evaluation: bool = False
    enable_visualization: bool = False
    
    # 로깅 설정
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    save_logs: bool = True
    
    # 오류 처리
    continue_on_error: bool = True
    max_consecutive_errors: int = 10
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.class_names is None:
            self.class_names = ["NonFight", "Fight"]
        if self.stages_to_run is None:
            self.stages_to_run = ["stage1", "stage2", "stage3", "stage4"]
        if self.multi_gpu_devices is None:
            self.multi_gpu_devices = [0, 1]
        if self.video_extensions is None:
            self.video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class ConfigLoader:
    """설정 로더"""
    
    def __init__(self, config_dir: Union[str, Path] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    
    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        # 절대 경로인지 확인
        if Path(config_file).is_absolute():
            config_path = Path(config_file)
        else:
            # 현재 디렉토리에서 먼저 찾기
            current_path = Path(config_file)
            if current_path.exists():
                config_path = current_path
            else:
                # configs 디렉토리에서 찾기
                config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logging.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # base_config 상속 처리
            if 'base_config' in config:
                base_config_file = config.pop('base_config')
                base_config = self.load_yaml(base_config_file)
                config = self._merge_configs(base_config, config)
            
            logging.info(f"Loaded config from: {config_path}")
            return config
            
        except Exception as e:
            logging.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def create_unified_config(
        self, 
        config_file: str = "base_config.yaml",
        args_dict: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None
    ) -> UnifiedConfig:
        """통합 설정 생성"""
        
        # YAML 설정 로드
        yaml_config = self.load_yaml(config_file)
        
        # 모드별 설정 오버라이드
        if mode:
            mode_config_file = f"{mode}_config.yaml"
            mode_config = self.load_yaml(mode_config_file)
            yaml_config = self._merge_configs(yaml_config, mode_config)
        
        # YAML에서 UnifiedConfig 매핑
        unified_dict = self._map_yaml_to_unified(yaml_config)
        
        # argparse 인수로 오버라이드
        if args_dict:
            unified_dict = self._merge_configs(unified_dict, args_dict)
        
        # UnifiedConfig 생성
        try:
            config = UnifiedConfig(**unified_dict)
            logging.info(f"Created unified config for mode: {config.mode}")
            return config
        except Exception as e:
            logging.error(f"Failed to create unified config: {e}")
            # 기본 설정으로 fallback
            return UnifiedConfig()
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """설정 병합 (재귀적)"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _map_yaml_to_unified(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """YAML 구조를 UnifiedConfig에 매핑"""
        mapping = {}
        
        # 모드 설정
        mapping['mode'] = yaml_config.get('mode', 'inference.analysis')
        
        # inference.analysis 섹션에서 설정 읽기
        inference_config = yaml_config.get('inference', {})
        analysis_config = inference_config.get('analysis', {})
        
        # inference 설정을 최상위로 올려서 직접 접근 가능하게 함
        if 'inference' not in mapping:
            mapping['inference'] = inference_config
        
        mapping['input_source'] = analysis_config.get('input', 'test_video.mp4')
        mapping['input_dir'] = analysis_config.get('input_dir', '')
        mapping['output_dir'] = analysis_config.get('output_dir', 'output')
        
        # 비디오 처리 설정
        files_config = yaml_config.get('files', {})
        mapping['video_extensions'] = files_config.get('video_extensions', ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'])
        mapping['processing_duration'] = analysis_config.get('processing_duration', 10.0)
        
        # 처리 모드 설정 (processing_mode 추가)
        mapping['processing_mode'] = 'analysis'
        
        # 분석 모드 설정
        analysis_mode = yaml_config.get('analysis_mode', {})
        
        # 실시간 모드 설정 (최상위 레벨에서 읽기)
        realtime_mode = yaml_config.get('realtime_mode', {})
        mapping['enable_realtime_display'] = realtime_mode.get('enable_display', False)
        mapping['display_width'] = realtime_mode.get('display_width', 1280)
        mapping['display_height'] = realtime_mode.get('display_height', 720)
        mapping['save_realtime_video'] = realtime_mode.get('save_realtime_video', False)
        mapping['realtime_output_path'] = realtime_mode.get('realtime_output_path', 'output/realtime_output.mp4')
        mapping['frame_buffer_size'] = realtime_mode.get('frame_buffer_size', 5)
        mapping['display_fps_limit'] = realtime_mode.get('display_fps_limit', 30)
        
        # realtime_mode의 동적 버퍼 설정도 추가 매핑
        window_size = realtime_mode.get('window_size', mapping.get('window_size', 100))
        stride = realtime_mode.get('stride', mapping.get('window_stride', 50))
        
        mapping['window_size'] = window_size
        mapping['window_stride'] = stride
        
        # 자동 계산된 값들
        mapping['buffer_size'] = window_size + stride  # 동적 버퍼 크기 자동 계산
        mapping['classification_delay'] = window_size  # 첫 윈도우 완성 후 분류 시작
        
        mapping['show_keypoints'] = realtime_mode.get('show_keypoints', True)
        mapping['show_tracking_ids'] = realtime_mode.get('show_tracking_ids', True)
        mapping['show_composite_score'] = realtime_mode.get('show_composite_score', False)
        
        # 모델 설정
        models = yaml_config.get('models', {})
        
        # 포즈 모델
        pose = models.get('pose_estimation', {})
        mapping['pose_model'] = pose.get('model_name', 'rtmo')
        mapping['pose_config'] = pose.get('config_file', '')
        mapping['pose_checkpoint'] = pose.get('checkpoint_path', '')
        mapping['pose_device'] = pose.get('device', 'cuda:0')
        mapping['pose_batch_size'] = pose.get('batch_size', 8)
        mapping['pose_input_size'] = tuple(pose.get('input_size', [640, 640]))
        mapping['pose_score_threshold'] = pose.get('score_threshold', 0.3)
        
        # 트래킹
        tracking = models.get('tracking', {})
        mapping['tracker_name'] = tracking.get('tracker_name', 'bytetrack')
        mapping['track_thresh'] = tracking.get('track_thresh', 0.4)
        mapping['track_buffer'] = tracking.get('track_buffer', 50)
        mapping['match_thresh'] = tracking.get('match_thresh', 0.4)
        
        # 스코어링
        scoring = models.get('scoring', {})
        mapping['scorer_name'] = scoring.get('scorer_name', 'region_based')
        mapping['scoring_quality_threshold'] = scoring.get('quality_threshold', 0.3)
        mapping['min_track_length'] = scoring.get('min_track_length', 10)
        
        # 분류
        classification = models.get('action_classification', {})
        mapping['classifier_name'] = classification.get('model_name', 'stgcn')
        mapping['classifier_config'] = classification.get('config_file', '')
        mapping['classifier_checkpoint'] = classification.get('checkpoint_path', '')
        mapping['classifier_device'] = classification.get('device', 'cuda:0')
        mapping['classifier_window_size'] = classification.get('window_size', 100)
        mapping['classifier_confidence_threshold'] = classification.get('confidence_threshold', 0.4)
        mapping['class_names'] = classification.get('class_names', ['NonFight', 'Fight'])
        
        # 성능 설정
        performance = yaml_config.get('performance', {})
        mapping['window_size'] = performance.get('window_size', 100)
        mapping['window_stride'] = performance.get('window_stride', 50)
        mapping['inference_stride'] = performance.get('inference_stride', 50)
        mapping['batch_size'] = performance.get('batch_size', 8)
        mapping['device'] = performance.get('device', 'cuda:0')
        mapping['mixed_precision'] = performance.get('mixed_precision', False)
        
        # 멀티GPU
        multi_gpu = performance.get('multi_gpu', {})
        mapping['multi_gpu_enable'] = multi_gpu.get('enable', False)
        mapping['multi_gpu_devices'] = multi_gpu.get('gpus', [0, 1])
        
        # 멀티프로세스
        multiprocess = performance.get('multiprocess', {})
        mapping['multiprocess_enable'] = multiprocess.get('enable', False)
        mapping['multiprocess_workers'] = multiprocess.get('workers', 4)
        
        # 실시간 설정 (inference 모드)
        realtime = yaml_config.get('realtime', {})
        mapping['target_fps'] = realtime.get('target_fps', 30.0)
        mapping['max_queue_size'] = realtime.get('max_queue_size', 200)
        mapping['skip_frames'] = realtime.get('skip_frames', 1)
        mapping['min_confidence'] = realtime.get('min_confidence', 0.5)
        mapping['alert_threshold'] = realtime.get('alert_threshold', 0.7)
        
        # 기능 설정
        features = yaml_config.get('features', {})
        mapping['enable_evaluation'] = features.get('enable_evaluation', False)
        
        # 시각화 설정: analysis_mode 우선, 그 다음 features 설정
        analysis_visualization = analysis_mode.get('enable_visualization', False)
        features_visualization = features.get('enable_visualization', False)
        mapping['enable_visualization'] = analysis_visualization or features_visualization
        
        mapping['enable_gpu'] = features.get('enable_gpu_acceleration', True)
        
        # 로깅 설정
        logging_config = yaml_config.get('logging', {})
        mapping['log_level'] = logging_config.get('level', 'INFO')
        mapping['log_format'] = logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        mapping['save_logs'] = logging_config.get('save_to_file', True)
        
        # 오류 처리
        error_handling = yaml_config.get('error_handling', {})
        mapping['continue_on_error'] = error_handling.get('continue_on_error', True)
        mapping['max_consecutive_errors'] = error_handling.get('max_consecutive_errors', 10)
        
        return mapping


def load_config(
    config_file: str = "base_config.yaml",
    mode: Optional[str] = None,
    args_dict: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """편의 함수: 통합 설정 로드"""
    loader = ConfigLoader(config_dir)
    
    # 통합 설정 파일인 경우 YAML로 로드하되 매핑 적용
    if config_file == "config.yaml" or "unified_config" in config_file or config_file == "unified_config.yaml" or "configs/config.yaml" in config_file or "test_single.yaml" in config_file:
        yaml_config = loader.load_yaml(config_file)
        # YAML 구조를 그대로 반환 (매핑 없이)
        return yaml_config
    
    # 기존 방식: UnifiedConfig 객체 생성 후 딕셔너리로 변환
    unified_config = loader.create_unified_config(config_file, args_dict, mode)
    return unified_config.to_dict()


class ConfigObject:
    """Dict를 dot notation으로 접근할 수 있는 객체로 변환"""
    def __init__(self, config_dict: Dict[str, Any]):
        self._config_dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def update(self, other_dict: Dict[str, Any]):
        """딕셔너리 업데이트 (호환성용)"""
        self._config_dict.update(other_dict)
        for key, value in other_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return self._config_dict.copy()
    
    def items(self):
        """딕셔너리 items() 메서드"""
        return self._config_dict.items()
    
    def keys(self):
        """딕셔너리 keys() 메서드"""
        return self._config_dict.keys()
    
    def values(self):
        """딕셔너리 values() 메서드"""
        return self._config_dict.values()

def get_legacy_config_dict(config: UnifiedConfig, pipeline_type: str) -> ConfigObject:
    """기존 코드와의 호환성을 위한 레거시 설정 객체 생성"""
    
    base_dict = {
        'mode': config.mode,
        'input': config.input_source,
        'output_dir': config.output_dir,
        'window_size': config.window_size,
        'inference_stride': config.inference_stride,
        'batch_size': config.batch_size,
        'device': config.device,
        'enable_evaluation': config.enable_evaluation,
        'enable_visualization': config.enable_visualization,
    }
    
    # 파이프라인별 특화 설정
    if pipeline_type == "inference":
        base_dict.update({
            'target_fps': config.target_fps,
            'max_queue_size': config.max_queue_size,
            'skip_frames': config.skip_frames,
            'min_confidence': config.min_confidence,
            'alert_threshold': config.alert_threshold,
            'resize_input': getattr(config, 'resize_input', None),
        })
    
    elif pipeline_type == "separated":
        base_dict.update({
            'window_stride': config.window_stride,
            'stages_to_run': config.stages_to_run,
            'enable_resume': config.enable_resume,
            'save_intermediate_results': config.save_intermediate_results,
            'save_visualizations': config.save_visualizations,
        })
    
    elif pipeline_type == "unified":
        base_dict.update({
            'window_stride': config.window_stride,
            'save_intermediate_results': config.save_intermediate_results,
        })
    
    # 설정 그룹 추가 (기존 코드 호환성)
    base_dict.update({
        'pose_config': {
            'model_name': config.pose_model,
            'config_file': config.pose_config,
            'checkpoint_path': config.pose_checkpoint,
            'device': config.pose_device,
            'batch_size': config.pose_batch_size,
            'input_size': config.pose_input_size,
            'score_threshold': config.pose_score_threshold,
        },
        'tracking_config': {
            'tracker_name': config.tracker_name,
            'track_thresh': config.track_thresh,
            'track_buffer': config.track_buffer,
            'match_thresh': config.match_thresh,
        },
        'scoring_config': {
            'scorer_name': config.scorer_name,
            'quality_threshold': config.scoring_quality_threshold,
            'min_track_length': config.min_track_length,
        },
        'classification_config': {
            'model_name': config.classifier_name,
            'config_file': config.classifier_config,
            'checkpoint_path': config.classifier_checkpoint,
            'device': config.classifier_device,
            'window_size': config.classifier_window_size,
            'confidence_threshold': config.classifier_confidence_threshold,
            'class_names': config.class_names,
        },
    })
    
    # 분리형 파이프라인 추가 설정
    if pipeline_type == "separated":
        base_dict.update({
            'stage1_output_dir': f"{config.output_dir}/stage1",
            'stage2_output_dir': f"{config.output_dir}/stage2", 
            'stage3_output_dir': f"{config.output_dir}/stage3",
            'stage4_output_dir': f"{config.output_dir}/stage4",
            'enable_multiprocessing': config.enable_multiprocessing,
            'num_workers': config.num_workers,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'enable_quality_filter': config.enable_quality_filter,
        })
    
    return ConfigObject(base_dict)