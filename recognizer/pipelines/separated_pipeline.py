"""
통합 분리형 파이프라인 (Unified Separated Pipeline)

annotation_pipeline.py 기능을 통합한 3단계 분리 처리:
- Stage 1: 포즈 추정 → 시각화용 PKL 저장
- Stage 2: 트래킹 → 시각화용 PKL 저장  
- Stage 3: 복합점수 반영 정렬 → 시각화용 PKL 저장
- Stage 4: STGCN 훈련용 통합 PKL 생성 (train/val/test)

각 단계별 독립 실행 가능, resume 기능 지원, 단계별 시각화 데이터 제공
"""

import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import multiprocessing as mp

from ..utils.factory import ModuleFactory
from ..utils.data_structure import (
    PersonPose, FramePoses, WindowAnnotation, ClassificationResult,
    PoseEstimationConfig, TrackingConfig, ScoringConfig, ActionClassificationConfig
)
from ..utils.multiprocess_manager import MultiprocessManager
from ..utils.file_utils import ensure_directory


@dataclass
class SeparatedPipelineConfig:
    """통합 분리형 파이프라인 설정"""
    # 모듈 설정
    pose_config: PoseEstimationConfig
    tracking_config: TrackingConfig
    scoring_config: ScoringConfig
    classification_config: ActionClassificationConfig
    
    # 윈도우 설정
    window_size: int = 100
    window_stride: int = 50
    
    # 출력 디렉토리 설정 (시각화 지원)
    stage1_output_dir: str = "output/separated/stage1_poses"           # 포즈 추정 → 시각화
    stage2_output_dir: str = "output/separated/stage2_tracking"        # 트래킹 → 시각화
    stage3_output_dir: str = "output/separated/stage3_scoring"         # 복합점수 정렬 → 시각화
    stage4_output_dir: str = "output/separated/stage4_unified"         # STGCN 훈련용 통합
    
    # 단계별 실행 제어 (annotation_pipeline 통합)
    stages_to_run: List[str] = field(default_factory=lambda: ["stage1", "stage2", "stage3", "stage4"])
    annotation_mode: bool = False  # True: annotation 모드 (stage1-2만), False: 전체 파이프라인
    
    # annotation_pipeline 품질 관리 설정
    min_pose_confidence: float = 0.3
    min_track_length: int = 10
    max_persons_per_frame: int = 10
    
    # 출력 형식 제어
    output_format: str = 'pickle'  # pickle, json
    save_poses: bool = True
    save_tracking: bool = True
    save_bboxes: bool = True
    save_keypoints: bool = True
    annotation_mode: bool = False  # True시 stage1+2+3만 실행 (분류 제외)
    
    # 품질 관리 (annotation_pipeline 통합)
    min_pose_confidence: float = 0.3
    min_track_length: int = 10
    max_persons_per_frame: int = 10
    
    # 데이터셋 분할 (STGCN 훈련용)
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # 출력 형식
    save_visualization_pkl: bool = True   # 시각화용 PKL 저장
    save_stgcn_pkl: bool = True          # STGCN 훈련용 PKL 저장
    
    # Resume 기능
    enable_resume: bool = True
    
    # 멀티프로세스 설정
    num_workers: int = field(default_factory=lambda: mp.cpu_count())
    enable_multiprocess: bool = True
    multiprocess_batch_size: int = 4
    multiprocess_timeout: float = 600.0


@dataclass
class StageResult:
    """단계별 처리 결과"""
    stage_id: int
    input_path: str
    output_path: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationData:
    """시각화용 데이터 구조"""
    stage: str  # stage1, stage2, stage3
    video_path: str
    video_info: Dict[str, Any]
    
    # Stage 1: 포즈 추정 결과
    pose_data: Optional[List[FramePoses]] = None
    
    # Stage 2: 트래킹 결과
    tracking_data: Optional[List[FramePoses]] = None
    track_summary: Optional[Dict[int, Dict[str, Any]]] = None
    
    # Stage 3: 복합점수 반영 정렬 결과
    scoring_data: Optional[List[WindowAnnotation]] = None
    ranked_windows: Optional[List[Dict[str, Any]]] = None
    
    # annotation_pipeline 호환 데이터 (stage1-2용)
    frame_annotations: Optional[List[Dict[str, Any]]] = None  # 프레임별 어노테이션
    
    # 공통 메타데이터
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    quality_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class STGCNData:
    """STGCN 훈련용 데이터 구조"""
    # 데이터셋 분할
    train_data: List[Dict[str, Any]]
    val_data: List[Dict[str, Any]]
    test_data: List[Dict[str, Any]]
    
    # 라벨 정보
    label_map: Dict[str, int]  # {"NonFight": 0, "Fight": 1}
    class_names: List[str]
    
    # 데이터 통계
    dataset_stats: Dict[str, Any]
    
    # STGCN 호환 메타데이터
    num_classes: int
    keypoint_info: Dict[str, Any]
    skeleton_info: List[List[int]]
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class SeparatedPipeline:
    """분리형 파이프라인"""
    
    def __init__(self, config: SeparatedPipelineConfig):
        """
        Args:
            config: 분리형 파이프라인 설정
        """
        self.config = config
        self.factory = ModuleFactory()
        
        # 모듈 인스턴스들
        self.pose_estimator = None
        self.tracker = None
        self.scorer = None
        self.classifier = None
        
        # 멀티프로세스 매니저
        self.multiprocess_manager = None
        
        # 통계 정보 (Stage 4 추가)
        self.stats = {
            'stage1': {'processed': 0, 'failed': 0, 'total_time': 0.0},
            'stage2': {'processed': 0, 'failed': 0, 'total_time': 0.0},
            'stage3': {'processed': 0, 'failed': 0, 'total_time': 0.0},
            'stage4': {'processed': 0, 'failed': 0, 'total_time': 0.0}
        }
        
        self._initialize_directories()
    
    def _initialize_directories(self):
        """출력 디렉토리 초기화"""
        ensure_directory(self.config.stage1_output_dir)
        ensure_directory(self.config.stage2_output_dir)
        ensure_directory(self.config.stage3_output_dir)
        ensure_directory(self.config.stage4_output_dir)
    
    def run_stage1(self, video_paths: List[Union[str, Path]]) -> List[StageResult]:
        """Stage 1: 포즈 추정 → pkl 저장
        
        Args:
            video_paths: 처리할 비디오 파일 경로 리스트
            
        Returns:
            각 비디오별 처리 결과
        """
        logging.info(f"Starting Stage 1: Pose Estimation for {len(video_paths)} videos")
        
        # Resume 모드에서 이미 처리된 파일 확인
        if self.config.enable_resume:
            video_paths = self._filter_completed_stage1(video_paths)
            if not video_paths:
                logging.info("All videos already processed in Stage 1")
                return []
        
        # 포즈 추정기 초기화
        if not self._initialize_pose_estimator():
            raise RuntimeError("Failed to initialize pose estimator")
        
        results = []
        start_time = time.time()
        
        if self.config.enable_multiprocess and len(video_paths) > 1:
            # 멀티프로세스 처리
            results = self._run_stage1_multiprocess(video_paths)
        else:
            # 단일 프로세스 처리
            for video_path in video_paths:
                result = self._process_stage1_single_video(video_path)
                results.append(result)
        
        # 통계 업데이트
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        self.stats['stage1']['processed'] = len(successful_results)
        self.stats['stage1']['failed'] = len(failed_results)
        self.stats['stage1']['total_time'] = total_time
        
        logging.info(f"Stage 1 completed: {len(successful_results)} success, "
                    f"{len(failed_results)} failed, {total_time:.2f}s")
        
        return results
    
    def run_stage2(self, stage1_results: Optional[List[StageResult]] = None) -> List[StageResult]:
        """Stage 2: 트래킹 & 복합점수 → pkl 저장
        
        Args:
            stage1_results: Stage 1 결과, None이면 자동으로 찾음
            
        Returns:
            각 비디오별 처리 결과
        """
        logging.info("Starting Stage 2: Tracking & Composite Scoring")
        
        # Stage 1 결과 파일 찾기
        if stage1_results is None:
            stage1_pkl_files = list(Path(self.config.stage1_output_dir).glob("*.pkl"))
        else:
            stage1_pkl_files = [Path(r.output_path) for r in stage1_results if r.success]
        
        # Resume 모드에서 이미 처리된 파일 확인
        if self.config.enable_resume:
            stage1_pkl_files = self._filter_completed_stage2(stage1_pkl_files)
            if not stage1_pkl_files:
                logging.info("All files already processed in Stage 2")
                return []
        
        # 트래커와 스코어러 초기화
        if not self._initialize_tracker_scorer():
            raise RuntimeError("Failed to initialize tracker and scorer")
        
        results = []
        start_time = time.time()
        
        if self.config.enable_multiprocess and len(stage1_pkl_files) > 1:
            # 멀티프로세스 처리
            results = self._run_stage2_multiprocess(stage1_pkl_files)
        else:
            # 단일 프로세스 처리
            for pkl_file in stage1_pkl_files:
                result = self._process_stage2_single_file(pkl_file)
                results.append(result)
        
        # 통계 업데이트
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        self.stats['stage2']['processed'] = len(successful_results)
        self.stats['stage2']['failed'] = len(failed_results)
        self.stats['stage2']['total_time'] = total_time
        
        logging.info(f"Stage 2 completed: {len(successful_results)} success, "
                    f"{len(failed_results)} failed, {total_time:.2f}s")
        
        return results
    
    def run_stage3(self, stage2_results: Optional[List[StageResult]] = None, 
                   class_labels: Optional[Dict[str, str]] = None) -> List[StageResult]:
        """Stage 3: 비디오별 pkl 통합 → 최종 결과
        
        Args:
            stage2_results: Stage 2 결과, None이면 자동으로 찾음
            class_labels: 비디오별 클래스 라벨 (선택사항)
            
        Returns:
            각 비디오별 최종 분류 결과
        """
        logging.info("Starting Stage 3: Video Integration & Classification")
        
        # Stage 2 결과 파일 찾기
        if stage2_results is None:
            stage2_pkl_files = list(Path(self.config.stage2_output_dir).glob("*.pkl"))
        else:
            stage2_pkl_files = [Path(r.output_path) for r in stage2_results if r.success]
        
        # Resume 모드에서 이미 처리된 파일 확인
        if self.config.enable_resume:
            stage2_pkl_files = self._filter_completed_stage3(stage2_pkl_files)
            if not stage2_pkl_files:
                logging.info("All files already processed in Stage 3")
                return []
        
        # 분류기 초기화
        if not self._initialize_classifier():
            raise RuntimeError("Failed to initialize classifier")
        
        results = []
        start_time = time.time()
        
        if self.config.enable_multiprocess and len(stage2_pkl_files) > 1:
            # 멀티프로세스 처리
            results = self._run_stage3_multiprocess(stage2_pkl_files, class_labels)
        else:
            # 단일 프로세스 처리
            for pkl_file in stage2_pkl_files:
                video_name = pkl_file.stem
                true_label = class_labels.get(video_name, None) if class_labels else None
                result = self._process_stage3_single_file(pkl_file, true_label)
                results.append(result)
        
        # 통계 업데이트
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        self.stats['stage3']['processed'] = len(successful_results)
        self.stats['stage3']['failed'] = len(failed_results)
        self.stats['stage3']['total_time'] = total_time
        
        logging.info(f"Stage 3 completed: {len(successful_results)} success, "
                    f"{len(failed_results)} failed, {total_time:.2f}s")
        
        # STGCN 호환 통합 데이터셋 생성
        if successful_results:
            self._create_unified_stgcn_dataset(successful_results)
        
        return results
    
    def run_full_pipeline(self, video_paths: List[Union[str, Path]], 
                         class_labels: Optional[Dict[str, str]] = None) -> Dict[str, List[StageResult]]:
        """전체 파이프라인 실행 (Stage 1 → 2 → 3)
        
        Args:
            video_paths: 처리할 비디오 파일 경로 리스트
            class_labels: 비디오별 클래스 라벨 (선택사항)
            
        Returns:
            각 단계별 처리 결과
        """
        logging.info(f"Starting full separated pipeline for {len(video_paths)} videos")
        
        # Stage 1: 포즈 추정
        stage1_results = self.run_stage1(video_paths)
        
        # Stage 2: 트래킹 & 스코어링
        stage2_results = self.run_stage2(stage1_results)
        
        # Stage 3: 통합 & 분류
        stage3_results = self.run_stage3(stage2_results, class_labels)
        
        results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'stage3': stage3_results
        }
        
        # 전체 통계 출력
        self._print_pipeline_summary()
        
        return results
    
    def _initialize_pose_estimator(self) -> bool:
        """포즈 추정기 초기화"""
        try:
            if self.pose_estimator is None:
                self.pose_estimator = self.factory.create_pose_estimator(
                    self.config.pose_config.model_name,
                    self.config.pose_config.__dict__
                )
                return self.pose_estimator.initialize_model()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize pose estimator: {str(e)}")
            return False
    
    def _initialize_tracker_scorer(self) -> bool:
        """트래커와 스코어러 초기화"""
        try:
            if self.tracker is None:
                self.tracker = self.factory.create_tracker(
                    self.config.tracking_config.tracker_name,
                    self.config.tracking_config.__dict__
                )
                if not self.tracker.initialize_tracker():
                    return False
            
            if self.scorer is None:
                self.scorer = self.factory.create_scorer(
                    self.config.scoring_config.scorer_name,
                    self.config.scoring_config.__dict__
                )
                return self.scorer.initialize_scorer()
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize tracker/scorer: {str(e)}")
            return False
    
    def _initialize_classifier(self) -> bool:
        """분류기 초기화"""
        try:
            if self.classifier is None:
                self.classifier = self.factory.create_classifier(
                    self.config.classification_config.model_name,
                    self.config.classification_config.__dict__
                )
                return self.classifier.initialize_model()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize classifier: {str(e)}")
            return False
    
    def _process_stage1_single_video(self, video_path: Union[str, Path]) -> StageResult:
        """단일 비디오의 Stage 1 처리"""
        video_path = Path(video_path)
        output_path = Path(self.config.stage1_output_dir) / f"{video_path.stem}.pkl"
        
        start_time = time.time()
        
        try:
            # 포즈 추정 실행
            frame_poses_list = self.pose_estimator.extract_poses_from_video(str(video_path))
            
            if not frame_poses_list:
                raise RuntimeError("No poses extracted from video")
            
            # 결과 저장
            with open(output_path, 'wb') as f:
                pickle.dump(frame_poses_list, f)
            
            processing_time = time.time() - start_time
            
            return StageResult(
                stage_id=1,
                input_path=str(video_path),
                output_path=str(output_path),
                processing_time=processing_time,
                success=True,
                metadata={
                    'total_frames': len(frame_poses_list),
                    'total_persons': sum(len(fp.persons) for fp in frame_poses_list)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Stage 1 failed for {video_path}: {str(e)}")
            
            return StageResult(
                stage_id=1,
                input_path=str(video_path),
                output_path=str(output_path),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _process_stage2_single_file(self, pkl_file: Path) -> StageResult:
        """단일 파일의 Stage 2 처리"""
        output_path = Path(self.config.stage2_output_dir) / pkl_file.name
        
        start_time = time.time()
        
        try:
            # Stage 1 결과 로드
            with open(pkl_file, 'rb') as f:
                frame_poses_list = pickle.load(f)
            
            # 트래킹 실행
            tracked_poses = []
            for frame_poses in frame_poses_list:
                tracked_frame = self.tracker.track_frame_poses(frame_poses)
                tracked_poses.append(tracked_frame)
            
            # 복합점수 계산
            scores = self.scorer.calculate_scores(tracked_poses)
            
            # 결과 저장
            stage2_data = {
                'tracked_poses': tracked_poses,
                'composite_scores': scores,
                'metadata': {
                    'total_frames': len(tracked_poses),
                    'unique_tracks': len(set(
                        person.track_id for fp in tracked_poses 
                        for person in fp.persons 
                        if person.track_id is not None
                    ))
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(stage2_data, f)
            
            processing_time = time.time() - start_time
            
            return StageResult(
                stage_id=2,
                input_path=str(pkl_file),
                output_path=str(output_path),
                processing_time=processing_time,
                success=True,
                metadata=stage2_data['metadata']
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Stage 2 failed for {pkl_file}: {str(e)}")
            
            return StageResult(
                stage_id=2,
                input_path=str(pkl_file),
                output_path=str(output_path),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _process_stage3_single_file(self, pkl_file: Path, 
                                   true_label: Optional[str] = None) -> StageResult:
        """단일 파일의 Stage 3 처리"""
        output_path = Path(self.config.stage3_output_dir) / pkl_file.name
        
        start_time = time.time()
        
        try:
            # Stage 2 결과 로드
            with open(pkl_file, 'rb') as f:
                stage2_data = pickle.load(f)
            
            tracked_poses = stage2_data['tracked_poses']
            
            # 윈도우 생성 및 분류
            windows = self._create_windows_from_poses(tracked_poses)
            classification_results = []
            
            for window in windows:
                result = self.classifier.classify_single_window(window)
                classification_results.append(result)
            
            # 비디오 전체 예측 (가장 높은 신뢰도)
            if classification_results:
                best_result = max(classification_results, key=lambda x: x.confidence)
                video_prediction = best_result.predicted_class
                video_confidence = best_result.confidence
            else:
                video_prediction = "NonFight"  # 기본값
                video_confidence = 0.0
            
            # 최종 결과 저장
            final_result = {
                'video_name': pkl_file.stem,
                'predicted_class': video_prediction,
                'confidence': video_confidence,
                'true_label': true_label,
                'window_results': classification_results,
                'metadata': {
                    'total_windows': len(windows),
                    'avg_confidence': sum(r.confidence for r in classification_results) / len(classification_results) if classification_results else 0.0
                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(final_result, f)
            
            processing_time = time.time() - start_time
            
            return StageResult(
                stage_id=3,
                input_path=str(pkl_file),
                output_path=str(output_path),
                processing_time=processing_time,
                success=True,
                metadata={
                    **final_result['metadata'],
                    'predicted_class': video_prediction,
                    'confidence': video_confidence
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"Stage 3 failed for {pkl_file}: {str(e)}")
            
            return StageResult(
                stage_id=3,
                input_path=str(pkl_file),
                output_path=str(output_path),
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _create_windows_from_poses(self, tracked_poses: List[FramePoses]) -> List[WindowAnnotation]:
        """포즈 리스트에서 윈도우 생성"""
        windows = []
        
        for start_idx in range(0, len(tracked_poses) - self.config.window_size + 1, 
                              self.config.window_stride):
            end_idx = start_idx + self.config.window_size
            window_poses = tracked_poses[start_idx:end_idx]
            
            # 고유한 person 수 계산
            unique_persons = set()
            for frame_poses in window_poses:
                for person in frame_poses.persons:
                    if person.track_id is not None:
                        unique_persons.add(person.track_id)
            
            window = WindowAnnotation(
                window_id=start_idx,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                poses=window_poses,
                total_persons=len(unique_persons)
            )
            windows.append(window)
        
        return windows
    
    def _filter_completed_stage1(self, video_paths: List[Union[str, Path]]) -> List[Path]:
        """Stage 1에서 이미 완료된 비디오 제외"""
        remaining_paths = []
        
        for video_path in video_paths:
            video_path = Path(video_path)
            output_path = Path(self.config.stage1_output_dir) / f"{video_path.stem}.pkl"
            
            if not output_path.exists():
                remaining_paths.append(video_path)
            else:
                logging.debug(f"Skipping {video_path.name} (already processed in Stage 1)")
        
        return remaining_paths
    
    def _filter_completed_stage2(self, pkl_files: List[Path]) -> List[Path]:
        """Stage 2에서 이미 완료된 파일 제외"""
        remaining_files = []
        
        for pkl_file in pkl_files:
            output_path = Path(self.config.stage2_output_dir) / pkl_file.name
            
            if not output_path.exists():
                remaining_files.append(pkl_file)
            else:
                logging.debug(f"Skipping {pkl_file.name} (already processed in Stage 2)")
        
        return remaining_files
    
    def _filter_completed_stage3(self, pkl_files: List[Path]) -> List[Path]:
        """Stage 3에서 이미 완료된 파일 제외"""
        remaining_files = []
        
        for pkl_file in pkl_files:
            output_path = Path(self.config.stage3_output_dir) / pkl_file.name
            
            if not output_path.exists():
                remaining_files.append(pkl_file)
            else:
                logging.debug(f"Skipping {pkl_file.name} (already processed in Stage 3)")
        
        return remaining_files
    
    def _create_unified_stgcn_dataset(self, stage3_results: List[StageResult]):
        """STGCN 호환 통합 데이터셋 생성"""
        logging.info("Creating unified STGCN-compatible dataset...")
        
        # 모든 비디오 결과 수집
        all_video_data = []
        
        for result in stage3_results:
            if not result.success:
                continue
                
            try:
                # Stage 3 결과 파일 로드
                with open(result.output_path, 'rb') as f:
                    video_result = pickle.load(f)
                
                # 비디오 메타데이터 추출
                video_name = Path(result.input_path).stem
                
                # 클래스 라벨 추출 (폴더명 또는 메타데이터에서)
                if 'Fight' in result.input_path:
                    label = 1
                    label_folder = 'Fight'
                elif 'NonFight' in result.input_path:
                    label = 0
                    label_folder = 'NonFight'
                else:
                    # 메타데이터에서 추출 시도
                    label = result.metadata.get('true_label', 0)
                    label_folder = 'Fight' if label == 1 else 'NonFight'
                
                # Stage 2에서 트래킹 데이터 로드
                stage2_pkl = Path(self.config.stage2_output_dir) / f"{video_name}.pkl"
                if not stage2_pkl.exists():
                    logging.warning(f"Stage 2 data not found for {video_name}")
                    continue
                    
                with open(stage2_pkl, 'rb') as f:
                    tracking_data = pickle.load(f)
                
                # STGCN 호환 형식으로 비디오 데이터 구성 (레거시 형식 복제)
                video_data = {
                    'video_name': video_name,
                    'label_folder': label_folder,
                    'label': label,
                    'dataset_name': 'separated_pipeline',
                    'total_frames': len(tracking_data.get('tracked_poses', [])),
                    'num_windows': len(tracking_data.get('windows', [])),
                    'windows': self._convert_windows_to_legacy_format(
                        tracking_data.get('windows', []),
                        video_name, 
                        label
                    ),
                    'tracking_settings': {
                        'window_size': self.config.window_size,
                        'window_stride': self.config.window_stride,
                        'classification_confidence': result.metadata.get('confidence', 0.0)
                    }
                }
                
                all_video_data.append(video_data)
                
            except Exception as e:
                logging.error(f"Error processing {result.input_path} for unified dataset: {str(e)}")
                continue
        
        if not all_video_data:
            logging.warning("No valid video data found for unified dataset")
            return
        
        logging.info(f"Loaded {len(all_video_data)} video results for unified dataset")
        
        # 카테고리별로 분리
        fight_videos = [v for v in all_video_data if v['label'] == 1]
        nonfight_videos = [v for v in all_video_data if v['label'] == 0]
        
        logging.info(f"Fight videos: {len(fight_videos)}, NonFight videos: {len(nonfight_videos)}")
        
        # 데이터셋 분할 (레거시 로직과 동일)
        train_data, val_data, test_data = self._split_dataset_for_stgcn(fight_videos, nonfight_videos)
        
        # 통합 pkl 파일 생성 (레거시 형식과 동일)
        unified_dir = Path(self.config.stage3_output_dir) / "unified"
        ensure_directory(str(unified_dir))
        
        dataset_name = "separated_pipeline_dataset"
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        saved_files = []
        for split_name, split_data in splits.items():
            if split_data:
                pkl_filename = f"{dataset_name}_{split_name}_windows.pkl"
                pkl_path = unified_dir / pkl_filename
                
                with open(pkl_path, 'wb') as f:
                    pickle.dump(split_data, f)
                
                saved_files.append(str(pkl_path))
                logging.info(f"  {split_name}: {len(split_data)} videos -> {pkl_path}")
            else:
                logging.info(f"  {split_name}: No data")
        
        # 통계 정보 저장
        self._save_unified_dataset_statistics(all_video_data, dataset_name, unified_dir)
        
        logging.info(f"STGCN-compatible unified dataset created: {len(saved_files)} files")
        for file_path in saved_files:
            logging.info(f"  - {file_path}")
    
    def _convert_windows_to_legacy_format(self, windows: List[WindowAnnotation], 
                                        video_name: str, label: int) -> List[Dict]:
        """윈도우를 레거시 형식으로 변환"""
        legacy_windows = []
        
        for window_idx, window in enumerate(windows):
            # 레거시 형식에 맞는 윈도우 데이터 구성
            window_result = {
                'window_idx': window_idx,
                'start_frame': window.start_frame,
                'end_frame': window.end_frame,
                'num_frames': window.end_frame - window.start_frame + 1,
                'annotation': {
                    'annotation': self._extract_annotation_data(window),
                    'keypoints': self._extract_keypoints_data(window),
                    'total_frames': len(window.poses)
                },
                'segment_video_path': None,
                'persons_ranking': self._extract_persons_ranking_legacy(window),
                'composite_score': self._calculate_window_composite_score(window)
            }
            
            legacy_windows.append(window_result)
        
        # 복합점수로 정렬 (내림차순)
        legacy_windows.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        
        return legacy_windows
    
    def _extract_annotation_data(self, window: WindowAnnotation) -> Dict:
        """윈도우에서 어노테이션 데이터 추출"""
        annotation_data = {}
        
        # 각 person별 데이터 구성
        person_tracks = {}
        for frame_poses in window.poses:
            for person in frame_poses.persons:
                if person.track_id is not None:
                    if person.track_id not in person_tracks:
                        person_tracks[person.track_id] = []
                    person_tracks[person.track_id].append(person)
        
        # person별 어노테이션 생성
        for rank, (track_id, person_poses) in enumerate(person_tracks.items()):
            person_key = f"person_{rank + 1}"
            
            # 평균 스코어 계산
            avg_score = sum(p.score for p in person_poses) / len(person_poses) if person_poses else 0.0
            
            annotation_data[person_key] = {
                'rank': rank + 1,
                'track_id': track_id,
                'composite_score': avg_score,
                'keypoints_sequence': [p.keypoints.tolist() if hasattr(p.keypoints, 'tolist') else p.keypoints 
                                     for p in person_poses],
                'bbox_sequence': [p.bbox.tolist() if hasattr(p.bbox, 'tolist') else p.bbox 
                                for p in person_poses],
                'score_sequence': [p.score for p in person_poses]
            }
        
        return annotation_data
    
    def _extract_keypoints_data(self, window: WindowAnnotation) -> List:
        """윈도우에서 키포인트 데이터 추출"""
        keypoints_data = []
        
        for frame_poses in window.poses:
            frame_keypoints = []
            for person in frame_poses.persons:
                if hasattr(person.keypoints, 'tolist'):
                    frame_keypoints.append(person.keypoints.tolist())
                else:
                    frame_keypoints.append(person.keypoints)
            keypoints_data.append(frame_keypoints)
        
        return keypoints_data
    
    def _extract_persons_ranking_legacy(self, window: WindowAnnotation) -> List[Dict]:
        """레거시 형식의 person 랭킹 추출"""
        person_tracks = {}
        for frame_poses in window.poses:
            for person in frame_poses.persons:
                if person.track_id is not None:
                    if person.track_id not in person_tracks:
                        person_tracks[person.track_id] = []
                    person_tracks[person.track_id].append(person)
        
        persons = []
        for rank, (track_id, person_poses) in enumerate(person_tracks.items()):
            avg_score = sum(p.score for p in person_poses) / len(person_poses) if person_poses else 0.0
            
            persons.append({
                'person_id': f"person_{rank + 1}",
                'rank': rank + 1,
                'composite_score': avg_score,
                'track_id': track_id
            })
        
        # 스코어로 정렬
        persons.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return persons
    
    def _calculate_window_composite_score(self, window: WindowAnnotation) -> float:
        """윈도우의 복합점수 계산"""
        total_score = 0.0
        person_count = 0
        
        for frame_poses in window.poses:
            for person in frame_poses.persons:
                total_score += person.score
                person_count += 1
        
        return total_score / person_count if person_count > 0 else 0.0
    
    def _split_dataset_for_stgcn(self, fight_videos: List[Dict], 
                               nonfight_videos: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """데이터셋을 train/val/test로 분할 (레거시 로직 복제)"""
        import random
        
        # 재현 가능한 분할을 위한 시드 설정
        random.seed(42)
        
        # 기본 분할 비율 (레거시와 동일)
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        def split_category(videos: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            """카테고리별 분할"""
            random.shuffle(videos)
            total = len(videos)
            
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            train = videos[:train_end]
            val = videos[train_end:val_end]
            test = videos[val_end:]
            
            return train, val, test
        
        # 각 카테고리별로 분할
        fight_train, fight_val, fight_test = split_category(fight_videos)
        nonfight_train, nonfight_val, nonfight_test = split_category(nonfight_videos)
        
        # 통합
        train_data = fight_train + nonfight_train
        val_data = fight_val + nonfight_val
        test_data = fight_test + nonfight_test
        
        # 셔플
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def _save_unified_dataset_statistics(self, all_video_data: List[Dict], 
                                       dataset_name: str, output_dir: Path):
        """통합 데이터셋 통계 정보 저장"""
        import json
        
        stats = {
            'dataset_name': dataset_name,
            'total_videos': len(all_video_data),
            'fight_videos': len([v for v in all_video_data if v['label'] == 1]),
            'nonfight_videos': len([v for v in all_video_data if v['label'] == 0]),
            'total_windows': sum(v['num_windows'] for v in all_video_data),
            'avg_windows_per_video': sum(v['num_windows'] for v in all_video_data) / len(all_video_data),
            'avg_frames_per_video': sum(v['total_frames'] for v in all_video_data) / len(all_video_data),
            'config': {
                'window_size': self.config.window_size,
                'window_stride': self.config.window_stride
            },
            'split_ratios': {
                'train': 0.7,
                'val': 0.15,
                'test': 0.15
            }
        }
        
        # 윈도우별 통계
        all_windows = []
        for video in all_video_data:
            all_windows.extend(video['windows'])
        
        if all_windows:
            window_scores = [w.get('composite_score', 0.0) for w in all_windows]
            stats['window_statistics'] = {
                'total_windows': len(all_windows),
                'avg_composite_score': sum(window_scores) / len(window_scores),
                'max_composite_score': max(window_scores),
                'min_composite_score': min(window_scores)
            }
        
        # 통계 파일 저장
        stats_file = output_dir / f"{dataset_name}_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Dataset statistics saved to: {stats_file}")
        
        # 요약 출력
        logging.info(f"\nDataset Summary:")
        logging.info(f"  Total Videos: {stats['total_videos']} (Fight: {stats['fight_videos']}, NonFight: {stats['nonfight_videos']})")
        logging.info(f"  Total Windows: {stats['total_windows']}")
        logging.info(f"  Avg Windows/Video: {stats['avg_windows_per_video']:.1f}")
        logging.info(f"  Avg Frames/Video: {stats['avg_frames_per_video']:.1f}")
        if 'window_statistics' in stats:
            logging.info(f"  Avg Composite Score: {stats['window_statistics']['avg_composite_score']:.3f}")
            logging.info(f"  Score Range: {stats['window_statistics']['min_composite_score']:.3f} - {stats['window_statistics']['max_composite_score']:.3f}")
    
    def _run_stage1_multiprocess(self, video_paths: List[Path]) -> List[StageResult]:
        """Stage 1 멀티프로세스 실행"""
        with MultiprocessManager(self.config.num_workers) as manager:
            task_ids = []
            
            # 작업 제출
            for video_path in video_paths:
                task_id = manager.submit_task(
                    'process_stage1_pose_extraction',
                    str(video_path),
                    self.config.pose_config.__dict__,
                    self.config.stage1_output_dir
                )
                task_ids.append(task_id)
            
            # 결과 수집
            results = manager.get_batch_results(task_ids, self.config.multiprocess_timeout)
            
            # StageResult 객체로 변환
            stage_results = []
            for result in results:
                if result.success:
                    stage_results.append(result.result)
                else:
                    # 실패한 작업에 대한 StageResult 생성
                    stage_results.append(StageResult(
                        stage_id=1,
                        input_path="unknown",
                        output_path="unknown",
                        processing_time=0.0,
                        success=False,
                        error_message=result.error
                    ))
            
            return stage_results
    
    def _run_stage2_multiprocess(self, pkl_files: List[Path]) -> List[StageResult]:
        """Stage 2 멀티프로세스 실행"""
        with MultiprocessManager(self.config.num_workers) as manager:
            task_ids = []
            
            # 작업 제출
            for pkl_file in pkl_files:
                task_id = manager.submit_task(
                    'process_stage2_tracking_scoring',
                    str(pkl_file),
                    self.config.tracking_config.__dict__,
                    self.config.scoring_config.__dict__,
                    self.config.stage2_output_dir
                )
                task_ids.append(task_id)
            
            # 결과 수집
            results = manager.get_batch_results(task_ids, self.config.multiprocess_timeout)
            
            # StageResult 객체로 변환
            stage_results = []
            for result in results:
                if result.success:
                    stage_results.append(result.result)
                else:
                    stage_results.append(StageResult(
                        stage_id=2,
                        input_path="unknown",
                        output_path="unknown",
                        processing_time=0.0,
                        success=False,
                        error_message=result.error
                    ))
            
            return stage_results
    
    def _run_stage3_multiprocess(self, pkl_files: List[Path], 
                                 class_labels: Optional[Dict[str, str]]) -> List[StageResult]:
        """Stage 3 멀티프로세스 실행"""
        with MultiprocessManager(self.config.num_workers) as manager:
            task_ids = []
            
            # 작업 제출
            for pkl_file in pkl_files:
                video_name = pkl_file.stem
                true_label = class_labels.get(video_name, None) if class_labels else None
                
                task_id = manager.submit_task(
                    'process_stage3_classification',
                    str(pkl_file),
                    self.config.classification_config.__dict__,
                    self.config.window_size,
                    self.config.window_stride,
                    self.config.stage3_output_dir,
                    true_label
                )
                task_ids.append(task_id)
            
            # 결과 수집
            results = manager.get_batch_results(task_ids, self.config.multiprocess_timeout)
            
            # StageResult 객체로 변환
            stage_results = []
            for result in results:
                if result.success:
                    stage_results.append(result.result)
                else:
                    stage_results.append(StageResult(
                        stage_id=3,
                        input_path="unknown",
                        output_path="unknown",
                        processing_time=0.0,
                        success=False,
                        error_message=result.error
                    ))
            
            return stage_results
    
    def _print_pipeline_summary(self):
        """파이프라인 실행 요약 출력"""
        logging.info("=== Separated Pipeline Summary ===")
        
        for stage, stats in self.stats.items():
            logging.info(f"{stage.upper()}: {stats['processed']} processed, "
                        f"{stats['failed']} failed, {stats['total_time']:.2f}s")
        
        total_processed = sum(stats['processed'] for stats in self.stats.values())
        total_failed = sum(stats['failed'] for stats in self.stats.values())
        total_time = sum(stats['total_time'] for stats in self.stats.values())
        
        logging.info(f"TOTAL: {total_processed} processed, {total_failed} failed, {total_time:.2f}s")
    
    def get_stage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """단계별 통계 반환"""
        return self.stats.copy()
    
    def cleanup(self):
        """리소스 정리"""
        if self.pose_estimator:
            self.pose_estimator.cleanup()
        
        if self.tracker:
            self.tracker.cleanup()
        
        if self.scorer:
            self.scorer.cleanup()
        
        if self.classifier:
            self.classifier.cleanup()
        
        if self.multiprocess_manager:
            self.multiprocess_manager.cleanup()
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.cleanup()


# 멀티프로세스용 전역 함수들
def process_stage1_pose_extraction(video_path: str, pose_config_dict: Dict[str, Any], 
                                  output_dir: str) -> StageResult:
    """Stage 1 포즈 추정 멀티프로세스 함수"""
    import time
    import pickle
    from pathlib import Path
    from ..utils.factory import ModuleFactory
    from ..utils.data_structure import PoseEstimationConfig
    
    video_path = Path(video_path)
    output_path = Path(output_dir) / f"{video_path.stem}.pkl"
    start_time = time.time()
    
    try:
        # 포즈 추정기 생성
        factory = ModuleFactory()
        pose_config = PoseEstimationConfig(**pose_config_dict)
        pose_estimator = factory.create_pose_estimator(pose_config.model_name, pose_config_dict)
        
        if not pose_estimator.initialize_model():
            raise RuntimeError("Failed to initialize pose estimator")
        
        # 포즈 추정 실행
        frame_poses_list = pose_estimator.extract_poses_from_video(str(video_path))
        
        if not frame_poses_list:
            raise RuntimeError("No poses extracted from video")
        
        # 결과 저장
        with open(output_path, 'wb') as f:
            pickle.dump(frame_poses_list, f)
        
        processing_time = time.time() - start_time
        
        # 리소스 정리
        pose_estimator.cleanup()
        
        return StageResult(
            stage_id=1,
            input_path=str(video_path),
            output_path=str(output_path),
            processing_time=processing_time,
            success=True,
            metadata={
                'total_frames': len(frame_poses_list),
                'total_persons': sum(len(fp.persons) for fp in frame_poses_list)
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return StageResult(
            stage_id=1,
            input_path=str(video_path),
            output_path=str(output_path),
            processing_time=processing_time,
            success=False,
            error_message=str(e)
        )


def process_stage2_tracking_scoring(pkl_file_path: str, tracking_config_dict: Dict[str, Any],
                                   scoring_config_dict: Dict[str, Any], output_dir: str) -> StageResult:
    """Stage 2 트래킹 & 스코어링 멀티프로세스 함수"""
    import time
    import pickle
    from pathlib import Path
    from ..utils.factory import ModuleFactory
    from ..utils.data_structure import TrackingConfig, ScoringConfig
    
    pkl_file = Path(pkl_file_path)
    output_path = Path(output_dir) / pkl_file.name
    start_time = time.time()
    
    try:
        # 트래커와 스코어러 생성
        factory = ModuleFactory()
        
        tracking_config = TrackingConfig(**tracking_config_dict)
        tracker = factory.create_tracker(tracking_config.tracker_name, tracking_config_dict)
        if not tracker.initialize_tracker():
            raise RuntimeError("Failed to initialize tracker")
        
        scoring_config = ScoringConfig(**scoring_config_dict)
        scorer = factory.create_scorer(scoring_config.scorer_name, scoring_config_dict)
        if not scorer.initialize_scorer():
            raise RuntimeError("Failed to initialize scorer")
        
        # Stage 1 결과 로드
        with open(pkl_file, 'rb') as f:
            frame_poses_list = pickle.load(f)
        
        # 트래킹 실행
        tracked_poses = []
        for frame_poses in frame_poses_list:
            tracked_frame = tracker.track_frame_poses(frame_poses)
            tracked_poses.append(tracked_frame)
        
        # 복합점수 계산
        scores = scorer.calculate_scores(tracked_poses)
        
        # 결과 저장
        stage2_data = {
            'tracked_poses': tracked_poses,
            'composite_scores': scores,
            'metadata': {
                'total_frames': len(tracked_poses),
                'unique_tracks': len(set(
                    person.track_id for fp in tracked_poses 
                    for person in fp.persons 
                    if person.track_id is not None
                ))
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(stage2_data, f)
        
        processing_time = time.time() - start_time
        
        # 리소스 정리
        tracker.cleanup()
        scorer.cleanup()
        
        return StageResult(
            stage_id=2,
            input_path=str(pkl_file),
            output_path=str(output_path),
            processing_time=processing_time,
            success=True,
            metadata=stage2_data['metadata']
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return StageResult(
            stage_id=2,
            input_path=str(pkl_file),
            output_path=str(output_path),
            processing_time=processing_time,
            success=False,
            error_message=str(e)
        )


def process_stage3_classification(pkl_file_path: str, classification_config_dict: Dict[str, Any],
                                 window_size: int, window_stride: int, output_dir: str, 
                                 true_label: Optional[str] = None) -> StageResult:
    """Stage 3 분류 멀티프로세스 함수"""
    import time
    import pickle
    from pathlib import Path
    from ..utils.factory import ModuleFactory
    from ..utils.data_structure import ActionClassificationConfig, WindowAnnotation
    
    pkl_file = Path(pkl_file_path)
    output_path = Path(output_dir) / pkl_file.name
    start_time = time.time()
    
    try:
        # 분류기 생성
        factory = ModuleFactory()
        classification_config = ActionClassificationConfig(**classification_config_dict)
        classifier = factory.create_classifier(classification_config.model_name, classification_config_dict)
        
        if not classifier.initialize_model():
            raise RuntimeError("Failed to initialize classifier")
        
        # Stage 2 결과 로드
        with open(pkl_file, 'rb') as f:
            stage2_data = pickle.load(f)
        
        tracked_poses = stage2_data['tracked_poses']
        
        # 윈도우 생성
        windows = []
        for start_idx in range(0, len(tracked_poses) - window_size + 1, window_stride):
            end_idx = start_idx + window_size
            window_poses = tracked_poses[start_idx:end_idx]
            
            # 고유한 person 수 계산
            unique_persons = set()
            for frame_poses in window_poses:
                for person in frame_poses.persons:
                    if person.track_id is not None:
                        unique_persons.add(person.track_id)
            
            window = WindowAnnotation(
                window_id=start_idx,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                poses=window_poses,
                total_persons=len(unique_persons)
            )
            windows.append(window)
        
        # 분류 실행
        classification_results = []
        for window in windows:
            result = classifier.classify_single_window(window)
            classification_results.append(result)
        
        # 비디오 전체 예측
        if classification_results:
            best_result = max(classification_results, key=lambda x: x.confidence)
            video_prediction = best_result.predicted_class
            video_confidence = best_result.confidence
        else:
            video_prediction = "NonFight"
            video_confidence = 0.0
        
        # 최종 결과 저장
        final_result = {
            'video_name': pkl_file.stem,
            'predicted_class': video_prediction,
            'confidence': video_confidence,
            'true_label': true_label,
            'window_results': classification_results,
            'metadata': {
                'total_windows': len(windows),
                'avg_confidence': sum(r.confidence for r in classification_results) / len(classification_results) if classification_results else 0.0
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(final_result, f)
        
        processing_time = time.time() - start_time
        
        # 리소스 정리
        classifier.cleanup()
        
        return StageResult(
            stage_id=3,
            input_path=str(pkl_file),
            output_path=str(output_path),
            processing_time=processing_time,
            success=True,
            metadata={
                **final_result['metadata'],
                'predicted_class': video_prediction,
                'confidence': video_confidence
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return StageResult(
            stage_id=3,
            input_path=str(pkl_file),
            output_path=str(output_path),
            processing_time=processing_time,
            success=False,
            error_message=str(e)
        )


# annotation_pipeline.py 통합 메서드들
class AnnotationIntegratedPipeline(SeparatedPipeline):
    """
    annotation_pipeline.py 기능을 통합한 파이프라인
    반복되는 코드를 제거하고 기존 SeparatedPipeline을 확장
    """
    
    def process_video_for_annotation(self, video_path: Union[str, Path], 
                                   label: Optional[str] = None) -> VisualizationData:
        """비디오 어노테이션 데이터 생성 (annotation_pipeline 호환)
        
        Args:
            video_path: 비디오 파일 경로
            label: 비디오 라벨 (Fight/NonFight 등)
            
        Returns:
            어노테이션 데이터 (시각화용 형식)
        """
        video_path = str(video_path)
        start_time = time.time()
        
        try:
            logging.info(f"Processing video for annotation: {video_path}")
            
            # 비디오 정보 추출
            video_info = self._extract_video_info(video_path)
            video_info['label'] = label
            
            # 1. 포즈 추정
            logging.info("Extracting poses...")
            if not self._initialize_pose_estimator():
                raise RuntimeError("Failed to initialize pose estimator")
                
            frame_poses = self.pose_estimator.extract_video_poses(video_path)
            if not frame_poses:
                raise RuntimeError("No poses extracted")
            
            # 품질 필터링
            filtered_poses = self._filter_poses_by_quality(frame_poses)
            logging.info(f"Filtered {len(filtered_poses)} high-quality frames from {len(frame_poses)}")
            
            # 2. 트래킹
            logging.info("Tracking persons...")
            if not self._initialize_tracker_scorer():
                raise RuntimeError("Failed to initialize tracker")
                
            tracked_poses = self.tracker.track_video_poses(filtered_poses)
            
            # 트랙 필터링 및 정리
            valid_tracks = self._filter_tracks_by_length(tracked_poses)
            logging.info(f"Found {len(self._get_unique_track_ids(valid_tracks))} valid tracks")
            
            # 3. 어노테이션 데이터 생성
            frame_annotations = self._create_frame_annotations(valid_tracks)
            track_summary = self._create_track_summary(valid_tracks)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 처리 통계
            processing_stats = {
                'processing_time': processing_time,
                'total_frames': len(frame_poses),
                'filtered_frames': len(filtered_poses),
                'valid_frames': len(valid_tracks),
                'total_persons': self._count_total_persons(valid_tracks),
                'unique_tracks': len(self._get_unique_track_ids(valid_tracks)),
                'avg_persons_per_frame': self._calculate_avg_persons_per_frame(valid_tracks)
            }
            
            # 시각화용 데이터 구성 (stage1-2 데이터)
            visualization_data = VisualizationData(
                stage="annotation",
                video_path=video_path,
                video_info=video_info,
                pose_data=filtered_poses,
                tracking_data=valid_tracks,
                track_summary=track_summary,
                frame_annotations=frame_annotations,
                processing_stats=processing_stats
            )
            
            logging.info(f"Annotation processing completed in {processing_time:.2f}s")
            return visualization_data
            
        except Exception as e:
            logging.error(f"Error processing video for annotation: {str(e)}")
            raise
    
    def _extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """비디오 기본 정보 추출"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'filename': Path(video_path).name,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def _filter_poses_by_quality(self, frame_poses: List[FramePoses]) -> List[FramePoses]:
        """품질 기준으로 포즈 필터링"""
        filtered = []
        
        for frame_pose in frame_poses:
            valid_persons = []
            
            for person in frame_pose.persons:
                # 신뢰도 확인
                if person.score >= self.config.min_pose_confidence:
                    valid_persons.append(person)
            
            # 최대 person 수 제한
            if len(valid_persons) > self.config.max_persons_per_frame:
                # 신뢰도 순으로 정렬하여 상위 N개만 선택
                valid_persons.sort(key=lambda x: x.score, reverse=True)
                valid_persons = valid_persons[:self.config.max_persons_per_frame]
            
            if valid_persons:
                filtered_frame = FramePoses(
                    frame_idx=frame_pose.frame_idx,
                    timestamp=frame_pose.timestamp,
                    persons=valid_persons
                )
                filtered.append(filtered_frame)
        
        return filtered
    
    def _filter_tracks_by_length(self, tracked_poses: List[FramePoses]) -> List[FramePoses]:
        """트랙 길이 기준으로 필터링"""
        # 트랙별 프레임 수 계산
        track_counts = {}
        for frame_pose in tracked_poses:
            for person in frame_pose.persons:
                if person.track_id is not None:
                    track_counts[person.track_id] = track_counts.get(person.track_id, 0) + 1
        
        # 유효한 트랙 ID 선별
        valid_track_ids = {
            track_id for track_id, count in track_counts.items()
            if count >= self.config.min_track_length
        }
        
        logging.info(f"Valid tracks: {len(valid_track_ids)} out of {len(track_counts)}")
        
        # 유효한 트랙만 남기기
        filtered = []
        for frame_pose in tracked_poses:
            valid_persons = [
                person for person in frame_pose.persons
                if person.track_id in valid_track_ids
            ]
            
            if valid_persons:
                filtered_frame = FramePoses(
                    frame_idx=frame_pose.frame_idx,
                    timestamp=frame_pose.timestamp,
                    persons=valid_persons
                )
                filtered.append(filtered_frame)
        
        return filtered
    
    def _create_frame_annotations(self, tracked_poses: List[FramePoses]) -> List[Dict[str, Any]]:
        """프레임별 어노테이션 데이터 생성"""
        annotations = []
        
        for frame_pose in tracked_poses:
            frame_data = {
                'frame_idx': frame_pose.frame_idx,
                'timestamp': frame_pose.timestamp,
                'persons': []
            }
            
            for person in frame_pose.persons:
                person_data = {
                    'track_id': person.track_id,
                    'score': person.score
                }
                
                # 바운딩 박스 저장
                if self.config.save_bboxes and person.bbox:
                    person_data['bbox'] = person.bbox
                
                # 키포인트 저장
                if self.config.save_keypoints and person.keypoints:
                    person_data['keypoints'] = person.keypoints.tolist() if hasattr(person.keypoints, 'tolist') else person.keypoints
                
                frame_data['persons'].append(person_data)
            
            annotations.append(frame_data)
        
        return annotations
    
    def _create_track_summary(self, tracked_poses: List[FramePoses]) -> Dict[int, Dict[str, Any]]:
        """트랙별 요약 정보 생성"""
        track_data = {}
        
        # 트랙별 데이터 수집
        for frame_pose in tracked_poses:
            for person in frame_pose.persons:
                track_id = person.track_id
                
                if track_id not in track_data:
                    track_data[track_id] = {
                        'first_frame': frame_pose.frame_idx,
                        'last_frame': frame_pose.frame_idx,
                        'frame_count': 0,
                        'avg_score': 0.0,
                        'scores': [],
                        'bboxes': []
                    }
                
                track_info = track_data[track_id]
                track_info['last_frame'] = frame_pose.frame_idx
                track_info['frame_count'] += 1
                track_info['scores'].append(person.score)
                
                if person.bbox:
                    track_info['bboxes'].append(person.bbox)
        
        # 요약 통계 계산
        summary = {}
        for track_id, data in track_data.items():
            summary[track_id] = {
                'track_id': track_id,
                'first_frame': data['first_frame'],
                'last_frame': data['last_frame'],
                'duration': data['last_frame'] - data['first_frame'] + 1,
                'frame_count': data['frame_count'],
                'avg_score': sum(data['scores']) / len(data['scores']) if data['scores'] else 0.0,
                'min_score': min(data['scores']) if data['scores'] else 0.0,
                'max_score': max(data['scores']) if data['scores'] else 0.0
            }
        
        return summary
    
    def _get_unique_track_ids(self, tracked_poses: List[FramePoses]) -> set:
        """고유한 트랙 ID 집합 반환"""
        track_ids = set()
        for frame_pose in tracked_poses:
            for person in frame_pose.persons:
                if person.track_id is not None:
                    track_ids.add(person.track_id)
        return track_ids
    
    def _count_total_persons(self, tracked_poses: List[FramePoses]) -> int:
        """총 person 인스턴스 수"""
        total = 0
        for frame_pose in tracked_poses:
            total += len(frame_pose.persons)
        return total
    
    def _calculate_avg_persons_per_frame(self, tracked_poses: List[FramePoses]) -> float:
        """프레임당 평균 person 수"""
        if not tracked_poses:
            return 0.0
        
        total_persons = sum(len(frame_pose.persons) for frame_pose in tracked_poses)
        return total_persons / len(tracked_poses)
    
    def save_visualization_data(self, visualization_data: VisualizationData, 
                               output_path: Union[str, Path], 
                               stage_prefix: str = ""):
        """시각화 데이터 저장 (단계별 PKL)
        
        Args:
            visualization_data: 시각화용 데이터
            output_path: 출력 파일 경로
            stage_prefix: 단계 접두사 (stage1_, stage2_, stage3_)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 단계별 파일명 조정
        if stage_prefix:
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.parent / f"{stage_prefix}{stem}{suffix}"
        
        if self.config.output_format == 'json':
            self._save_as_json(visualization_data, output_path.with_suffix('.json'))
        elif self.config.output_format == 'pickle':
            self._save_as_pickle(visualization_data, output_path.with_suffix('.pkl'))
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
    
    def _save_as_json(self, visualization_data: VisualizationData, output_path: Path):
        """JSON 형태로 저장"""
        import json
        
        # VisualizationData를 dict로 변환 (직렬화 가능하게)
        data = {
            'stage': visualization_data.stage,
            'video_path': visualization_data.video_path,
            'video_info': visualization_data.video_info,
            'processing_stats': visualization_data.processing_stats,
            'quality_stats': visualization_data.quality_stats
        }
        
        # 단계별 데이터 추가
        if visualization_data.pose_data:
            data['pose_data'] = self._serialize_frame_poses(visualization_data.pose_data)
        
        if visualization_data.tracking_data:
            data['tracking_data'] = self._serialize_frame_poses(visualization_data.tracking_data)
            
        if visualization_data.track_summary:
            data['track_summary'] = visualization_data.track_summary
            
        if visualization_data.frame_annotations:
            data['frame_annotations'] = visualization_data.frame_annotations
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Visualization data saved as JSON: {output_path}")
    
    def _save_as_pickle(self, visualization_data: VisualizationData, output_path: Path):
        """Pickle 형태로 저장"""
        import pickle
        
        with open(output_path, 'wb') as f:
            pickle.dump(visualization_data, f)
        
        logging.info(f"Visualization data saved as pickle: {output_path}")
    
    def _serialize_frame_poses(self, frame_poses: List[FramePoses]) -> List[Dict]:
        """Frame poses 데이터 JSON 직렬화를 위한 변환"""
        serialized = []
        for frame_pose in frame_poses:
            frame_data = {
                'frame_idx': frame_pose.frame_idx,
                'timestamp': frame_pose.timestamp,
                'persons': []
            }
            
            for person in frame_pose.persons:
                person_data = {
                    'track_id': person.track_id,
                    'score': person.score,
                    'bbox': person.bbox.tolist() if hasattr(person.bbox, 'tolist') else person.bbox,
                    'keypoints': person.keypoints.tolist() if hasattr(person.keypoints, 'tolist') else person.keypoints
                }
                frame_data['persons'].append(person_data)
            
            serialized.append(frame_data)
        
        return serialized
    
    def create_stgcn_compatible_dataset(self, 
                                      stage3_results: List[StageResult],
                                      dataset_name: str = "violence_detection") -> STGCNData:
        """
        STGCN 호환 데이터셋 생성
        
        Args:
            stage3_results: Stage 3 처리 결과 리스트
            dataset_name: 데이터셋 이름
            
        Returns:
            STGCN 호환 데이터 구조
        """
        import random
        import numpy as np
        
        # 성공한 결과만 수집
        successful_results = [r for r in stage3_results if r.success]
        
        all_video_data = []
        label_counts = {'Fight': 0, 'NonFight': 0}
        
        for result in successful_results:
            try:
                # Stage 3 결과 로드
                with open(result.output_path, 'rb') as f:
                    stage3_data = pickle.load(f)
                
                video_name = Path(result.input_path).stem
                
                # 라벨 추출 (metadata에서 또는 파일명에서)
                if 'label' in result.metadata:
                    label_str = result.metadata['label']
                elif 'Fight' in video_name or 'F_' in video_name:
                    label_str = 'Fight'
                elif 'NonFight' in video_name or 'NF_' in video_name:
                    label_str = 'NonFight'
                else:
                    logging.warning(f"Cannot determine label for {video_name}, skipping")
                    continue
                
                label_int = 1 if label_str == 'Fight' else 0
                label_counts[label_str] += 1
                
                # 윈도우 데이터 추출
                windows = stage3_data.get('windows', [])
                
                video_data = {
                    'video_name': video_name,
                    'label': label_int,
                    'label_str': label_str,
                    'windows': windows,
                    'num_windows': len(windows),
                    'total_frames': sum(w.get('num_frames', 0) for w in windows)
                }
                
                all_video_data.append(video_data)
                
            except Exception as e:
                logging.error(f"Error processing {result.output_path}: {str(e)}")
                continue
        
        # 데이터셋 분할 (7:2:1)
        random.shuffle(all_video_data)
        
        train_size = int(len(all_video_data) * self.config.train_ratio)
        val_size = int(len(all_video_data) * self.config.val_ratio)
        
        train_data = all_video_data[:train_size]
        val_data = all_video_data[train_size:train_size + val_size]
        test_data = all_video_data[train_size + val_size:]
        
        # STGCN 호환 데이터 구조 생성
        stgcn_data = STGCNData(
            train_data=self._convert_to_stgcn_format(train_data),
            val_data=self._convert_to_stgcn_format(val_data),
            test_data=self._convert_to_stgcn_format(test_data),
            label_map={'NonFight': 0, 'Fight': 1},
            class_names=['NonFight', 'Fight'],
            dataset_stats={
                'total_videos': len(all_video_data),
                'fight_videos': label_counts['Fight'],
                'nonfight_videos': label_counts['NonFight'],
                'train_videos': len(train_data),
                'val_videos': len(val_data),
                'test_videos': len(test_data)
            },
            num_classes=2,
            keypoint_info={
                'num_keypoints': 17,  # COCO 포맷
                'keypoint_names': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
                'keypoint_format': 'coco'
            },
            skeleton_info=[
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ],
            processing_metadata={
                'dataset_name': dataset_name,
                'window_size': self.config.window_size,
                'window_stride': self.config.window_stride,
                'created_timestamp': time.time()
            }
        )
        
        return stgcn_data
    
    def _convert_to_stgcn_format(self, video_data_list: List[Dict]) -> List[Dict]:
        """비디오 데이터를 STGCN 형식으로 변환"""
        stgcn_format = []
        
        for video_data in video_data_list:
            for window_idx, window in enumerate(video_data['windows']):
                try:
                    # 키포인트 데이터 추출
                    keypoints_sequence = window.get('annotation', {}).get('keypoints', [])
                    
                    if not keypoints_sequence:
                        continue
                    
                    # STGCN 입력 형식: (C, T, V, M)
                    # C: 채널 (x, y, confidence), T: 시간, V: 조인트, M: 사람 수
                    
                    stgcn_item = {
                        'keypoint': keypoints_sequence,
                        'label': video_data['label'],
                        'video_name': video_data['video_name'],
                        'window_idx': window_idx,
                        'num_frames': len(keypoints_sequence),
                        'composite_score': window.get('composite_score', 0.0)
                    }
                    
                    stgcn_format.append(stgcn_item)
                    
                except Exception as e:
                    logging.warning(f"Error converting window {window_idx} from {video_data['video_name']}: {str(e)}")
                    continue
        
        return stgcn_format
    
    def save_stgcn_dataset(self, stgcn_data: STGCNData, output_dir: Union[str, Path]):
        """
        STGCN 데이터셋 저장
        
        Args:
            stgcn_data: STGCN 데이터 구조
            output_dir: 출력 디렉터리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 분할별 저장
        for split_name, split_data in [('train', stgcn_data.train_data), 
                                     ('val', stgcn_data.val_data), 
                                     ('test', stgcn_data.test_data)]:
            if split_data:
                split_file = output_dir / f"{split_name}.pkl"
                with open(split_file, 'wb') as f:
                    pickle.dump(split_data, f)
                logging.info(f"Saved {split_name} split: {len(split_data)} samples to {split_file}")
        
        # 메타데이터 저장
        metadata_file = output_dir / "metadata.pkl"
        metadata = {
            'label_map': stgcn_data.label_map,
            'class_names': stgcn_data.class_names,
            'dataset_stats': stgcn_data.dataset_stats,
            'num_classes': stgcn_data.num_classes,
            'keypoint_info': stgcn_data.keypoint_info,
            'skeleton_info': stgcn_data.skeleton_info,
            'processing_metadata': stgcn_data.processing_metadata
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"STGCN dataset saved to {output_dir}")
        logging.info(f"Train: {len(stgcn_data.train_data)}, Val: {len(stgcn_data.val_data)}, Test: {len(stgcn_data.test_data)} samples")
        
    def run_annotation_mode(self, video_paths: List[Union[str, Path]], 
                           output_dir: Union[str, Path], 
                           labels: Optional[Dict[str, str]] = None) -> List[VisualizationData]:
        """
        어노테이션 모드 실행 (stage1-2만 실행)
        
        Args:
            video_paths: 처리할 비디오 파일 경로 리스트
            output_dir: 출력 디렉터리
            labels: 비디오별 라벨 매핑 (optional)
            
        Returns:
            어노테이션 결과 리스트
        """
        logging.info(f"Running annotation mode for {len(video_paths)} videos")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                label = labels.get(video_name) if labels else None
                
                # 어노테이션 데이터 생성
                annotation_data = self.process_video_for_annotation(video_path, label)
                
                # Stage 1 데이터 저장 (포즈 추정)
                stage1_output = output_dir / "stage1_poses" / f"{video_name}.pkl"
                stage1_vis_data = VisualizationData(
                    stage="stage1",
                    video_path=annotation_data.video_path,
                    video_info=annotation_data.video_info,
                    pose_data=annotation_data.pose_data,
                    processing_stats=annotation_data.processing_stats
                )
                self.save_visualization_data(stage1_vis_data, stage1_output)
                
                # Stage 2 데이터 저장 (트래킹)
                stage2_output = output_dir / "stage2_tracking" / f"{video_name}.pkl"
                stage2_vis_data = VisualizationData(
                    stage="stage2",
                    video_path=annotation_data.video_path,
                    video_info=annotation_data.video_info,
                    pose_data=annotation_data.pose_data,
                    tracking_data=annotation_data.tracking_data,
                    track_summary=annotation_data.track_summary,
                    frame_annotations=annotation_data.frame_annotations,
                    processing_stats=annotation_data.processing_stats
                )
                self.save_visualization_data(stage2_vis_data, stage2_output)
                
                results.append(annotation_data)
                
            except Exception as e:
                logging.error(f"Failed to process {video_path}: {str(e)}")
                continue
        
        logging.info(f"Annotation mode completed: {len(results)} videos processed")
        return results
        
    def run_stage_based_pipeline(self, video_paths: List[Union[str, Path]], 
                                class_labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        단계별 파이프라인 실행 (사용자 요구사항에 맞는 구조)
        
        단계별 PKL 출력:
        1. 포즈 추정 → 포즈 추정 시각화용 PKL
        2. 포즈추정 + 트래킹 → 트래킹까지 시각화용 PKL  
        3. 포즈추정 + 트래킹 + 복합점수 반영한 정렬 → 시각화용 PKL
        4. 마지막 통합 PKL(train, val, test) → STGCN 바로 학습 가능한 구조
        
        Args:
            video_paths: 처리할 비디오 파일 경로 리스트
            class_labels: 비디오별 클래스 라벨 (선택사항)
            
        Returns:
            각 단계별 처리 결과와 최종 STGCN 데이터
        """
        logging.info(f"Starting stage-based pipeline for {len(video_paths)} videos")
        
        # annotation_mode가 활성화된 경우 stage1-2만 실행
        if self.config.annotation_mode:
            return {
                'annotation_results': self.run_annotation_mode(video_paths, self.config.stage1_output_dir, class_labels),
                'mode': 'annotation_only'
            }
        
        # 전체 파이프라인 실행
        results = {}
        
        # Stage 1: 포즈 추정 → 시각화용 PKL
        if "stage1" in self.config.stages_to_run:
            logging.info("=== Stage 1: Pose Estimation ===")
            stage1_results = self.run_stage1(video_paths)
            results['stage1'] = stage1_results
            
            # 시각화용 PKL 생성
            for result in stage1_results:
                if result.success:
                    self._create_stage1_visualization_pkl(result)
        
        # Stage 2: 트래킹 → 시각화용 PKL
        if "stage2" in self.config.stages_to_run:
            logging.info("=== Stage 2: Tracking ===")
            stage1_results = results.get('stage1', [])
            stage2_results = self.run_stage2(stage1_results)
            results['stage2'] = stage2_results
            
            # 시각화용 PKL 생성
            for result in stage2_results:
                if result.success:
                    self._create_stage2_visualization_pkl(result)
        
        # Stage 3: 복합점수 반영 정렬 → 시각화용 PKL
        if "stage3" in self.config.stages_to_run:
            logging.info("=== Stage 3: Scoring & Classification ===")
            stage2_results = results.get('stage2', [])
            stage3_results = self.run_stage3(stage2_results, class_labels)
            results['stage3'] = stage3_results
            
            # 시각화용 PKL 생성
            for result in stage3_results:
                if result.success:
                    self._create_stage3_visualization_pkl(result)
        
        # Stage 4: STGCN 훈련용 통합 PKL 생성
        if "stage4" in self.config.stages_to_run and 'stage3' in results:
            logging.info("=== Stage 4: STGCN Dataset Creation ===")
            stage3_results = results['stage3']
            stgcn_data = self.create_stgcn_compatible_dataset(stage3_results)
            
            # STGCN 데이터셋 저장
            stgcn_output_dir = Path(self.config.stage4_output_dir)
            self.save_stgcn_dataset(stgcn_data, stgcn_output_dir)
            
            results['stage4'] = {
                'stgcn_data': stgcn_data,
                'output_dir': str(stgcn_output_dir)
            }
        
        # 전체 통계 출력
        self._print_pipeline_summary()
        
        return results
    
    def _create_stage1_visualization_pkl(self, stage1_result: StageResult):
        """Stage 1 시각화용 PKL 생성"""
        try:
            # Stage 1 결과 로드
            with open(stage1_result.output_path, 'rb') as f:
                frame_poses = pickle.load(f)
            
            # 비디오 정보 추출
            video_info = self._extract_video_info(stage1_result.input_path)
            
            # 시각화용 데이터 구성
            vis_data = VisualizationData(
                stage="stage1",
                video_path=stage1_result.input_path,
                video_info=video_info,
                pose_data=frame_poses,
                processing_stats=stage1_result.metadata
            )
            
            # 저장
            video_name = Path(stage1_result.input_path).stem
            output_path = Path(self.config.stage1_output_dir) / f"{video_name}_vis.pkl"
            self.save_visualization_data(vis_data, output_path, "stage1_")
            
        except Exception as e:
            logging.error(f"Failed to create stage1 visualization PKL: {str(e)}")
    
    def _create_stage2_visualization_pkl(self, stage2_result: StageResult):
        """Stage 2 시각화용 PKL 생성"""
        try:
            # Stage 2 결과 로드
            with open(stage2_result.output_path, 'rb') as f:
                stage2_data = pickle.load(f)
            
            tracked_poses = stage2_data['tracked_poses']
            
            # 비디오 정보 추출
            video_info = self._extract_video_info(stage2_result.input_path)
            
            # 트랙 요약 생성
            track_summary = self._create_track_summary(tracked_poses)
            
            # 시각화용 데이터 구성
            vis_data = VisualizationData(
                stage="stage2",
                video_path=stage2_result.input_path,
                video_info=video_info,
                tracking_data=tracked_poses,
                track_summary=track_summary,
                processing_stats=stage2_result.metadata
            )
            
            # 저장
            video_name = Path(stage2_result.input_path).stem
            output_path = Path(self.config.stage2_output_dir) / f"{video_name}_vis.pkl"
            self.save_visualization_data(vis_data, output_path, "stage2_")
            
        except Exception as e:
            logging.error(f"Failed to create stage2 visualization PKL: {str(e)}")
    
    def _create_stage3_visualization_pkl(self, stage3_result: StageResult):
        """Stage 3 시각화용 PKL 생성"""
        try:
            # Stage 3 결과 로드
            with open(stage3_result.output_path, 'rb') as f:
                stage3_data = pickle.load(f)
            
            # 비디오 정보 추출
            video_info = self._extract_video_info(stage3_result.input_path)
            
            # 시각화용 데이터 구성
            vis_data = VisualizationData(
                stage="stage3",
                video_path=stage3_result.input_path,
                video_info=video_info,
                scoring_data=stage3_data.get('windows', []),
                ranked_windows=stage3_data.get('ranked_windows', []),
                processing_stats=stage3_result.metadata
            )
            
            # 저장
            video_name = Path(stage3_result.input_path).stem
            output_path = Path(self.config.stage3_output_dir) / f"{video_name}_vis.pkl"
            self.save_visualization_data(vis_data, output_path, "stage3_")
            
        except Exception as e:
            logging.error(f"Failed to create stage3 visualization PKL: {str(e)}")


# 기존 SeparatedPipeline을 통합된 형태로 업데이트
# 사용 시에는 AnnotationIntegratedPipeline을 사용하여 모든 기능 이용 가능