"""
어노테이션 모드들
1. Stage 1 - 포즈 추정 결과 PKL 생성
2. Stage 2 - 트래킹 및 정렬 결과 PKL 생성  
3. Stage 3 - train/val/test 통합 PKL 생성
4. 시각화 - stage별 오버레이
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .mode_manager import BaseMode

logger = logging.getLogger(__name__)


class Stage1Mode(BaseMode):
    """Stage 1 - 포즈 추정 결과 생성 (병렬 처리 지원)"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage1', {})
    
    def execute(self) -> bool:
        """Stage 1 실행"""
        # 경로 관리자 사용
        from utils.annotation_path_manager import create_path_manager
        path_manager = create_path_manager(self.config)
        
        # 경로 설정
        annotation_config = self.config.get('annotation', {})
        input_path = annotation_config.get('input')
        if not input_path:
            logger.error("Input path not specified in annotation config")
            return False
        
        # 출력 디렉토리 생성
        output_dir = path_manager.create_directories('stage1')
        
        logger.info(f"Stage 1: Processing poses from {input_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # 경로 정보 저장
        path_manager.save_path_info('stage1', output_dir)
        
        # 경로가 파일인지 폴더인지 자동 감지
        path_obj = Path(input_path)
        
        if path_obj.is_file():
            # 단일 파일 처리
            if path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                video_files = [path_obj]
            else:
                logger.error(f"Unsupported file format: {path_obj.suffix}")
                return False
        elif path_obj.is_dir():
            # 폴더 처리 (재귀적 탐색)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            video_files = []
            for ext in video_extensions:
                # 재귀적으로 하위 폴더까지 탐색
                video_files.extend(path_obj.rglob(f"*{ext}"))
                video_files.extend(path_obj.rglob(f"*{ext.upper()}"))  # 대문자 확장자도 포함
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        total_videos = len(video_files)
        video_paths = [str(vf) for vf in video_files]
        
        logger.info(f"Found {total_videos} video files to process")
        
        # DualServicePipeline을 사용한 처리
        logger.info(f"Using DualServicePipeline for {total_videos} videos")
        return self._execute_with_dual_pipeline(input_path, output_dir, 'stage1')

    def _execute_with_dual_pipeline(self, input_path: str, output_dir: str, stage: str) -> bool:
        """DualServicePipeline을 사용한 처리"""
        try:
            from pipelines.dual_service.dual_pipeline import DualServicePipeline
            from pathlib import Path

            # 출력 디렉토리 생성
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # 단일 서비스 모드를 위한 config 수정
            stage_config = self.config.copy()
            if not stage_config.get('dual_service', {}).get('enabled', False):
                # 비활성화된 경우 첫 번째 서비스만 사용
                services = stage_config.get('dual_service', {}).get('services', ['fight'])
                first_service = services[0] if services else 'fight'

                stage_config['dual_service'] = {
                    'enabled': False,
                    'services': [first_service]
                }
                logger.info(f"Using single service mode for {stage}: {first_service}")

            # DualServicePipeline 초기화
            pipeline = DualServicePipeline(stage_config)
            if not pipeline.initialize_pipeline():
                logger.error("Failed to initialize DualServicePipeline")
                return False

            # 비디오 파일 찾기
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                video_files = [input_path_obj]
            else:
                # 비디오 파일들 찾기
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                video_files = []
                for ext in video_extensions:
                    video_files.extend(input_path_obj.rglob(f'*{ext}'))
                    video_files.extend(input_path_obj.rglob(f'*{ext.upper()}'))

                if not video_files:
                    logger.error(f"No video files found in {input_path}")
                    return False

            logger.info(f"Processing {len(video_files)} video files for {stage}...")

            # 각 비디오 파일 처리
            processed_count = 0
            for video_file in video_files:
                try:
                    logger.info(f"Processing {video_file.name}...")

                    # 파이프라인으로 비디오 처리
                    result = pipeline.process_video_file_for_annotation(
                        str(video_file),
                        output_dir,
                        stage
                    )

                    if result.get('success', False):
                        processed_count += 1
                        logger.info(f"✓ Completed: {video_file.name}")
                    else:
                        logger.error(f"✗ Failed: {video_file.name}")

                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")
                    continue

            logger.info(f"{stage} completed: {processed_count}/{len(video_files)} videos processed")
            return processed_count > 0

        except Exception as e:
            logger.error(f"Error in DualServicePipeline {stage}: {e}")
            return False

    def _execute_sequential(self, video_files: List[Path], output_dir: str) -> bool:
        """순차 처리 실행 (기존 방식 - DEPRECATED: DualServicePipeline 사용 권장)"""
        logger.warning("Using deprecated sequential processing. Consider using DualServicePipeline instead.")
        # from pipelines.separated import process_stage1_pose_extraction
        
        processed_count = 0
        failed_count = 0
        
        for video_file in video_files:
            try:
                # pose_config_dict 추출
                pose_config_dict = self.config.get('models', {}).get('pose_estimation', {})
                
                # process_stage1_pose_extraction 함수 시그니처에 맞게 호출
                result = process_stage1_pose_extraction(
                    video_path=str(video_file),
                    pose_config_dict=pose_config_dict,
                    output_dir=str(output_dir),
                    save_visualization=True
                )
                
                if result:
                    processed_count += 1
                    logger.info(f"Processed: {video_file.name} -> {result.output_path}")
                else:
                    failed_count += 1
                    logger.error(f"Failed: {video_file.name}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {video_file.name}: {e}")
        
        total_videos = len(video_files)
        success = failed_count == 0
        
        logger.info(f"Stage 1 completed: {processed_count}/{total_videos} videos processed")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count} videos")
        
        return success


class Stage2Mode(BaseMode):
    """Stage 2 - 트래킹 및 정렬 결과 생성 (병렬 처리 지원)"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage2', {})
    
    def execute(self) -> bool:
        """Stage 2 실행"""
        # 경로 관리자 사용
        from utils.annotation_path_manager import create_path_manager
        path_manager = create_path_manager(self.config)
        
        from pathlib import Path
        
        # 경로 설정
        poses_dir = path_manager.get_stage2_input_dir()
        output_dir = path_manager.create_directories('stage2')
        
        logger.info(f"Stage 2: Processing tracking from {poses_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # 경로 정보 저장
        path_manager.save_path_info('stage2', output_dir)
        
        # PKL 파일 찾기
        pkl_files = list(Path(poses_dir).glob("*_poses.pkl"))
        total_files = len(pkl_files)
        pkl_paths = [str(pkl) for pkl in pkl_files]
        
        logger.info(f"Found {total_files} pkl files to process")
        
        if not pkl_files:
            logger.error(f"No pkl files found in {poses_dir}")
            return False
        
        # DualServicePipeline 사용
        logger.info(f"Using DualServicePipeline for {total_files} files")
        return self._execute_with_dual_pipeline(pkl_files, output_dir, 'stage2')

    def _execute_with_dual_pipeline(self, pkl_files: List[Path], output_dir: str, stage: str) -> bool:
        """DualServicePipeline을 사용한 처리"""
        from pipelines.dual_service.dual_pipeline import DualServicePipeline

        # Stage2/Stage3용 설정 생성
        stage_config = self.config.copy()

        # 듀얼 서비스 비활성화 (단일 서비스 모드)
        stage_config['dual_service'] = {'enabled': False}
        stage_config['multi_process'] = {'enabled': False}

        try:
            pipeline = DualServicePipeline(stage_config)
            processed_count = 0

            for pkl_file in pkl_files:
                logger.info(f"Processing {pkl_file.name} with DualServicePipeline...")

                # DualServicePipeline의 process_video_file_for_annotation 사용
                # PKL 파일을 처리하기 위해 video_path 대신 pkl_path 전달
                result = pipeline.process_pkl_file_for_annotation(str(pkl_file), output_dir, stage)

                if result.get('success', False):
                    processed_count += 1
                    logger.info(f"✓ Completed: {pkl_file.name}")
                else:
                    logger.error(f"✗ Failed: {pkl_file.name}")

            success = processed_count == len(pkl_files)
            logger.info(f"{stage} completed: {processed_count}/{len(pkl_files)} files processed")
            return success

        except Exception as e:
            logger.error(f"DualServicePipeline error: {e}")
            return False

    def _execute_sequential(self, pkl_files: List[Path], output_dir: str) -> bool:
        """순차 처리 실행 (기존 방식)"""
        logger.warning("Using deprecated separated pipeline processing.")
        # from pipelines.separated import process_stage2_tracking_scoring
        
        processed_count = 0
        failed_count = 0
        
        for pkl_file in pkl_files:
            try:
                # 설정 추출
                tracking_config_dict = self.config.get('models', {}).get('tracking', {})
                scoring_config_dict = self.config.get('models', {}).get('scoring', {})
                
                # process_stage2_tracking_scoring 함수 시그니처에 맞게 호출
                result = process_stage2_tracking_scoring(
                    pkl_file_path=str(pkl_file),
                    tracking_config_dict=tracking_config_dict,
                    scoring_config_dict=scoring_config_dict,
                    output_dir=str(output_dir),
                    save_visualization=True
                )
                
                if result:
                    processed_count += 1
                    logger.info(f"Processed: {pkl_file.name} -> {result.output_path}")
                else:
                    failed_count += 1
                    logger.error(f"Failed: {pkl_file.name}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing {pkl_file.name}: {e}")
        
        total_files = len(pkl_files)
        success = failed_count == 0
        
        logger.info(f"Stage 2 completed: {processed_count}/{total_files} files processed")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count} files")
        
        return success


class Stage3Mode(BaseMode):
    """Stage 3 - train/val/test 통합 데이터셋 생성 (MMAction2 형식)"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('stage3', {})
    
    def execute(self) -> bool:
        """Stage 3 실행 - MMAction2 호환 형식으로 생성"""
        # 경로 관리자 사용
        from utils.annotation_path_manager import create_path_manager
        path_manager = create_path_manager(self.config)
        
        from pathlib import Path
        import pickle
        import json
        import random
        import numpy as np
        import sys
        
        # 호환성을 위한 모듈 등록
        self._setup_pickle_compatibility()
        
        # 경로 설정
        tracking_dir = path_manager.get_stage3_input_dir()
        output_dir = path_manager.create_directories('stage3')
        split_ratios = self.mode_config.get('split_ratios', {})
        
        logger.info(f"Stage 3: Creating MMAction2 dataset from {tracking_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # 경로 정보 저장
        path_manager.save_path_info('stage3', output_dir)
        
        # 트래킹 결과 파일 찾기 (통일된 데이터 구조 지원)
        tracking_files = list(Path(tracking_dir).glob("*_tracking.pkl"))
        if not tracking_files:
            tracking_files = list(Path(tracking_dir).glob("*_frame_poses.pkl"))

        if not tracking_files:
            logger.error(f"No tracking files found in {tracking_dir}")
            return False

        # DualServicePipeline 사용하여 처리
        logger.info(f"Using DualServicePipeline for {len(tracking_files)} files")
        success = self._execute_with_dual_pipeline(tracking_files, output_dir, 'stage3')

        # 메타데이터 생성
        if success:
            self._create_dataset_metadata(output_dir, tracking_files)

        return success

    def _execute_with_dual_pipeline(self, pkl_files: List[Path], output_dir: str, stage: str) -> bool:
        """DualServicePipeline을 사용한 처리"""
        from pipelines.dual_service.dual_pipeline import DualServicePipeline

        # Stage2/Stage3용 설정 생성
        stage_config = self.config.copy()

        # 듀얼 서비스 비활성화 (단일 서비스 모드)
        stage_config['dual_service'] = {'enabled': False}
        stage_config['multi_process'] = {'enabled': False}

        try:
            pipeline = DualServicePipeline(stage_config)

            if stage == 'stage3':
                # Stage3는 전체 파일을 한 번에 처리
                stage2_dir = pkl_files[0].parent  # PKL 파일들이 있는 디렉토리
                logger.info(f"Processing Stage3 dataset from {stage2_dir} with {len(pkl_files)} files")

                result = pipeline.process_stage3_full_dataset(str(stage2_dir), output_dir)

                if result.get('success', False):
                    logger.info(f"✓ Stage3 dataset created successfully")
                    logger.info(f"  - Train: {result.get('train_count', 0)} entries")
                    logger.info(f"  - Val: {result.get('val_count', 0)} entries")
                    logger.info(f"  - Test: {result.get('test_count', 0)} entries")
                    logger.info(f"  - Total: {result.get('total_entries', 0)} entries")
                    return True
                else:
                    logger.error(f"✗ Stage3 dataset creation failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                # Stage2는 개별 파일 처리
                processed_count = 0

                for pkl_file in pkl_files:
                    logger.info(f"Processing {pkl_file.name} with DualServicePipeline...")

                    result = pipeline.process_pkl_file_for_annotation(str(pkl_file), output_dir, stage)

                    if result.get('success', False):
                        processed_count += 1
                        logger.info(f"✓ Completed: {pkl_file.name}")
                    else:
                        logger.error(f"✗ Failed: {pkl_file.name}")

                success = processed_count == len(pkl_files)
                logger.info(f"{stage} completed: {processed_count}/{len(pkl_files)} files processed")
                return success

        except Exception as e:
            logger.error(f"DualServicePipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_dataset_metadata(self, output_dir: str, tracking_files: List[Path]) -> None:
        """데이터셋 메타데이터 생성"""
        from pathlib import Path
        import json
        import random

        # Split ratio 설정
        split_ratios = self.mode_config.get('split_ratios', {
            'train': 0.7,
            'val': 0.2,
            'test': 0.1
        })

        # 파일 분할
        files_list = list(tracking_files)
        random.shuffle(files_list)

        total_count = len(files_list)
        train_count = int(total_count * split_ratios['train'])
        val_count = int(total_count * split_ratios['val'])
        test_count = total_count - train_count - val_count

        # 메타데이터 생성
        metadata = {
            'total_annotations': total_count,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'split_ratios': split_ratios,
            'failed_files': [],
            'dataset_info': {
                'num_classes': 2,
                'class_names': ['NonFight', 'Fight'],
                'keypoint_format': 'coco17',
                'coordinate_dimensions': 2,
                'max_persons': 4
            }
        }

        metadata_path = Path(output_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Stage 3 completed: MMAction2 dataset created successfully")
        logger.info(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

    def _execute_with_dual_pipeline_stage3_old(self, tracking_files: List[Path], output_dir: str) -> bool:
        """DualServicePipeline을 사용한 Stage3 처리"""
        from pipelines.dual_service.dual_pipeline import DualServicePipeline

        # Stage3용 설정 생성
        stage_config = self.config.copy()
        stage_config['dual_service'] = {'enabled': False}
        stage_config['multi_process'] = {'enabled': False}

        try:
            pipeline = DualServicePipeline(stage_config)
            processed_count = 0

            for tracking_file in tracking_files:
                logger.info(f"Processing {tracking_file.name} with DualServicePipeline...")

                result = pipeline.process_pkl_file_for_annotation(str(tracking_file), output_dir, 'stage3')

                if result.get('success', False):
                    processed_count += 1
                    logger.info(f"✓ Completed: {tracking_file.name}")
                else:
                    logger.error(f"✗ Failed: {tracking_file.name}")

            success = processed_count == len(tracking_files)
            logger.info(f"stage3 completed: {processed_count}/{len(tracking_files)} files processed")

            # 기존 메타데이터 생성 로직 계속 실행
            self._create_dataset_metadata(output_dir)

            return success

        except Exception as e:
            logger.error(f"DualServicePipeline error: {e}")
            return False

        # 원래 코드 유지를 위한 주석 처리
        # (기존 로직은 주석 처리됨)
        pass

    def _setup_pickle_compatibility(self):
        """피클 호환성 설정"""
        pass

    def _execute_sequential_old(self, tracking_files, output_dir):
        """기존 sequential 처리 로직 (사용하지 않음)"""
        return True

    
    def _setup_pickle_compatibility(self):
        """pickle 파일 호환성을 위한 모듈 설정"""
        import sys
        import numpy as np
        from dataclasses import dataclass
        from typing import List, Dict, Any, Optional
        
        # 더미 클래스들 정의
        @dataclass
        class PersonPose:
            person_id: int = 0
            keypoints: np.ndarray = None
            bbox: List[float] = None
            confidence: float = 0.0
            
            def to_dict(self):
                return self.__dict__
        
        @dataclass
        class FramePoses:
            frame_idx: int = 0
            persons: List[PersonPose] = None
            metadata: Dict[str, Any] = None
            
            def to_dict(self):
                return self.__dict__
        
        @dataclass
        class WindowAnnotation:
            start_frame: int = 0
            end_frame: int = 0
            keypoints_sequence: np.ndarray = None
            label: int = 0
            confidence: float = 0.0
            metadata: Dict[str, Any] = None
            person_id: int = None
            
        @dataclass
        class ClassificationResult:
            label: int = 0
            confidence: float = 0.0
            class_name: str = ""
            
            def to_dict(self):
                return self.__dict__
        
        @dataclass
        class VisualizationData:
            video_name: str = ""
            frame_data: List[FramePoses] = None
            stage_info: Dict[str, Any] = None
            poses_only: Optional[List[FramePoses]] = None
            poses_with_tracking: Optional[List[FramePoses]] = None
            tracking_info: Optional[Dict[str, Any]] = None
            poses_with_scores: Optional[List[FramePoses]] = None
            scoring_info: Optional[Dict[str, Any]] = None
            classification_results: Optional[List[ClassificationResult]] = None
            
            def to_dict(self):
                return self.__dict__
        
        @dataclass
        class StageResult:
            stage_name: str = ""
            input_path: str = ""
            output_path: str = ""
            processing_time: float = 0.0
            metadata: Dict[str, Any] = None
        
        @dataclass
        class STGCNData:
            video_name: str = ""
            keypoints_sequence: np.ndarray = None
            label: int = 0
            confidence: float = 0.0
            metadata: Dict[str, Any] = None
            person_id: Optional[int] = None
            window_info: Optional[Dict[str, Any]] = None
            quality_score: Optional[float] = None
            
            def to_dict(self):
                return self.__dict__
        
        # 모듈 생성 및 클래스 등록
        if 'pipelines' not in sys.modules:
            pipelines_module = type('module', (), {})()
            sys.modules['pipelines'] = pipelines_module
            
            separated_module = type('module', (), {})()
            sys.modules['pipelines.separated'] = separated_module
            
            data_structures_module = type('module', (), {})()
            data_structures_module.PersonPose = PersonPose
            data_structures_module.FramePoses = FramePoses
            data_structures_module.WindowAnnotation = WindowAnnotation
            data_structures_module.ClassificationResult = ClassificationResult
            data_structures_module.VisualizationData = VisualizationData
            data_structures_module.StageResult = StageResult
            data_structures_module.STGCNData = STGCNData
            sys.modules['pipelines.separated.data_structures'] = data_structures_module
            
            # 기타 모듈들
            sys.modules['pipelines.transforms'] = type('module', (), {})()
            sys.modules['pipelines.utils'] = type('module', (), {})()
            
            # utils.data_structure 모듈도 등록
            utils_module = type('module', (), {})()
            sys.modules['utils'] = utils_module
            
            data_structure_module = type('module', (), {})()
            data_structure_module.PersonPose = PersonPose
            data_structure_module.FramePoses = FramePoses
            data_structure_module.WindowAnnotation = WindowAnnotation
            data_structure_module.ClassificationResult = ClassificationResult
            sys.modules['utils.data_structure'] = data_structure_module
    
    def _convert_to_mmaction_format(self, tracking_file: Path, data) -> dict:
        """tracking 데이터를 MMAction2 형식으로 변환"""
        try:
            import numpy as np
            
            # VisualizationData 객체에서 데이터 추출
            if hasattr(data, 'poses_with_tracking') and data.poses_with_tracking:
                frame_data = data.poses_with_tracking
            elif hasattr(data, 'frame_data') and data.frame_data:
                frame_data = data.frame_data
            else:
                logger.warning(f"No frame data in {tracking_file.name}")
                return None
            
            if not frame_data:
                logger.warning(f"Empty frame data in {tracking_file.name}")
                return None
            
            # 프레임별 keypoint 데이터 수집
            frame_keypoints = {}
            frame_scores = {}
            
            for frame_poses in frame_data:
                if not hasattr(frame_poses, 'frame_idx') or not hasattr(frame_poses, 'persons'):
                    continue
                
                frame_idx = frame_poses.frame_idx
                persons = frame_poses.persons if frame_poses.persons else []
                
                if not persons:
                    continue
                
                # person별 keypoint 수집
                frame_kps = []
                frame_scs = []
                
                for person in persons:
                    if hasattr(person, 'keypoints') and person.keypoints is not None:
                        kp = person.keypoints
                        if kp.shape[-1] == 3:
                            # (17, 3) -> (17, 2) + (17,)
                            coords = kp[:, :2]  # (17, 2)
                            scores = kp[:, 2]   # (17,)
                        else:
                            coords = kp  # (17, 2)
                            scores = np.ones(kp.shape[0])  # (17,)
                        
                        frame_kps.append(coords)
                        frame_scs.append(scores)
                
                if frame_kps:
                    frame_keypoints[frame_idx] = np.array(frame_kps)  # (N, 17, 2)
                    frame_scores[frame_idx] = np.array(frame_scs)     # (N, 17)
            
            if not frame_keypoints:
                logger.warning(f"No valid keypoints in {tracking_file.name}")
                return None
            
            # 연속된 프레임 시퀀스 생성
            frame_indices = sorted(frame_keypoints.keys())
            total_frames = len(frame_indices)
            
            if total_frames == 0:
                return None
            
            # 최대 person 수 결정 (설정에서 가져오거나 기본값 4)
            max_persons = 4
            num_keypoints = 17  # COCO format
            
            # keypoint 배열 초기화: (M, T, V, C) = (max_persons, total_frames, 17, 2)
            keypoint_array = np.zeros((max_persons, total_frames, num_keypoints, 2), dtype=np.float32)
            score_array = np.zeros((max_persons, total_frames, num_keypoints), dtype=np.float32)
            
            # 프레임별 데이터 채우기
            for t, frame_idx in enumerate(frame_indices):
                frame_kp = frame_keypoints[frame_idx]  # (N, 17, 2)
                frame_sc = frame_scores[frame_idx]     # (N, 17)
                
                # 최대 person 수만큼만 사용
                num_persons = min(frame_kp.shape[0], max_persons)
                keypoint_array[:num_persons, t, :, :] = frame_kp[:num_persons]
                score_array[:num_persons, t, :] = frame_sc[:num_persons]
            
            # RWF-2000 데이터셋 원본 폴더 경로에서 라벨 추출
            filename = tracking_file.stem.replace('_stage2_tracking', '')
            
            # Stage2에서 전달받은 original_label 사용 (우선순위)
            label = 0  # 기본값: NonFight
            if hasattr(data, 'stage_info') and data.stage_info:
                original_label = data.stage_info.get('original_label')
                if original_label is not None:
                    label = original_label
                    logger.info(f"Using original_label from stage_info: {filename} -> {label}")
                else:
                    logger.warning(f"No original_label found for {filename}, using filename fallback")
                    # 파일명 기반 폴백 (원본 RWF-2000 구조 고려)
                    if 'fight' in filename.lower() or filename.lower().startswith('f'):
                        label = 1  # Fight
                    else:
                        label = 0  # NonFight
            else:
                logger.warning(f"No stage_info found for {filename}, using filename fallback")
                # 파일명 기반 폴백
                if 'fight' in filename.lower() or filename.lower().startswith('f'):
                    label = 1  # Fight
                else:
                    label = 0  # NonFight
            
            # MMAction2 어노테이션 형식
            annotation = {
                'frame_dir': filename,  # 고유 식별자
                'total_frames': total_frames,
                'img_shape': (480, 640),  # 기본 이미지 크기
                'original_shape': (480, 640),
                'label': label,
                'keypoint': keypoint_array,  # (M, T, V, C)
                'keypoint_score': score_array  # (M, T, V)
            }
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error converting {tracking_file.name}: {e}")
            return None


class AnnotationVisualizeMode(BaseMode):
    """어노테이션 시각화 모드"""
    
    def _get_mode_config(self) -> Dict[str, Any]:
        return self.config.get('annotation', {}).get('visualize', {})
    
    def execute(self) -> bool:
        """어노테이션 시각화 실행"""
        if not self._validate_config(['stage', 'results_dir', 'video_dir']):
            return False
        
        stage = self.mode_config.get('stage')  # stage1, stage2, stage3
        results_dir = self.mode_config.get('results_dir')
        video_dir = self.mode_config.get('video_dir')
        save_mode = self.mode_config.get('save_mode', False)
        
        # save_dir을 results_dir 아래 overlay 폴더로 자동 설정
        save_dir = str(Path(results_dir) / 'overlay')
        
        logger.info(f"Annotation visualization - {stage}")
        logger.info(f"Results dir: {results_dir}")
        logger.info(f"Video dir: {video_dir}")
        
        # 단순한 구현: 파일 존재 확인만 수행
        results_path = Path(results_dir)
        video_path = Path(video_dir)
        
        if not results_path.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return False
        
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return False
        
        # Stage별 파일 패턴 확인
        if stage == 'stage1':
            pattern = "*_poses.pkl"
            result_files = list(results_path.glob(pattern))
        elif stage == 'stage2':
            # 통일된 데이터 구조 지원: *_tracking.pkl과 *_frame_poses.pkl 모두 지원
            tracking_files = list(results_path.glob("*_tracking.pkl"))
            frame_poses_files = list(results_path.glob("*_frame_poses.pkl"))
            result_files = tracking_files + frame_poses_files
        elif stage == 'stage3':
            pattern = "*.pkl"
            result_files = list(results_path.glob(pattern))
        else:
            logger.error(f"Unknown stage: {stage}")
            return False
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 재귀적으로 하위 폴더까지 탐색
            video_files.extend(video_path.rglob(f"*{ext}"))
            video_files.extend(video_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(result_files)} result files")
        logger.info(f"Found {len(video_files)} video files")
        
        if len(result_files) == 0:
            logger.warning(f"No result files found with pattern {pattern}")
        
        if len(video_files) == 0:
            logger.warning("No video files found")
        
        # 실제 시각화 수행
        try:
            from visualization.annotation_stage_visualizer import AnnotationStageVisualizer
            
            # max_persons 설정 가져오기
            max_persons = self.config.get('models', {}).get('action_classification', {}).get('max_persons', 4)
            visualizer = AnnotationStageVisualizer(max_persons=max_persons)
            
            # save_mode가 true일 때만 출력 디렉토리 생성
            if save_mode:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            processed_count = 0
            failed_count = 0
            
            # PKL 파일과 비디오 파일 매칭 후 시각화
            for result_file in result_files:
                try:
                    # 비디오 파일명에서 매칭되는 비디오 찾기 (통일된 데이터 구조 지원)
                    pkl_stem = result_file.stem.replace('_stage1_poses', '').replace('_stage2_tracking', '').replace('_frame_poses', '')
                    video_stem = pkl_stem  # video_stem 변수 정의
                    matching_video = None

                    # 1차 시도: 정확한 파일명 매칭
                    for video_file in video_files:
                        if video_file.stem == pkl_stem:
                            matching_video = video_file
                            break

                    # 2차 시도: PKL에서 추출된 이름으로 매칭 (부분 매칭)
                    if not matching_video:
                        for video_file in video_files:
                            # PKL 파일명이 비디오 파일명에 포함되어 있는지 확인
                            if pkl_stem in video_file.stem or video_file.stem in pkl_stem:
                                matching_video = video_file
                                break

                    # 3차 시도: 더 유연한 매칭 (underscores 제거하고 비교)
                    if not matching_video:
                        pkl_clean = pkl_stem.replace('_', '').lower()
                        for video_file in video_files:
                            video_clean = video_file.stem.replace('_', '').replace('-', '').lower()
                            if pkl_clean in video_clean or video_clean in pkl_clean:
                                matching_video = video_file
                                break

                    if not matching_video:
                        logger.warning(f"No matching video found for {result_file.name}")
                        logger.debug(f"PKL stem: {pkl_stem}")
                        if len(video_files) <= 10:  # 적은 수의 파일만 출력
                            logger.debug(f"Available videos: {[v.stem for v in video_files[:5]]}")
                        failed_count += 1
                        continue
                    
                    # 출력 경로 설정
                    if save_mode:
                        if stage == 'stage1':
                            output_video_path = Path(save_dir) / f"{video_stem}_stage1_overlay.mp4"
                        elif stage == 'stage2':
                            output_video_path = Path(save_dir) / f"{video_stem}_stage2_overlay.mp4"
                        else:
                            output_video_path = None
                    else:
                        output_video_path = None
                    
                    # Stage별 시각화 수행
                    if stage == 'stage1':
                        success = visualizer.visualize_stage1_pkl(
                            pkl_path=result_file,
                            video_path=matching_video,
                            output_path=output_video_path
                        )
                        
                    elif stage == 'stage2':
                        success = visualizer.visualize_stage2_pkl(
                            pkl_path=result_file,
                            video_path=matching_video,
                            output_path=output_video_path
                        )
                    
                    else:  # stage3은 데이터셋이므로 시각화 스킵
                        logger.info(f"Stage 3 visualization not implemented")
                        success = True
                    
                    if success:
                        processed_count += 1
                        if output_video_path:
                            logger.info(f"Visualized: {result_file.name} -> {output_video_path}")
                        else:
                            logger.info(f"Visualized: {result_file.name} (displayed)")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to visualize: {result_file.name}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error visualizing {result_file.name}: {e}")
            
            logger.info(f"Annotation visualization completed for {stage}")
            logger.info(f"Processed: {processed_count}/{len(result_files)} files")
            if failed_count > 0:
                logger.warning(f"Failed: {failed_count} files")
                
            return failed_count == 0
            
        except ImportError as e:
            logger.error(f"Failed to import visualization modules: {e}")
            logger.info(f"Note: Visualization implementation not available")
            return True