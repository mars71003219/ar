"""
RTMO 포즈 추정기 구현

기존 rtmo_gcn_pipeline의 EnhancedRTMOPoseExtractor를 
새로운 표준 인터페이스에 맞게 재구성한 버전입니다.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import logging

try:
    from mmpose.apis import inference_bottomup, init_model
    import torch
    MMPOSE_AVAILABLE = True
except ImportError as e:
    MMPOSE_AVAILABLE = False
    logging.warning(f"MMPose not available: {e}")

try:
    from pose_estimation.base import BasePoseEstimator
    from utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig
except ImportError:
    from ..base import BasePoseEstimator
    from ...utils.data_structure import PersonPose, FramePoses, PoseEstimationConfig


class RTMOPoseEstimator(BasePoseEstimator):
    """RTMO 포즈 추정기 구현"""
    
    def __init__(self, config: PoseEstimationConfig):
        """
        Args:
            config: RTMO 포즈 추정 설정
        """
        super().__init__(config)
        
        if not MMPOSE_AVAILABLE:
            raise RuntimeError("MMPose is required for RTMO pose estimation")
        
        # RTMO 특화 설정
        self.config_file = config.config_file
        self.checkpoint_file = config.model_path  # config.checkpoint_file 대신 model_path 사용
        
        # Enhanced RTMO Extractor 인스턴스
        self.enhanced_extractor = None
        
        # 통계
        self.stats = {
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'avg_persons_per_frame': 0.0,
            'processing_times': []
        }
        
        # 자동 초기화
        self.initialize_model()
    
    def initialize_model(self) -> bool:
        """RTMO 모델 초기화"""
        try:
            if self.is_initialized:
                return True
                
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            if not os.path.exists(self.checkpoint_file):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_file}")
            
            # Enhanced RTMO Extractor 사용
            try:
                from .enhanced_rtmo_extractor import EnhancedRTMOPoseExtractor
                self.enhanced_extractor = EnhancedRTMOPoseExtractor(
                    config_file=self.config_file,
                    checkpoint_file=self.checkpoint_file,
                    device=self.device,
                    score_thr=self.score_threshold,
                    quality_threshold=self.score_threshold  # 품질 임계값도 동일하게 설정
                )
                print("Using Enhanced RTMO extractor")
            except ImportError:
                # Enhanced extractor가 없으면 기본 방식 사용
                print(f"Initializing RTMO model: {self.config_file}")
                self.model = init_model(
                    self.config_file, 
                    self.checkpoint_file, 
                    device=self.device
                )
                self._configure_model()
                print("Using standard RTMO initialization")
            
            self.is_initialized = True
            print("RTMO model initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize RTMO model: {str(e)}")
            self.is_initialized = False
            return False
    
    def _configure_model(self):
        """모델 설정 적용"""
        if hasattr(self.model.cfg, 'model'):
            if hasattr(self.model.cfg.model, 'test_cfg'):
                self.model.cfg.model.test_cfg.score_thr = self.score_threshold
                self.model.cfg.model.test_cfg.nms_thr = self.nms_threshold
            else:
                self.model.cfg.model.test_cfg = dict(
                    score_thr=self.score_threshold, 
                    nms_thr=self.nms_threshold
                )
        
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'test_cfg'):
            self.model.head.test_cfg.score_thr = self.score_threshold
            self.model.head.test_cfg.nms_thr = self.nms_threshold
    
    def extract_poses(self, frame: np.ndarray, frame_idx: int = 0) -> List[PersonPose]:
        """단일 프레임에서 포즈 추출"""
        # 자동 초기화
        if not self.ensure_initialized():
            raise RuntimeError("Failed to initialize RTMO model")
        
        if not self.validate_frame(frame):
            return []
        
        try:
            import time
            start_time = time.time()
            
            persons = []
            
            if self.enhanced_extractor:
                # Enhanced extractor 사용
                pose_result = self.enhanced_extractor.extract_single_frame_poses(frame)
                logging.info(f"RTMO pose_result type: {type(pose_result)}")
                if pose_result:
                    if hasattr(pose_result, 'pred_instances'):
                        instances = pose_result.pred_instances
                        logging.info(f"pred_instances found with {len(instances.keypoints) if hasattr(instances, 'keypoints') else 0} detections")
                        if hasattr(instances, 'keypoints'):
                            logging.info(f"keypoints shape: {instances.keypoints.shape}")
                        if hasattr(instances, 'bboxes'):
                            logging.info(f"bboxes shape: {instances.bboxes.shape}")
                    
                    converted_poses = self.enhanced_extractor._convert_pose_data_sample(pose_result)
                    logging.info(f"Converted poses: {len(converted_poses)} persons")
                    for i, pose in enumerate(converted_poses):
                        if 'keypoints' in pose:
                            kpt = pose['keypoints']
                            logging.info(f"Person {i} keypoints shape: {kpt.shape}, non-zero: {np.count_nonzero(kpt)}")
                    persons = self._convert_enhanced_poses_to_persons(converted_poses, frame_idx)
                else:
                    logging.warning("RTMO returned empty pose_result")
            else:
                # 기본 방식 사용
                pose_results = inference_bottomup(self.model, frame)
                persons = self._convert_mmpose_results_to_persons(pose_results, frame_idx)
            
            # 품질 필터링
            persons = self.filter_low_quality_poses(persons)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            self.stats['total_persons_detected'] += len(persons)
            
            return persons
            
        except Exception as e:
            logging.error(f"Error in pose extraction for frame {frame_idx}: {str(e)}")
            return []
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FramePoses:
        """단일 프레임 처리 - 파이프라인 인터페이스용"""
        persons = self.extract_poses(frame, frame_idx)
        
        # FramePoses 객체 생성
        frame_poses = FramePoses(
            frame_idx=frame_idx,
            persons=persons,
            timestamp=frame_idx / 30.0,  # 기본 30 FPS 가정
            image_shape=(frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)
        )
        
        return frame_poses
    
    def extract_video_poses(self, video_path: str) -> List[FramePoses]:
        """전체 비디오에서 포즈 추출"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Extracting poses from: {video_path}")
        
        # CUDA 메모리 정리
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_poses_list = []
        frame_idx = 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 포즈 추출
                persons = self.extract_poses(frame, frame_idx)
                
                # FramePoses 생성
                timestamp = frame_idx / fps if fps > 0 else frame_idx
                frame_poses = FramePoses(
                    frame_idx=frame_idx,
                    persons=persons,
                    timestamp=timestamp,
                    image_shape=(height, width)
                )
                
                frame_poses_list.append(frame_poses)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 통계 업데이트
        self.stats['total_frames_processed'] += len(frame_poses_list)
        if len(frame_poses_list) > 0:
            total_persons = sum(len(fp.persons) for fp in frame_poses_list)
            self.stats['avg_persons_per_frame'] = total_persons / len(frame_poses_list)
        
        print(f"Extracted poses from {len(frame_poses_list)} frames")
        print(f"Average persons per frame: {self.stats['avg_persons_per_frame']:.2f}")
        
        return frame_poses_list
    
    def _convert_mmpose_results_to_persons(self, pose_results: List[Dict], frame_idx: int) -> List[PersonPose]:
        """MMPose 결과를 PersonPose 형식으로 변환"""
        persons = []
        
        if not pose_results:
            return persons
        
        for person_idx, pose_result in enumerate(pose_results):
            try:
                # MMPose 결과에서 정보 추출
                if 'keypoints' in pose_result:
                    keypoints = np.array(pose_result['keypoints'])  # [17, 3]
                else:
                    continue
                
                # 바운딩 박스 계산 (키포인트에서 추정)
                try:
                    valid_keypoints = keypoints[keypoints[:, 2] > 0.3]  # 신뢰도 > 0.3
                    if len(valid_keypoints) == 0:
                        continue
                    
                    # 방어적 프로그래밍: 배열 형태 확인
                    if valid_keypoints.shape[0] == 0 or valid_keypoints.shape[1] < 2:
                        continue
                        
                    x_coords = valid_keypoints[:, 0]
                    y_coords = valid_keypoints[:, 1]
                    
                    # 좌표가 유효한지 확인
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        continue
                        
                    x1, y1 = np.min(x_coords), np.min(y_coords)
                    x2, y2 = np.max(x_coords), np.max(y_coords)
                except (IndexError, ValueError) as e:
                    logging.warning(f"Error processing keypoints for person {person_idx}: {str(e)}")
                    continue
                
                # 바운딩 박스 확장 (여유분 추가)
                padding = 0.1
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - w * padding)
                y1 = max(0, y1 - h * padding)
                x2 = x2 + w * padding
                y2 = y2 + h * padding
                
                bbox = [x1, y1, x2, y2]
                
                # 전체 신뢰도 점수 계산
                try:
                    valid_score_indices = keypoints[:, 2] > 0
                    if np.any(valid_score_indices):
                        valid_scores = keypoints[valid_score_indices, 2]
                        overall_score = np.mean(valid_scores) if len(valid_scores) > 0 else 0.0
                    else:
                        overall_score = 0.0
                except (IndexError, ValueError):
                    overall_score = 0.0
                
                # PersonPose 생성
                person = PersonPose(
                    person_id=person_idx,
                    bbox=bbox,
                    keypoints=keypoints,
                    score=float(overall_score)
                )
                
                persons.append(person)
                
            except Exception as e:
                logging.warning(f"Error converting pose result {person_idx}: {str(e)}")
                continue
        
        return persons
    
    def _convert_enhanced_poses_to_persons(self, converted_poses, frame_idx):
        """Enhanced extractor의 결과를 PersonPose 형식으로 변환"""
        persons = []
        
        if not converted_poses:
            return persons
        
        for person_idx, pose_data in enumerate(converted_poses):
            try:
                # Enhanced extractor의 표준 형식에서 정보 추출
                keypoints = pose_data.get('keypoints', [])
                bbox = pose_data.get('bbox', [])
                score = pose_data.get('score', 0.0)
                
                if len(keypoints) == 0 or len(bbox) < 4:
                    continue
                
                # keypoints를 numpy 배열로 변환 (17, 3) 형태
                try:
                    keypoints_array = np.array(keypoints)
                    if keypoints_array.ndim == 1 and len(keypoints_array) >= 51:
                        # 평면화된 경우 (51,) -> (17, 3)으로 변형
                        keypoints_array = keypoints_array.reshape(-1, 3)
                    elif keypoints_array.ndim == 2 and keypoints_array.shape[1] != 3:
                        # 잘못된 형태인 경우 건너뛰기
                        logging.warning(f"Invalid keypoints shape for person {person_idx}: {keypoints_array.shape}")
                        continue
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error processing keypoints array for person {person_idx}: {str(e)}")
                    continue
                
                # bbox는 [x1, y1, x2, y2] 형태 (score 제외)
                try:
                    bbox_coords = bbox[:4] if len(bbox) >= 4 else bbox
                    if len(bbox_coords) < 4:
                        logging.warning(f"Invalid bbox for person {person_idx}: {bbox}")
                        continue
                except (IndexError, TypeError):
                    logging.warning(f"Error processing bbox for person {person_idx}: {bbox}")
                    continue
                
                # PersonPose 생성
                person = PersonPose(
                    person_id=person_idx,
                    bbox=bbox_coords,
                    keypoints=keypoints_array,
                    score=float(score)
                )
                
                persons.append(person)
                
            except Exception as e:
                logging.warning(f"Error converting enhanced pose result {person_idx}: {str(e)}")
                continue
        
        return persons
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'model_name': 'RTMO',
            'config_file': self.config_file,
            'checkpoint_file': self.checkpoint_file,
            'device': self.device,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'is_initialized': self.is_initialized,
            'statistics': self.stats.copy()
        }
        
        if self.is_initialized and hasattr(self.model, 'cfg'):
            try:
                info['model_config'] = {
                    'type': getattr(self.model.cfg.model, 'type', 'Unknown'),
                    'backbone': getattr(self.model.cfg.model, 'backbone', {}),
                    'head': getattr(self.model.cfg.model, 'head', {})
                }
            except:
                pass
        
        return info
    
    def set_thresholds(self, score_threshold: Optional[float] = None, 
                      nms_threshold: Optional[float] = None,
                      keypoint_threshold: Optional[float] = None):
        """임계값 설정 및 모델에 적용"""
        super().set_thresholds(score_threshold, nms_threshold, keypoint_threshold)
        
        # 모델이 초기화되어 있으면 설정 적용
        if self.is_initialized:
            self._configure_model()
    
    def cleanup(self):
        """리소스 정리"""
        super().cleanup()
        
        # 모델 해제
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        self.is_initialized = False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['std_processing_time'] = np.std(stats['processing_times'])
            stats['fps_estimate'] = 1.0 / np.mean(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['std_processing_time'] = 0.0
            stats['fps_estimate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_frames_processed': 0,
            'total_persons_detected': 0,
            'avg_persons_per_frame': 0.0,
            'processing_times': []
        }