#!/usr/bin/env python3
"""
RTMO-m Pose Estimation Module
RTMO-m 포즈 추정 모듈 - 학습된 모델을 사용한 실시간 포즈 추정
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional
import logging
import os
import warnings

# OpenCV 경고 메시지 억제 (버전 호환성 고려)
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    # 이전 OpenCV 버전에서는 해당 메서드가 없음
    pass
warnings.filterwarnings("ignore", category=UserWarning)

from mmpose.apis import init_model as init_pose_model, inference_bottomup
from mmaction.utils import frame_extract
import tempfile

logger = logging.getLogger(__name__)

class RTMOPoseEstimator:
    """
    RTMO-m 포즈 추정기
    학습된 RTMO-m 모델을 사용하여 비디오에서 포즈를 추정
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
        """
        초기화
        
        Args:
            config_path: RTMO 모델 설정 파일 경로
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 추론 디바이스
        """
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 모델 로드
        logger.info("RTMO-m 모델 로딩 중...")
        self.model = self._load_model()
        logger.info("RTMO-m 모델 로딩 완료")
        
    def _load_model(self):
        """RTMO 모델 로드"""
        try:
            model = init_pose_model(
                self.config_path, 
                self.checkpoint_path, 
                device=self.device
            )
            return model
        except Exception as e:
            logger.error(f"RTMO 모델 로드 실패: {e}")
            raise
    
    def extract_frames_from_video(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        비디오에서 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            max_frames: 최대 프레임 수 (None이면 전체)
            
        Returns:
            추출된 프레임 리스트
        """
        frames = []
        
        try:
            # 먼저 비디오 파일 존재 확인
            if not os.path.exists(video_path):
                logger.error(f"비디오 파일이 존재하지 않습니다: {video_path}")
                return frames
            
            # 임시 디렉토리에 프레임 추출
            with tempfile.TemporaryDirectory() as temp_dir:
                result = frame_extract(video_path, out_dir=temp_dir)
                
                # frame_extract는 tuple을 반환하며, 첫 번째 요소가 프레임 경로 리스트
                if isinstance(result, tuple) and len(result) > 0:
                    frame_paths = result[0]
                else:
                    frame_paths = result
                
                if not frame_paths:
                    logger.error(f"frame_extract가 빈 결과를 반환했습니다: {video_path}")
                    return frames
                
                if max_frames:
                    frame_paths = frame_paths[:max_frames]
                
                for frame_path in frame_paths:
                    # 경로를 문자열로 변환
                    frame_path_str = str(frame_path)
                    if os.path.exists(frame_path_str):
                        frame = cv2.imread(frame_path_str)
                        if frame is not None:
                            frames.append(frame)
                        
        except Exception as e:
            logger.error(f"프레임 추출 실패: {e}")
            return frames  # raise 대신 빈 리스트 반환
        
        return frames
    
    def estimate_poses_single_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        단일 프레임에서 포즈 추정
        
        Args:
            frame: 입력 프레임 (H, W, C)
            
        Returns:
            (keypoints, scores) - 키포인트 좌표와 신뢰도 점수
            keypoints: (N, 17, 2) - N명의 17개 키포인트 좌표
            scores: (N, 17) - N명의 17개 키포인트 신뢰도
        """
        try:
            result = inference_bottomup(self.model, frame)
            return self._extract_pose_data(result)
        except Exception as e:
            logger.warning(f"단일 프레임 포즈 추정 실패: {e}")
            return None, None
    
    def estimate_poses_batch(self, frames: List[np.ndarray]) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        배치 프레임에서 포즈 추정
        
        Args:
            frames: 입력 프레임 리스트
            
        Returns:
            포즈 결과 리스트 [(keypoints, scores), ...]
        """
        pose_results = []
        
        for i, frame in enumerate(frames):
            keypoints, scores = self.estimate_poses_single_frame(frame)
            pose_results.append((keypoints, scores))
        return pose_results
    
    def _extract_pose_data(self, result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RTMO 결과에서 포즈 데이터 추출
        
        Args:
            result: RTMO 추론 결과
            
        Returns:
            (keypoints, scores) 또는 (None, None)
        """
        try:
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if hasattr(result, 'pred_instances') and len(result.pred_instances) > 0:
                instances = result.pred_instances
                
                keypoints_list = []
                scores_list = []
                
                for instance in instances:
                    if hasattr(instance, 'keypoints') and hasattr(instance, 'keypoint_scores'):
                        # 키포인트 좌표 추출
                        kpts = instance.keypoints
                        if len(kpts.shape) > 2:
                            kpts = kpts[0]  # (1, 17, 2) -> (17, 2)
                        
                        # 키포인트 점수 추출
                        scrs = instance.keypoint_scores
                        if len(scrs.shape) > 1:
                            scrs = scrs[0]  # (1, 17) -> (17,)
                        
                        # numpy 배열로 변환
                        if hasattr(kpts, 'cpu'):
                            kpts = kpts.cpu().numpy()
                        if hasattr(scrs, 'cpu'):
                            scrs = scrs.cpu().numpy()
                        
                        keypoints_list.append(kpts)
                        scores_list.append(scrs)
                
                if keypoints_list:
                    return np.array(keypoints_list), np.array(scores_list)
            
            return None, None
            
        except Exception as e:
            logger.warning(f"포즈 데이터 추출 실패: {e}")
            return None, None
    
    def estimate_poses_from_video(self, video_path: str, max_frames: Optional[int] = None) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        비디오 파일에서 직접 포즈 추정
        
        Args:
            video_path: 비디오 파일 경로
            max_frames: 최대 프레임 수
            
        Returns:
            포즈 결과 리스트
        """
        # 프레임 추출
        frames = self.extract_frames_from_video(video_path, max_frames)
        
        if not frames:
            logger.error("비디오에서 프레임을 추출할 수 없습니다")
            return []
        
        # 포즈 추정
        pose_results = self.estimate_poses_batch(frames)
        
        return pose_results
    
    def get_valid_poses_count(self, pose_results: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]) -> int:
        """유효한 포즈가 검출된 프레임 수 반환"""
        valid_count = 0
        for keypoints, scores in pose_results:
            if keypoints is not None and scores is not None and len(keypoints) > 0:
                valid_count += 1
        return valid_count
    
    def filter_low_confidence_poses(self, keypoints: np.ndarray, scores: np.ndarray, 
                                  threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        낮은 신뢰도의 키포인트 필터링
        
        Args:
            keypoints: 키포인트 좌표 (N, 17, 2)
            scores: 키포인트 신뢰도 (N, 17)
            threshold: 신뢰도 임계값
            
        Returns:
            필터링된 (keypoints, scores)
        """
        # 낮은 신뢰도 키포인트를 0으로 설정
        mask = scores < threshold
        filtered_keypoints = keypoints.copy()
        filtered_keypoints[mask] = 0
        
        return filtered_keypoints, scores