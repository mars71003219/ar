#!/usr/bin/env python3
"""
Optimized RTMO Pose Estimator
최적화된 RTMO 포즈 추정기 - 배치 처리 및 성능 개선
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

# OpenCV 경고 메시지 억제
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)

from mmpose.apis import init_model as init_pose_model, inference_bottomup

logger = logging.getLogger(__name__)

class OptimizedRTMOPoseEstimator:
    """
    최적화된 RTMO-m 포즈 추정기
    - 배치 처리로 GPU 활용도 극대화
    - 해상도 적응형 추론
    - 멀티스레드 전처리
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda:0',
                 batch_size: int = 8, target_fps: int = 30):
        """
        초기화
        
        Args:
            config_path: RTMO 모델 설정 파일 경로
            checkpoint_path: 학습된 모델 체크포인트 경로
            device: 추론 디바이스
            batch_size: 배치 처리 크기
            target_fps: 목표 FPS (해상도 자동 조정용)
        """
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.target_fps = target_fps
        
        # 모델 로드
        logger.info(f"최적화된 RTMO-m 모델 로딩 중... (배치크기: {batch_size})")
        self.model = self._load_model()
        
        # GPU 워밍업
        self._warmup_model()
        
        logger.info("최적화된 RTMO-m 모델 로딩 완료")
    
    def _load_model(self):
        """모델 로드"""
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
    
    def _warmup_model(self):
        """GPU 워밍업 - 첫 추론 시 지연 방지"""
        logger.info("GPU 워밍업 중...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            _ = inference_bottomup(self.model, dummy_image)
            logger.info("GPU 워밍업 완료")
        except Exception as e:
            logger.warning(f"GPU 워밍업 실패: {e}")
    
    def _adaptive_resize(self, frame: np.ndarray, target_fps: int = 30) -> np.ndarray:
        """
        적응형 해상도 조정
        목표 FPS에 따라 해상도를 동적으로 조정
        """
        h, w = frame.shape[:2]
        
        # 목표 FPS에 따른 해상도 스케일링
        if target_fps >= 30:
            # 고성능 모드: 원본 해상도
            scale_factor = 1.0
        elif target_fps >= 20:
            # 균형 모드: 80% 해상도
            scale_factor = 0.8
        else:
            # 효율 모드: 60% 해상도
            scale_factor = 0.6
        
        if scale_factor < 1.0:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            # 8의 배수로 조정 (모델 최적화)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            frame = cv2.resize(frame, (new_w, new_h))
            
        return frame
    
    def _preprocess_frames_parallel(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """멀티스레드 전처리"""
        def preprocess_single(frame):
            # 적응형 해상도 조정
            frame = self._adaptive_resize(frame, self.target_fps)
            return frame
        
        # ThreadPoolExecutor로 병렬 전처리
        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_frames = list(executor.map(preprocess_single, frames))
        
        return processed_frames
    
    def estimate_poses_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        배치 포즈 추정
        
        Args:
            frames: 입력 프레임 리스트
            
        Returns:
            [(keypoints, scores), ...] 포즈 결과 리스트
        """
        if not frames:
            return []
        
        # 전처리 (병렬)
        processed_frames = self._preprocess_frames_parallel(frames)
        
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(processed_frames), self.batch_size):
            batch_frames = processed_frames[i:i + self.batch_size]
            batch_results = []
            
            # GPU 동기화
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 배치 내 개별 처리 (MMPose의 배치 지원 한계로 인해)
            for frame in batch_frames:
                try:
                    result = inference_bottomup(self.model, frame)
                    
                    if result and len(result.pred_instances) > 0:
                        keypoints = result.pred_instances.keypoints.cpu().numpy()
                        scores = result.pred_instances.keypoint_scores.cpu().numpy()
                        batch_results.append((keypoints, scores))
                    else:
                        # 빈 결과
                        batch_results.append((np.empty((0, 17, 2)), np.empty((0, 17))))
                        
                except Exception as e:
                    logger.warning(f"프레임 추론 실패: {e}")
                    batch_results.append((np.empty((0, 17, 2)), np.empty((0, 17))))
            
            results.extend(batch_results)
        
        return results
    
    def estimate_poses_from_video_optimized(self, video_path: str, 
                                          max_frames: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        최적화된 비디오 포즈 추정
        
        Args:
            video_path: 비디오 파일 경로
            max_frames: 최대 처리 프레임 수
            
        Returns:
            포즈 추정 결과 리스트
        """
        logger.info(f"최적화된 비디오 포즈 추정 시작: {video_path}")
        start_time = time.time()
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {video_path}")
            return []
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"비디오 정보: {total_frames}프레임, {fps:.1f} FPS")
        
        # 프레임 읽기 (메모리 효율성을 위해 청크 단위로)
        chunk_size = self.batch_size * 4  # 배치 크기의 4배씩 처리
        all_results = []
        
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            frames = []
            
            # 청크 프레임 읽기
            for frame_idx in range(chunk_start, chunk_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"프레임 {frame_idx} 읽기 실패")
                    continue
                    
                frames.append(frame)
            
            if not frames:
                continue
            
            # 배치 포즈 추정
            chunk_results = self.estimate_poses_batch(frames)
            all_results.extend(chunk_results)
            
            # 진행 상황 로깅
            progress = (chunk_end / total_frames) * 100
            elapsed = time.time() - start_time
            current_fps = chunk_end / elapsed if elapsed > 0 else 0
            
            logger.info(f"진행률: {progress:.1f}% ({chunk_end}/{total_frames}), "
                       f"현재 FPS: {current_fps:.1f}")
        
        cap.release()
        
        # 최종 성능 리포트
        total_time = time.time() - start_time
        final_fps = len(all_results) / total_time if total_time > 0 else 0
        
        logger.info(f"최적화된 포즈 추정 완료: {len(all_results)}프레임, "
                   f"{total_time:.2f}초, {final_fps:.1f} FPS")
        
        return all_results
    
    def get_valid_poses_count(self, pose_results: List[Tuple]) -> int:
        """유효한 포즈 개수 반환"""
        return sum(1 for keypoints, scores in pose_results if len(keypoints) > 0)

# 성능 비교 함수
def compare_performance(original_estimator, optimized_estimator, video_path: str):
    """원본 vs 최적화 버전 성능 비교"""
    print("🚀 포즈 추정 성능 비교 테스트")
    
    # 원본 버전
    print("1. 원본 RTMO 추정기 테스트...")
    start_time = time.time()
    original_results = original_estimator.estimate_poses_from_video(video_path, max_frames=100)
    original_time = time.time() - start_time
    original_fps = len(original_results) / original_time if original_time > 0 else 0
    
    # 최적화 버전
    print("2. 최적화된 RTMO 추정기 테스트...")
    start_time = time.time()
    optimized_results = optimized_estimator.estimate_poses_from_video_optimized(video_path, max_frames=100)
    optimized_time = time.time() - start_time
    optimized_fps = len(optimized_results) / optimized_time if optimized_time > 0 else 0
    
    # 결과 비교
    speedup = optimized_fps / original_fps if original_fps > 0 else 0
    
    print(f"\n📊 성능 비교 결과:")
    print(f"원본:     {original_fps:.1f} FPS ({original_time:.2f}초)")
    print(f"최적화:   {optimized_fps:.1f} FPS ({optimized_time:.2f}초)")
    print(f"성능 향상: {speedup:.1f}x 빠름")
    
    return {
        'original_fps': original_fps,
        'optimized_fps': optimized_fps,
        'speedup': speedup,
        'original_time': original_time,
        'optimized_time': optimized_time
    }