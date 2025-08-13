#!/usr/bin/env python3
"""
윈도우 처리 로직 - 비디오 프레임을 윈도우 단위로 처리하는 클래스
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .pose_extractor import EnhancedRTMOPoseExtractor
from .scoring_system import EnhancedFightInvolvementScorer


class WindowProcessor:
    """윈도우 기반 비디오 처리 클래스"""
    
    def __init__(self, config, device='cuda:0', 
                 clip_len=64, stride=10,
                 track_high_thresh=0.6, track_low_thresh=0.1,
                 track_max_disappeared=30, track_min_hits=3,
                 quality_threshold=0.3, min_track_length=10,
                 weights=None):
        """
        Args:
            config: 설정 객체
            device: 추론에 사용할 디바이스
            clip_len: 윈도우 길이 (프레임 수)
            stride: 윈도우 간격
            track_high_thresh: ByteTracker 높은 임계값
            track_low_thresh: ByteTracker 낮은 임계값
            track_max_disappeared: 트랙 최대 소실 프레임
            track_min_hits: 트랙 최소 히트 수
            quality_threshold: 품질 임계값
            min_track_length: 최소 트랙 길이
            weights: 복합점수 가중치
        """
        self.device = device
        self.clip_len = clip_len
        self.stride = stride
        
        # 트래킹 파라미터
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_max_disappeared = track_max_disappeared
        self.track_min_hits = track_min_hits
        self.quality_threshold = quality_threshold
        self.min_track_length = min_track_length
        self.weights = weights or [0.45, 0.10, 0.30, 0.10, 0.05]
        
        # 포즈 추출기 초기화
        self.pose_extractor = EnhancedRTMOPoseExtractor(
            config_file=config.config_file,
            checkpoint_file=config.checkpoint_file,
            device=device,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            track_max_disappeared=track_max_disappeared,
            track_min_hits=track_min_hits,
            quality_threshold=quality_threshold,
            min_track_length=min_track_length,
            weights=weights
        )
    
    def process_video(self, video_path: str, output_dir: str = None, input_dir: str = None) -> Dict[str, Any]:
        """전체 비디오를 윈도우 단위로 처리"""
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Processing video: {video_name}")
            
            # 1. 전체 비디오에서 포즈 추출
            all_pose_results = self._extract_full_video_poses(video_path, output_dir)
            if not all_pose_results:
                return self._create_failure_analysis(
                    video_path,
                    "POSE_EXTRACTION_EMPTY",
                    "포즈 추출 결과가 비어있음",
                    {"stage": "pose_extraction", "poses_detected": 0}
                )
            
            # 2. 윈도우별 처리
            windows_data = self._process_windows_with_tracking(
                all_pose_results, video_path, output_dir, input_dir, self.stride
            )
            
            if not windows_data:
                return self._create_failure_analysis(
                    video_path,
                    "WINDOW_PROCESSING_FAILED", 
                    "윈도우 처리 실패 - 유효한 윈도우 생성되지 않음",
                    {"stage": "window_processing", "total_frames": len(all_pose_results), "windows_generated": 0}
                )
            
            # 3. 결과 구성
            video_result = {
                'video_name': video_name,
                'video_path': video_path,
                'total_frames': len(all_pose_results),
                'num_windows': len(windows_data),
                'windows': windows_data
            }
            
            return video_result
            
        except Exception as e:
            import traceback
            video_name = os.path.basename(video_path)
            failure_analysis = self._analyze_video_failure(video_path, e, traceback.format_exc())
            print(f"비디오 처리 실패: {video_name}")
            print(f"  실패 단계: {failure_analysis.get('failure_stage', 'UNKNOWN')}")
            print(f"  원인: {failure_analysis.get('root_cause', 'Unknown error')}")
            return failure_analysis
    
    def _extract_full_video_poses(self, video_path: str, output_dir: str = None):
        """전체 비디오에서 포즈 추출"""
        try:
            pose_results = self.pose_extractor.extract_poses_only(video_path)
            
            if not pose_results:
                return self._create_failure_analysis(
                    video_path,
                    "POSE_EXTRACTION_EMPTY",
                    "RTMO 포즈 추출 결과 비어있음 - 비디오에서 사람이 검출되지 않음",
                    {
                        "stage": "pose_extraction",
                        "extractor_device": self.device,
                        "poses_detected": 0,
                        "video_accessible": True
                    }
                )
                
            return pose_results
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "POSE_EXTRACTION_EXCEPTION",
                f"포즈 추출 중 예외 발생: {str(e)}",
                {
                    "stage": "pose_extraction",
                    "extractor_device": self.device,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _process_windows_with_tracking(self, all_pose_results, video_path, output_dir, input_dir, stride=10):
        """윈도우별 트래킹 처리"""
        try:
            total_frames = len(all_pose_results)
            windows_data = []
            failed_windows = []
            total_windows = len(range(0, total_frames - self.clip_len + 1, stride))
            
            print(f"Processing {total_windows} windows for {total_frames} frames (stride={stride})")
            
            for window_idx, start_frame in enumerate(range(0, total_frames - self.clip_len + 1, stride)):
                end_frame = min(start_frame + self.clip_len, total_frames)
                window_pose_results = all_pose_results[start_frame:end_frame]
                
                window_data = self._process_single_window(
                    window_pose_results, 
                    window_idx, 
                    start_frame, 
                    end_frame,
                    video_path,
                    output_dir,
                    input_dir
                )
                
                if window_data:
                    if isinstance(window_data, dict) and 'failure_stage' in window_data:
                        # 개별 윈도우 실패를 기록하되 전체 처리는 계속
                        failed_windows.append({
                            'window_idx': window_idx,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'failure_reason': window_data.get('failure_reason', 'Unknown'),
                            'failure_stage': window_data.get('failure_stage', 'Unknown')
                        })
                        print(f"Window {window_idx} ({start_frame}-{end_frame}) failed: {window_data.get('failure_reason', 'Unknown')}")
                    else:
                        windows_data.append(window_data)
                        print(f"Window {window_idx} ({start_frame}-{end_frame}) processed successfully")
            
            # 결과 요약 및 상태 결정
            success_rate = len(windows_data) / total_windows if total_windows > 0 else 0
            print(f"Window processing completed: {len(windows_data)}/{total_windows} successful ({success_rate:.1%})")
            
            if len(failed_windows) > 0:
                print(f"Failed windows summary:")
                for failed in failed_windows:
                    print(f"  - Window {failed['window_idx']}: {failed['failure_reason']}")
            
            # 성공률이 최소 기준을 만족하는지 확인
            if success_rate < 0.3:  # 30% 미만 성공시 실패로 간주
                return self._create_failure_analysis(
                    video_path,
                    "WINDOW_PROCESSING_FAILED",
                    f"윈도우 처리 성공률이 너무 낮음: {success_rate:.1%} (최소 30% 필요)",
                    {
                        "stage": "window_processing",
                        "total_windows": total_windows,
                        "successful_windows": len(windows_data),
                        "failed_windows": len(failed_windows),
                        "success_rate": success_rate,
                        "failed_details": failed_windows
                    }
                )
            
            return windows_data
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "WINDOW_PROCESSING_EXCEPTION",
                f"윈도우 처리 중 예외 발생: {str(e)}",
                {
                    "stage": "window_processing",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _process_single_window(self, window_pose_results, window_idx, start_frame, end_frame, 
                             video_path, output_dir, input_dir):
        """단일 윈도우 처리"""
        try:
            # 포즈 추출기의 트래킹 기능 사용
            window_result = self.pose_extractor.apply_tracking_to_poses(
                window_pose_results, start_frame, end_frame, window_idx
            )
            
            if not window_result:
                return self._create_failure_analysis(
                    video_path,
                    "TRACKING_FAILED",
                    f"윈도우 {window_idx} 트래킹 실패",
                    {
                        "stage": "single_window_processing",
                        "window_idx": window_idx,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frames_processed": len(window_pose_results)
                    }
                )
            
            return window_result
            
        except Exception as e:
            import traceback
            return self._create_failure_analysis(
                video_path,
                "WINDOW_PROCESSING_EXCEPTION",
                f"윈도우 {window_idx} 처리 중 예외: {str(e)}",
                {
                    "stage": "single_window_processing",
                    "window_idx": window_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    
    def _create_failure_analysis(self, video_path: str, failure_stage: str, 
                               failure_reason: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """실패 분석 결과 생성"""
        video_name = os.path.basename(video_path)
        
        return {
            'video_name': video_name,
            'video_path': video_path,
            'success': False,
            'failure_stage': failure_stage,
            'failure_reason': failure_reason,
            'root_cause': failure_reason,
            'metadata': metadata,
            'timestamp': None,
            'error_details': {
                'stage': metadata.get('stage', 'unknown'),
                'type': metadata.get('exception_type', 'ProcessingError'),
                'message': failure_reason
            }
        }
    
    def _analyze_video_failure(self, video_path: str, exception: Exception, 
                             traceback_str: str) -> Dict[str, Any]:
        """비디오 처리 실패 분석"""
        video_name = os.path.basename(video_path)
        exception_type = type(exception).__name__
        
        # 실패 원인 분석
        traceback_lower = traceback_str.lower()
        
        if 'cuda' in str(exception).lower() or 'gpu' in str(exception).lower():
            failure_stage = "GPU_ERROR"
            root_cause = "GPU 메모리 부족 또는 CUDA 오류"
        elif 'memory' in str(exception).lower():
            failure_stage = "MEMORY_ERROR"
            root_cause = "메모리 부족"
        elif 'opencv' in traceback_lower or 'video' in traceback_lower:
            failure_stage = "VIDEO_ERROR"
            root_cause = "비디오 파일 읽기 오류"
        elif 'pose_extraction' in traceback_lower:
            failure_stage = "POSE_EXTRACTION_FAILED"
            root_cause = "포즈 추출 실패"
        elif 'window_processing' in traceback_lower:
            failure_stage = "WINDOW_PROCESSING_FAILED"
            root_cause = "윈도우 트래킹 실패"
        else:
            failure_stage = "UNKNOWN_ERROR"
            root_cause = f"알 수 없는 오류: {str(exception)}"
        
        return {
            'video_name': video_name,
            'video_path': video_path,
            'success': False,
            'failure_stage': failure_stage,
            'failure_reason': str(exception),
            'root_cause': root_cause,
            'metadata': {
                'exception_type': exception_type,
                'exception_message': str(exception),
                'traceback': traceback_str
            },
            'error_details': {
                'stage': 'video_processing',
                'type': exception_type,
                'message': str(exception)
            }
        }
    
    def sort_windows_by_composite_score(self, windows_data):
        """윈도우를 복합점수 기준으로 정렬"""
        def get_max_composite_score(window):
            """윈도우의 최대 복합점수 추출"""
            try:
                annotation = window.get('annotation', {})
                persons = annotation.get('persons', {})
                
                if not persons:
                    return 0.0
                
                max_score = 0.0
                for person_data in persons.values():
                    score = person_data.get('composite_score', 0.0)
                    if isinstance(score, (int, float)):
                        max_score = max(max_score, float(score))
                    elif hasattr(score, 'item'):  # numpy scalar
                        max_score = max(max_score, float(score.item()))
                
                return max_score
            except Exception as e:
                print(f"점수 추출 오류: {str(e)}")
                return 0.0
        
        try:
            # 복합점수 기준으로 내림차순 정렬
            sorted_windows = sorted(
                windows_data, 
                key=get_max_composite_score, 
                reverse=True
            )
            return sorted_windows
        except Exception as e:
            print(f"윈도우 정렬 오류: {str(e)}")
            return windows_data