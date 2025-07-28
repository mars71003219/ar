#!/usr/bin/env python3
"""
Main Pipeline Module
메인 파이프라인 모듈 - 전체 추론 과정을 통합 관리
"""

import os
import os.path as osp
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import json
import time
from pathlib import Path

from config import INFERENCE_CONFIG, validate_config, check_gpu_availability
from pose_estimator import RTMOPoseEstimator
from fight_tracker import FightPrioritizedTracker
from action_classifier import STGCNActionClassifier
from metrics_calculator import MetricsCalculator
from video_overlay import VideoOverlayGenerator

logger = logging.getLogger(__name__)

class EndToEndPipeline:
    """
    엔드투엔드 추론 파이프라인
    RTMO 포즈 추정 → Fight-우선 트래킹 → STGCN++ 분류 → 성능 평가 → 오버레이 생성
    """
    
    def __init__(self, pose_config: str, pose_checkpoint: str,
                 gcn_config: str, gcn_checkpoint: str,
                 device: str = 'cuda:0'):
        """
        초기화
        
        Args:
            pose_config: RTMO 모델 설정 파일
            pose_checkpoint: RTMO 체크포인트
            gcn_config: STGCN++ 설정 파일
            gcn_checkpoint: STGCN++ 체크포인트
            device: 추론 디바이스
        """
        self.device = device
        
        # 모듈 초기화
        logger.info("파이프라인 구성 요소 초기화 중...")
        
        # 1. RTMO 포즈 추정기
        self.pose_estimator = RTMOPoseEstimator(
            pose_config, pose_checkpoint, device
        )
        
        # 2. Fight-우선 트래커 (ByteTrack 기반)
        bytetrack_config = {
            'high_thresh': 0.6,
            'low_thresh': 0.1,
            'max_disappeared': 30,
            'min_hits': 3
        }
        
        self.tracker = FightPrioritizedTracker(
            frame_width=INFERENCE_CONFIG['frame_extraction']['resize'][0],
            frame_height=INFERENCE_CONFIG['frame_extraction']['resize'][1],
            region_weights=INFERENCE_CONFIG['region_weights'],
            composite_weights=INFERENCE_CONFIG['composite_weights'],
            bytetrack_config=bytetrack_config
        )
        
        # 3. STGCN++ 분류기
        self.classifier = STGCNActionClassifier(
            gcn_config, gcn_checkpoint, device
        )
        
        # 4. 메트릭 계산기
        self.metrics_calculator = MetricsCalculator()
        
        # 5. 비디오 오버레이 생성기 (ByteTrack 기반, 사용자 요청 색상)
        self.overlay_generator = VideoOverlayGenerator(
            stgcn_joint_color=(0, 255, 128),    # 연녹색 형광 (BGR)
            stgcn_skeleton_color=(0, 255, 128), # 연녹색 형광
            other_joint_color=(255, 0, 0),      # 파란색 (BGR)
            other_skeleton_color=(255, 0, 0),   # 파란색
            text_color=INFERENCE_CONFIG['overlay']['text_color'],
            font_scale=INFERENCE_CONFIG['overlay']['font_scale'],
            thickness=INFERENCE_CONFIG['overlay']['thickness'],
            point_radius=INFERENCE_CONFIG['overlay']['point_radius']
        )
        
        logger.info("파이프라인 구성 요소 초기화 완료")
    
    def process_single_video(self, video_path: str, ground_truth_label: Optional[int] = None,
                           generate_overlay: bool = True) -> Dict:
        """
        단일 비디오 처리
        
        Args:
            video_path: 비디오 파일 경로
            ground_truth_label: 실제 라벨 (0: NonFight, 1: Fight)
            generate_overlay: 오버레이 비디오 생성 여부
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        video_name = osp.basename(video_path)
        
        try:
            # 1. RTMO 포즈 추정
            pose_results = self.pose_estimator.estimate_poses_from_video(
                video_path, 
                max_frames=INFERENCE_CONFIG['frame_extraction']['max_frames']
            )
            
            if not pose_results:
                return self._create_empty_result(video_path, ground_truth_label)
            
            # 2. Fight-우선 트래킹 및 정렬 (ByteTrack 기반)
            self.tracker.reset()  # 트래커 상태 초기화
            
            # 포즈 결과에서 detection 데이터 추출
            detections_list = []
            keypoints_list = []
            scores_list = []
            
            for keypoints, scores in pose_results:
                if keypoints is not None and scores is not None and len(keypoints) > 0:
                    # Detection 데이터 생성
                    keypoints_frame_list = [keypoints[i] for i in range(len(keypoints))]
                    scores_frame_list = [scores[i] for i in range(len(scores))]
                    
                    detections = self.tracker.create_detections_from_pose_results(
                        keypoints_frame_list, scores_frame_list
                    )
                    detections_list.append(detections)
                    keypoints_list.append(keypoints_frame_list)
                    scores_list.append(scores_frame_list)
                else:
                    # 빈 프레임
                    detections_list.append(np.empty((0, 5)))
                    keypoints_list.append([])
                    scores_list.append([])
            
            # Fight-우선 트래킹 (ByteTrack 기반, STGCN++와 동기화)
            if detections_list and any(len(det) > 0 for det in detections_list):
                selected_keypoints, selected_scores, track_ids_per_frame = self.tracker.process_video_sequence_with_detections(
                    detections_list, keypoints_list, scores_list,
                    sequence_length=INFERENCE_CONFIG['sequence_length'],  # 윈도우 크기
                    num_person=INFERENCE_CONFIG['num_person']
                )
            else:
                # 빈 비디오 처리
                num_person = INFERENCE_CONFIG['num_person']
                seq_len = INFERENCE_CONFIG['sequence_length'] or 30
                selected_keypoints = np.zeros((seq_len, num_person, 17, 2))
                selected_scores = np.zeros((seq_len, num_person, 17))
                track_ids_per_frame = [[-1] * num_person for _ in range(seq_len)]
            
            # 3. STGCN++ 분류
            img_shape = INFERENCE_CONFIG['frame_extraction']['resize'][::-1]  # (W, H) -> (H, W)
            
            classification_result = self.classifier.classify_video_sequence(
                selected_keypoints, selected_scores, 
                window_size=INFERENCE_CONFIG['sequence_length'],  # 설정 기반 윈도우 크기
                stride=INFERENCE_CONFIG['sequence_length'] // 2,  # 설정 기반 stride
                img_shape=img_shape
            )
            
            # 4. 신뢰도 분석
            confidence_analysis = self.classifier.analyze_prediction_confidence(classification_result)
            
            # 5. 오버레이 비디오 생성 (옵션)
            overlay_path = None
            if generate_overlay and INFERENCE_CONFIG['overlay']['enabled']:
                overlay_path = self._generate_overlay_video(
                    video_path, pose_results, classification_result, ground_truth_label
                )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'video_path': video_path,
                'video_name': video_name,
                'ground_truth_label': ground_truth_label,
                'pose_estimation': {
                    'total_frames': len(pose_results),
                    'valid_frames': self.pose_estimator.get_valid_poses_count(pose_results)
                },
                'tracking': {
                    'sequence_length': len(selected_keypoints),
                    'selected_keypoints_shape': list(selected_keypoints.shape),
                    'selected_scores_shape': list(selected_scores.shape),
                    'track_ids_per_frame': track_ids_per_frame,
                    'unique_track_ids': list(set([tid for frame_tids in track_ids_per_frame for tid in frame_tids if tid >= 0])),
                    'total_tracks': len(set([tid for frame_tids in track_ids_per_frame for tid in frame_tids if tid >= 0]))
                },
                'classification': classification_result,
                'confidence_analysis': confidence_analysis,
                'overlay_path': overlay_path,
                'processing_time': processing_time,
                'status': 'success'
            }
            
            logger.info(f"비디오 처리 완료: {video_name} ({processing_time:.2f}초)")
            return result
            
        except Exception as e:
            logger.error(f"비디오 처리 실패: {video_name} - {e}")
            return {
                'video_path': video_path,
                'video_name': video_name,
                'ground_truth_label': ground_truth_label,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'status': 'failed'
            }
    
    def process_batch_videos(self, video_paths: List[str], ground_truth_labels: Optional[List[int]] = None,
                           generate_overlay: bool = True, save_individual_results: bool = True,
                           output_dir: str = "./results") -> Dict:
        """
        배치 비디오 처리
        
        Args:
            video_paths: 비디오 파일 경로 리스트
            ground_truth_labels: 실제 라벨 리스트
            generate_overlay: 오버레이 비디오 생성 여부
            save_individual_results: 개별 결과 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            전체 배치 처리 결과
        """
        start_time = time.time()
        
        logger.info(f"배치 처리 시작: {len(video_paths)}개 비디오")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 개별 결과 저장
        individual_results = []
        successful_results = []
        failed_results = []
        
        for i, video_path in enumerate(video_paths):
            gt_label = ground_truth_labels[i] if ground_truth_labels else None
            
            # 진행 바 표시
            progress = int((i / len(video_paths)) * 100)
            print(f"\r처리 중: {i+1}/{len(video_paths)} [{progress}%] - {osp.basename(video_path)[:30]}", end="", flush=True)
            
            result = self.process_single_video(video_path, gt_label, generate_overlay)
            
            # 처리 결과에 따른 상태 표시
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            print(f"\r처리 완료: {i+1}/{len(video_paths)} [{progress}%] {status_symbol} {osp.basename(video_path)[:25]}")  # 줄바꿈으로 로그와 분리
            individual_results.append(result)
            
            if result['status'] == 'success':
                successful_results.append(result)
            else:
                failed_results.append(result)
            
            # 개별 결과 저장
            if save_individual_results:
                video_name = osp.splitext(osp.basename(video_path))[0]
                result_path = osp.join(output_dir, f"{video_name}_result.json")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        print()  # 진행 바 완료 후 줄바꿈
        
        # 전체 성능 메트릭 계산
        performance_metrics = None
        if ground_truth_labels and successful_results:
            logger.info("전체 성능 메트릭 계산 중...")
            performance_metrics = self._calculate_batch_metrics(successful_results, ground_truth_labels)
        
        # 전체 처리 시간
        total_time = time.time() - start_time
        
        # 배치 결과 구성
        batch_result = {
            'summary': {
                'total_videos': len(video_paths),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(video_paths) if video_paths else 0.0,
                'total_processing_time': total_time,
                'average_time_per_video': total_time / len(video_paths) if video_paths else 0.0
            },
            'performance_metrics': performance_metrics,
            'individual_results': individual_results,
            'failed_videos': [r['video_name'] for r in failed_results],
            'config': INFERENCE_CONFIG
        }
        
        # 전체 결과 저장
        batch_result_path = osp.join(output_dir, "batch_results.json")
        with open(batch_result_path, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"배치 처리 완료: {len(successful_results)}/{len(video_paths)}개 성공 ({total_time:.2f}초)")
        
        return batch_result
    
    def _create_empty_result(self, video_path: str, ground_truth_label: Optional[int]) -> Dict:
        """빈 결과 생성"""
        return {
            'video_path': video_path,
            'video_name': osp.basename(video_path),
            'ground_truth_label': ground_truth_label,
            'classification': {
                'prediction': 0,
                'confidence': 0.0,
                'prediction_label': 'NonFight'
            },
            'processing_time': 0.0,
            'status': 'empty'
        }
    
    def _generate_overlay_video(self, video_path: str, pose_results: List, 
                              classification_result: Dict, ground_truth_label: Optional[int]) -> Optional[str]:
        """ByteTrack 기반 오버레이 비디오 생성"""
        try:
            video_name = osp.splitext(osp.basename(video_path))[0]
            overlay_dir = "/workspace/rtmo_gcn_pipeline/inference_pipeline/results/overlays"
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_path = osp.join(overlay_dir, f"{video_name}_bytetrack_overlay.mp4")
            
            # ByteTrack 기반 오버레이 생성 (사용자 요청에 따른 색상 구분)
            success = self.overlay_generator.create_overlay_with_bytetrack(
                video_path=video_path,
                pose_results=pose_results,
                classification_result=classification_result,
                output_path=overlay_path,
                num_person=INFERENCE_CONFIG['num_person'],
                tracker=self.tracker  # main pipeline의 tracker 재사용
            )
            
            return overlay_path if success else None
            
        except Exception as e:
            logger.warning(f"ByteTrack 오버레이 생성 실패: {e}")
            return None
    
    def _calculate_batch_metrics(self, successful_results: List[Dict], 
                               ground_truth_labels: List[int]) -> Dict:
        """배치 성능 메트릭 계산"""
        try:
            predictions = []
            confidences = []
            video_names = []
            actual_labels = []
            
            # 성공한 결과들에서 데이터 추출
            for result in successful_results:
                video_path = result['video_path']
                gt_label = result.get('ground_truth_label')
                
                if gt_label is not None:
                    predictions.append(result['classification']['prediction'])
                    confidences.append(result['classification']['confidence'])
                    video_names.append(result['video_name'])
                    actual_labels.append(gt_label)
            
            if predictions:
                return self.metrics_calculator.calculate_comprehensive_metrics(
                    predictions, actual_labels, confidences, video_names
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"배치 메트릭 계산 실패: {e}")
            return None
    
    def generate_comprehensive_report(self, batch_result: Dict, output_dir: str):
        """종합 분석 보고서 생성"""
        try:
            if batch_result.get('performance_metrics'):
                # 성능 보고서
                report_path = osp.join(output_dir, "performance_report.md")
                self.metrics_calculator.generate_report(
                    batch_result['performance_metrics'], report_path
                )
                
                # 메트릭 상세 저장
                metrics_path = osp.join(output_dir, "detailed_metrics.json")
                self.metrics_calculator.save_results(
                    batch_result['performance_metrics'], metrics_path
                )
                
                logger.info(f"종합 보고서 생성 완료: {report_path}")
            
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        logger.info("파이프라인 리소스 정리 중...")
        # 필요한 경우 GPU 메모리 정리
        if hasattr(self, 'pose_estimator'):
            del self.pose_estimator
        if hasattr(self, 'classifier'):
            del self.classifier
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("리소스 정리 완료")